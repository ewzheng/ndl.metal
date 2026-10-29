/*
File: ndarray_backend_metal.mm
Authors: Elias Zheng,

Host code (CPU) for metal ndarray_backend

NOTE: Kernels, ops in ndarray_backend_metal.metal
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "ndarray_backend_metal.h"

namespace needle{
namespace metal{
    ////////////////////////////////////////////////////////////////////////////////
    // Metal runtime setup
    ////////////////////////////////////////////////////////////////////////////////

    #define BASE_THREAD_NUM 256

    id<MTLDevice> Device() {
        static id<MTLDevice> device = [] {
            id<MTLDevice> d = MTLCreateSystemDefaultDevice();
            if (d == nil) throw std::runtime_error("no Metal device found");
            return d;
        }();
        return device;
    }

    id<MTLCommandQueue> Queue() {
        static id<MTLCommandQueue> queue = [Device() newCommandQueue];
        return queue;
    }

    // Compiles the shaders on first use. The Metal compiler can't find our header at runtime,
    // so we paste the header's text in place of its #include line before compiling.
    id<MTLLibrary> Library() {
        static id<MTLLibrary> lib = [] {
            NSString* common  = [NSString stringWithContentsOfFile:@NDL_METAL_COMMON_PATH
                                                          encoding:NSUTF8StringEncoding error:nil];
            NSString* kernels = [NSString stringWithContentsOfFile:@NDL_METAL_SHADER_PATH
                                                          encoding:NSUTF8StringEncoding error:nil];
            if (common == nil || kernels == nil) throw std::runtime_error("couldn't read Metal shader files");
            kernels = [kernels stringByReplacingOccurrencesOfString:@"#include \"ndarray_backend_metal.h\""
                                                         withString:common];

            NSError* err = nil;
            id<MTLLibrary> l = [Device() newLibraryWithSource:kernels options:nil error:&err];
            if (l == nil) throw std::runtime_error(err.localizedDescription.UTF8String);
            return l;
        }();
        return lib;
    }

    // Looks up a kernel by its [[host_name]] and builds its pipeline, cached after the first call
    id<MTLComputePipelineState> Pipeline(const std::string& name) {
        static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
        auto it = cache.find(name);
        if (it != cache.end()) return it->second;

        id<MTLFunction> fn = [Library() newFunctionWithName:@(name.c_str())];
        if (fn == nil) throw std::runtime_error("no Metal kernel named '" + name + "'");

        NSError* err = nil;
        id<MTLComputePipelineState> pso = [Device() newComputePipelineStateWithFunction:fn error:&err];
        if (pso == nil) throw std::runtime_error(err.localizedDescription.UTF8String);
        cache[name] = pso;
        return pso;
    }

    // Runs kernel `name` with n threads and blocks until it finishes. `bind` sets the kernel's
    // arguments with setBuffer/setBytes, using the same indices as its [[buffer(i)]] attributes.
    void Launch(const std::string& name, size_t n,
                const std::function<void(id<MTLComputeCommandEncoder>)>& bind) {
        
        // empty array, do nothing
        if (n == 0) return;

        @autoreleasepool{
            // encode kernel
            id<MTLComputePipelineState> pso = Pipeline(name);
            id<MTLCommandBuffer> cmd = [Queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            [enc setComputePipelineState:pso];
            bind(enc);

            // calculate grid and thread
            // TODO: generalize dimension for tiled matmul
            NSUInteger threadGroupSize = pso.maxTotalThreadsPerThreadgroup;
            if (threadGroupSize > n){
                threadGroupSize = n;
            }
            MTLSize groupSize = MTLSizeMake(threadGroupSize, 1, 1);
            MTLSize gridSize = MTLSizeMake(n, 1, 1);

            // dispatch and wait
            [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];

            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            if (cmd.error != nil){
                throw std::runtime_error(cmd.error.localizedDescription.UTF8String);
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Structs & Utilities
    ////////////////////////////////////////////////////////////////////////////////

    const size_t ELEM_SIZE = sizeof(scalar_t);

    struct MetalArray{
        MetalArray(const size_t size){
            buf = [Device() newBufferWithLength:std::max<size_t>(size, 1) * ELEM_SIZE options:MTLResourceStorageModeShared];
            if (buf == nil) throw std::runtime_error("Metal buffer allocation failed");

            // assign ptr and size fields
            this->ptr = (scalar_t*)buf.contents;
            this->size = size;
        }
        size_t ptr_as_int() { return (size_t)ptr; }
        
        id<MTLBuffer> buf; // GPU buffer (use for metal code)
        scalar_t *ptr;     // CPU pointer to buffer contents (host side)
        size_t size;  
    };

    MetalVec VecToMetal(const std::vector<int32_t>& x) {
        MetalVec shape;
        if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded MetalVec supported max dimesions");
        shape.size = x.size();
        for (size_t i = 0; i < x.size(); i++) {
            shape.data[i] = x[i];
        }
        return shape;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Host Launchers
    ////////////////////////////////////////////////////////////////////////////////

    void LaunchBinary(const MetalArray& a, const MetalArray& b, MetalArray *out, const std::string& name){
        uint32_t size = out->size;
        
        Launch(name, out->size, [&](id<MTLComputeCommandEncoder> enc){
            [enc setBuffer:a.buf    offset:0 atIndex:0];
            [enc setBuffer:b.buf    offset:0 atIndex:1];
            [enc setBuffer:out->buf offset:0 atIndex:2];
            [enc setBytes:&size     length:sizeof(size) atIndex:3];
        });
    }

    void LaunchScalar(const MetalArray& a, const scalar_t val, MetalArray *out, const std::string& name){
        uint32_t size = out->size;

        Launch(name, out->size, [&](id<MTLComputeCommandEncoder> enc){
            [enc setBuffer:a.buf    offset:0 atIndex:0];
            [enc setBytes:&val      length:sizeof(val) atIndex:1];
            [enc setBuffer:out->buf offset:0 atIndex:2];
            [enc setBytes:&size     length:sizeof(size) atIndex:3];
        });
    }

    void LaunchUnary(const MetalArray& a, MetalArray *out, const std::string& name){
        uint32_t size = out->size;

        Launch(name, out->size, [&](id<MTLComputeCommandEncoder> enc){
            [enc setBuffer:a.buf    offset:0 atIndex:0];
            [enc setBuffer:out->buf offset:0 atIndex:1];
            [enc setBytes:&size     length:sizeof(size) atIndex:2];
        });
    }


    ////////////////////////////////////////////////////////////////////////////////
    // Operations
    ////////////////////////////////////////////////////////////////////////////////

    // TODO: define operations
    // TODO: Compact (same bindings as EwiseSetitem, launched over out->size)
    void EwiseAdd(const MetalArray& a, const MetalArray& b, MetalArray *out){
        return LaunchBinary(a, b, out, "ewise_add");
    }
    void ScalarAdd(const MetalArray& a, const scalar_t val, MetalArray *out){
        return LaunchScalar(a, val, out, "scalar_add");
    }

    // note: unused rn
    void EwiseSub(const MetalArray& a, const MetalArray& b, MetalArray *out){
        return LaunchBinary(a, b, out, "ewise_sub");
    }
    void ScalarSub(const MetalArray& a, const scalar_t val, MetalArray *out){
        return LaunchScalar(a, val, out, "scalar_sub");
    }

    void EwiseMul(const MetalArray& a, const MetalArray& b, MetalArray *out){
        return LaunchBinary(a, b, out, "ewise_mul");
    }
    void ScalarMul(const MetalArray& a, const scalar_t val, MetalArray *out){
        return LaunchScalar(a, val, out, "scalar_mul");
    }

    // setitem: shape/strides/offset describe the strided view of out being written to
    void EwiseSetitem(const MetalArray& a, MetalArray *out, std::vector<int32_t> shape,
                      std::vector<int32_t> strides, size_t offset){
        uint32_t size32 = a.size;
        uint32_t offset32 = offset;
        MetalVec shape_v = VecToMetal(shape);
        MetalVec strides_v = VecToMetal(strides);

        Launch("ewise_setitem", a.size, [&](id<MTLComputeCommandEncoder> enc){
            [enc setBuffer:a.buf      offset:0 atIndex:0];
            [enc setBuffer:out->buf   offset:0 atIndex:1];
            [enc setBytes:&size32     length:sizeof(size32)    atIndex:2];
            [enc setBytes:&shape_v    length:sizeof(shape_v)   atIndex:3];
            [enc setBytes:&strides_v  length:sizeof(strides_v) atIndex:4];
            [enc setBytes:&offset32   length:sizeof(offset32)  atIndex:5];
        });
    }

    void ScalarSetitem(size_t size, scalar_t val, MetalArray *out, std::vector<int32_t> shape,
                       std::vector<int32_t> strides, size_t offset){
        uint32_t size32 = size;
        uint32_t offset32 = offset;
        MetalVec shape_v = VecToMetal(shape);
        MetalVec strides_v = VecToMetal(strides);

        Launch("scalar_setitem", size, [&](id<MTLComputeCommandEncoder> enc){
            [enc setBytes:&val        length:sizeof(val)       atIndex:0];
            [enc setBuffer:out->buf   offset:0 atIndex:1];
            [enc setBytes:&size32     length:sizeof(size32)    atIndex:2];
            [enc setBytes:&shape_v    length:sizeof(shape_v)   atIndex:3];
            [enc setBytes:&strides_v  length:sizeof(strides_v) atIndex:4];
            [enc setBytes:&offset32   length:sizeof(offset32)  atIndex:5];
        });
    }

    // fill = scalar setitem over the whole array, viewed as 1D with stride 1 and offset 0
    void Fill(MetalArray *out, scalar_t val){
        ScalarSetitem(out->size, val, out, {(int32_t)out->size}, {1}, 0);
    }
}
}


PYBIND11_MODULE(ndarray_backend_metal, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace metal;

  m.attr("__device_name__") = "metal";
  m.attr("__tile_size__") = TILE;

  // Bind MetalArray
  py::class_<MetalArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &MetalArray::size)
      .def("ptr", &MetalArray::ptr_as_int);

    // return numpy array, copying from CPU
    m.def("to_numpy", [](const MetalArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                        size_t offset) {
        std::vector<size_t> numpy_strides = strides;
        std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                    [](size_t& c) { return c * ELEM_SIZE; });

        // copy memory to host
        scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
        if (host_ptr == 0) throw std::bad_alloc();
        std::memcpy(host_ptr, a.ptr, a.size * ELEM_SIZE);

        // return numpy array
        py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
        return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
    });

    // copy numpy array to GPU
    m.def("from_numpy", [](py::array_t<scalar_t> a, MetalArray* out) {
        if((size_t)a.size() != out->size) throw std::runtime_error("from_numpy: size mismatch");
        std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
    });

  // Uncomment each m.def once its function exists. Until then, calling it from Python
  // raises AttributeError: module 'ndarray_backend_metal' has no attribute '...'
    m.def("fill", Fill);
  // m.def("compact", Compact);
    m.def("ewise_setitem", EwiseSetitem);
    m.def("scalar_setitem", ScalarSetitem);
    m.def("ewise_add", EwiseAdd);
    m.def("scalar_add", ScalarAdd);
  // m.def("ewise_sub", EwiseSub);
  // m.def("scalar_sub", ScalarSub);

    m.def("ewise_mul", EwiseMul);
    m.def("scalar_mul", ScalarMul);
  // m.def("ewise_div", EwiseDiv);
  // m.def("scalar_div", ScalarDiv);
  // m.def("scalar_power", ScalarPower);

  // m.def("ewise_maximum", EwiseMaximum);
  // m.def("scalar_maximum", ScalarMaximum);
  // m.def("ewise_eq", EwiseEq);
  // m.def("scalar_eq", ScalarEq);
  // m.def("ewise_ge", EwiseGe);
  // m.def("scalar_ge", ScalarGe);

  // m.def("ewise_log", EwiseLog);
  // m.def("ewise_exp", EwiseExp);
  // m.def("ewise_tanh", EwiseTanh);

  // m.def("matmul", Matmul);

  // m.def("reduce_max", ReduceMax);
  // m.def("reduce_sum", ReduceSum);
}
