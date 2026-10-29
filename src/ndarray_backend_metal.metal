/*
File: ndarray_backend_metal.metal
Authors: Elias Zheng,

Metal kernels and operators for ndarray backend.

NOTE: Host code and launchers in ndarary_backend_metal.mm
*/

#include <metal_stdlib>
#include "ndarray_backend_metal.h"

using namespace metal;

////////////////////////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////////////////////////

// Convert an index into the compact array to the matching memory index in the strided array
uint CompactToStride(uint compact_idx, constant MetalVec& shape, constant MetalVec& strides, uint offset){
    uint stride_idx = offset;

    for (int d = int(shape.size) - 1; d >= 0; d--){
        uint coord_idx = compact_idx % uint(shape.data[d]);
        stride_idx += coord_idx * uint(strides.data[d]);
        compact_idx /= uint(shape.data[d]);
    }

    return stride_idx;
}

////////////////////////////////////////////////////////////////////////////////
// Operators
////////////////////////////////////////////////////////////////////////////////
struct IdentityOp{
    scalar_t operator()(scalar_t x) const {return x;};
};

struct AddOp{
    scalar_t operator()(scalar_t x, scalar_t y) const {return x + y;};
};

struct SubOp{
    scalar_t operator()(scalar_t x, scalar_t y) const {return x - y;};
};

struct MulOp{
    scalar_t operator()(scalar_t x, scalar_t y) const {return x * y;}
};

struct DivOp {
    scalar_t operator()(scalar_t x, scalar_t y) const {return x / y;}
};

////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////

// Unary operation kernel
template <typename Op>
kernel void UnaryKernel(
    device const scalar_t *a    [[buffer(0)]],
    device scalar_t       *out  [[buffer(1)]],
    constant uint&         size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]){
    
    if (gid < size){
        out[gid] = Op()(a[gid]);
    }
}

// Binary operation kernel
template <typename Op>
kernel void BinaryKernel(
    device const scalar_t *a    [[buffer(0)]],
    device const scalar_t *b    [[buffer(1)]],
    device scalar_t       *out  [[buffer(2)]],
    constant uint&         size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]){
    
    if (gid < size){
        out[gid] = Op()(a[gid], b[gid]);
    }
}

// Scalar operation kernel
template <typename Op>
kernel void ScalarKernel(
    device const scalar_t *a    [[buffer(0)]],
    constant scalar_t&     val  [[buffer(1)]],
    device scalar_t       *out  [[buffer(2)]],
    constant uint&         size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]){
    
    if (gid < size){
        out[gid] = Op()(a[gid], val);
    }
}

// TODO: compact kernel (reuse CompactToStride, out[gid] = a[strided index])

// setitem kernels aren't templates, so the function name is the name the host looks up

// Ewise setitem kernel: write compact array a into the strided view of out
kernel void ewise_setitem(
    device const scalar_t *a       [[buffer(0)]],
    device scalar_t       *out     [[buffer(1)]],
    constant uint&         size    [[buffer(2)]],
    constant MetalVec&     shape   [[buffer(3)]],
    constant MetalVec&     strides [[buffer(4)]],
    constant uint&         offset  [[buffer(5)]],
    uint gid [[thread_position_in_grid]]){

    if (gid < size){
        out[CompactToStride(gid, shape, strides, offset)] = a[gid];
    }
}

// Scalar setitem kernel: write val into every element of the strided view of out
kernel void scalar_setitem(
    constant scalar_t&     val     [[buffer(0)]],
    device scalar_t       *out     [[buffer(1)]],
    constant uint&         size    [[buffer(2)]],
    constant MetalVec&     shape   [[buffer(3)]],
    constant MetalVec&     strides [[buffer(4)]],
    constant uint&         offset  [[buffer(5)]],
    uint gid [[thread_position_in_grid]]){

    if (gid < size){
        out[CompactToStride(gid, shape, strides, offset)] = val;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Macros
////////////////////////////////////////////////////////////////////////////////

# define INIT_BINARY(name, Op) template [[host_name(name)]] kernel void BinaryKernel<Op>( \
    device const scalar_t *, device const scalar_t*, device scalar_t*, constant uint&, uint \
);

# define INIT_SCALAR(name, Op) template [[host_name(name)]] kernel void ScalarKernel<Op>( \
    device const scalar_t *, constant scalar_t&, device scalar_t *, constant uint&, uint \
);

# define INIT_UNARY(name, Op) template [[host_name(name)]] kernel void UnaryKernel<Op>(\
    device const scalar_t *, device scalar_t *, constant uint&, uint \
);

////////////////////////////////////////////////////////////////////////////////
// Operation Kernels
////////////////////////////////////////////////////////////////////////////////

// define an operation kernel with INIT_[type](name, [operator struct])
// match kernel names with binding names please!!!

INIT_UNARY("identity", IdentityOp);

INIT_BINARY("ewise_add", AddOp);
INIT_SCALAR("scalar_add", AddOp);

// TODO: add sub to backends
INIT_BINARY("ewise_sub", SubOp);
INIT_SCALAR("scalar_sub", SubOp);

INIT_BINARY("ewise_mul", MulOp);
INIT_SCALAR("scalar_mul", MulOp);