#ifndef NDL_METAL_COMMON_H
#define NDL_METAL_COMMON_H
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <cstdint>
#endif

// from ndarray_backend_cuda.cu
typedef float scalar_t;

#define TILE 4
#define MAX_VEC_SIZE 8

struct MetalVec { uint32_t size; int32_t data[MAX_VEC_SIZE]; };


#endif