#include <omp.h>
#include "vecscatter.h"
#include "brick.h"
#include "../../../gen/consts.h"
#include "./f3d.h"
#include <brick-hip.h>

__global__ void f3d_naive1(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0], bElem (*c)[8][8]) {
    const int radius = 1;
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;
    bElem base = in[i][j][k] * c[i][j][k];
    #pragma unroll
    for (int i_diff = -radius; i_diff <= radius; i_diff++) {
        #pragma unroll
        for (int j_diff = -radius; j_diff <= radius; j_diff++) {
            #pragma unroll
            for (int k_diff = -radius; k_diff <= radius; k_diff++) {		
		base += (in[i + i_diff][j + j_diff][k + k_diff] * c[i_diff + radius][j_diff + radius][k_diff + radius]);
            }
        }
    }
    out[i][j][k] = base;
}
__global__ void f3d_naive_bricks1(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem (*c)[8][8]) {
    const int radius = 1;
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;
    bElem base = bIn[b][i][j][k] * c[i][j][k];
    #pragma unroll
    for (int i_diff = -radius; i_diff <= radius; i_diff++) {
        #pragma unroll
        for (int j_diff = -radius; j_diff <= radius; j_diff++) {
            #pragma unroll
            for (int k_diff = -radius; k_diff <= radius; k_diff++) {
                base += (bIn[b][i + i_diff][j + j_diff][k + k_diff] * c[i_diff + radius][j_diff + radius][k_diff + radius]);
	    }
        }
    }
    bOut[b][i][j][k] = base;
}
#define bIn(a, b, c) arr_in[c][b][a]
#define bOut(a, b, c) arr_out[c][b][a]
__global__ void f3d_codegen1(bElem (*arr_in)[STRIDE1][STRIDE0], bElem (*arr_out)[STRIDE1][STRIDE0], bElem (*c)[8][8]) {
    long k = OFF2 + (blockIdx.z * TILE2);
    long j = OFF1 + (blockIdx.y * TILE1);
    long i = OFF0 + (blockIdx.x * VECSIZE);
    tile("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/f3d/intermediate_gen/f3d1.py", VSVEC, (TILE2, TILE1, VECSIZE), ("k", "j", "i"), (1, 1, VECSIZE));
}
#undef bIn
#undef bOut
__global__ void f3d_codegen_bricks1(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem (*c)[8][8]) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    brick("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/f3d/intermediate_gen/f3d1.py", VSVEC, (TILE2, TILE1, TILE0), (FOLD), b);
}
__global__ void f3d_naive2(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0], bElem (*c)[8][8]) {
    const int radius = 2;
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;
    bElem base = in[i][j][k] * c[i][j][k];
    #pragma unroll
    for (int i_diff = -radius; i_diff <= radius; i_diff++) {
        #pragma unroll
        for (int j_diff = -radius; j_diff <= radius; j_diff++) {
            #pragma unroll
            for (int k_diff = -radius; k_diff <= radius; k_diff++) {		
		base += (in[i + i_diff][j + j_diff][k + k_diff] * c[i_diff + radius][j_diff + radius][k_diff + radius]);
            }
        }
    }
    out[i][j][k] = base;
}
__global__ void f3d_naive_bricks2(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem (*c)[8][8]) {
    const int radius = 2;
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;
    bElem base = bIn[b][i][j][k] * c[i][j][k];
    #pragma unroll
    for (int i_diff = -radius; i_diff <= radius; i_diff++) {
        #pragma unroll
        for (int j_diff = -radius; j_diff <= radius; j_diff++) {
            #pragma unroll
            for (int k_diff = -radius; k_diff <= radius; k_diff++) {
                base += (bIn[b][i + i_diff][j + j_diff][k + k_diff] * c[i_diff + radius][j_diff + radius][k_diff + radius]);
	    }
        }
    }
    bOut[b][i][j][k] = base;
}
#define bIn(a, b, c) arr_in[c][b][a]
#define bOut(a, b, c) arr_out[c][b][a]
__global__ void f3d_codegen2(bElem (*arr_in)[STRIDE1][STRIDE0], bElem (*arr_out)[STRIDE1][STRIDE0], bElem (*c)[8][8]) {
    long k = OFF2 + (blockIdx.z * TILE2);
    long j = OFF1 + (blockIdx.y * TILE1);
    long i = OFF0 + (blockIdx.x * VECSIZE);
    tile("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/f3d/intermediate_gen/f3d2.py", VSVEC, (TILE2, TILE1, VECSIZE), ("k", "j", "i"), (1, 1, VECSIZE));
}
#undef bIn
#undef bOut
__global__ void f3d_codegen_bricks2(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem (*c)[8][8]) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    brick("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/f3d/intermediate_gen/f3d2.py", VSVEC, (TILE2, TILE1, TILE0), (FOLD), b);
}
