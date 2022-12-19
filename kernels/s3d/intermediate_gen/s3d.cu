#include <omp.h>
#include "vecscatter.h"
#include "brick.h"
// #include "../out/laplacian-stencils.h"
#include "../../../gen/consts.h"
#include "./s3d.h"
#include <brick-hip.h>

__global__ void s3d_naive2(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0], bElem* dev_coeff) {
    const size_t radius = 2;
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;
    bElem temp = dev_coeff[0] * in[k][j][i];
    #pragma unroll
    for (int a = 1; a <= radius; a++) {
        temp += dev_coeff[a] * (
            in[k][j][i + a] + in[k][j + a][i] + in[k + a][j][i] +
            in[k][j][i - a] + in[k][j - a][i] + in[k - a][j][i]);
    }
    out[k][j][i] = temp;
}
__global__ void s3d_naive_bricks2(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem *dev_coeff) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;
    bOut[b][k][j][i] = dev_coeff[0] * bIn[b][k][j][i];
    const size_t radius = 2;
    #pragma unroll
    for (int a = 1; a <= radius; a++) {
        bOut[b][k][j][i] += dev_coeff[a] * (
            bIn[b][k][j][i + a] + bIn[b][k][j + a][i] + bIn[b][k + a][j][i] +
            bIn[b][k][j][i - a] + bIn[b][k][j - a][i] + bIn[b][k - a][j][i]
        );
    }
}
__global__ void s3d_codegen_bricks2(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem *dev_coeff) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    brick("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/s3d/intermediate_gen/laplacian2.py", VSVEC, (TILE2, TILE1, TILE0), (FOLD), b);
}
#define bIn(a, b, c) arr_in[c][b][a]
#define bOut(a, b, c) arr_out[c][b][a]
__global__ void s3d_codegen2(bElem (*arr_in)[STRIDE1][STRIDE0], bElem (*arr_out)[STRIDE1][STRIDE0], bElem *dev_coeff) {
    long k = OFF2 + (blockIdx.z * TILE2);
    long j = OFF1 + (blockIdx.y * TILE1);
    long i = OFF0 + (blockIdx.x * VECSIZE);
    tile("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/s3d/intermediate_gen/laplacian2.py", VSVEC, (TILE2, TILE1, VECSIZE), ("k", "j", "i"), (1, 1, VECSIZE));
}
#undef bIn
#undef bOut
__global__ void s3d_naive3(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0], bElem* dev_coeff) {
    const size_t radius = 3;
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;
    bElem temp = dev_coeff[0] * in[k][j][i];
    #pragma unroll
    for (int a = 1; a <= radius; a++) {
        temp += dev_coeff[a] * (
            in[k][j][i + a] + in[k][j + a][i] + in[k + a][j][i] +
            in[k][j][i - a] + in[k][j - a][i] + in[k - a][j][i]);
    }
    out[k][j][i] = temp;
}
__global__ void s3d_naive_bricks3(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem *dev_coeff) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;
    bOut[b][k][j][i] = dev_coeff[0] * bIn[b][k][j][i];
    const size_t radius = 3;
    #pragma unroll
    for (int a = 1; a <= radius; a++) {
        bOut[b][k][j][i] += dev_coeff[a] * (
            bIn[b][k][j][i + a] + bIn[b][k][j + a][i] + bIn[b][k + a][j][i] +
            bIn[b][k][j][i - a] + bIn[b][k][j - a][i] + bIn[b][k - a][j][i]
        );
    }
}
__global__ void s3d_codegen_bricks3(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem *dev_coeff) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    brick("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/s3d/intermediate_gen/laplacian3.py", VSVEC, (TILE2, TILE1, TILE0), (FOLD), b);
}
#define bIn(a, b, c) arr_in[c][b][a]
#define bOut(a, b, c) arr_out[c][b][a]
__global__ void s3d_codegen3(bElem (*arr_in)[STRIDE1][STRIDE0], bElem (*arr_out)[STRIDE1][STRIDE0], bElem *dev_coeff) {
    long k = OFF2 + (blockIdx.z * TILE2);
    long j = OFF1 + (blockIdx.y * TILE1);
    long i = OFF0 + (blockIdx.x * VECSIZE);
    tile("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/s3d/intermediate_gen/laplacian3.py", VSVEC, (TILE2, TILE1, VECSIZE), ("k", "j", "i"), (1, 1, VECSIZE));
}
#undef bIn
#undef bOut
__global__ void s3d_naive5(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0], bElem* dev_coeff) {
    const size_t radius = 5;
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;
    bElem temp = dev_coeff[0] * in[k][j][i];
    #pragma unroll
    for (int a = 1; a <= radius; a++) {
        temp += dev_coeff[a] * (
            in[k][j][i + a] + in[k][j + a][i] + in[k + a][j][i] +
            in[k][j][i - a] + in[k][j - a][i] + in[k - a][j][i]);
    }
    out[k][j][i] = temp;
}
__global__ void s3d_naive_bricks5(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem *dev_coeff) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;
    bOut[b][k][j][i] = dev_coeff[0] * bIn[b][k][j][i];
    const size_t radius = 5;
    #pragma unroll
    for (int a = 1; a <= radius; a++) {
        bOut[b][k][j][i] += dev_coeff[a] * (
            bIn[b][k][j][i + a] + bIn[b][k][j + a][i] + bIn[b][k + a][j][i] +
            bIn[b][k][j][i - a] + bIn[b][k][j - a][i] + bIn[b][k - a][j][i]
        );
    }
}
__global__ void s3d_codegen_bricks5(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem *dev_coeff) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    brick("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/s3d/intermediate_gen/laplacian5.py", VSVEC, (TILE2, TILE1, TILE0), (FOLD), b);
}
#define bIn(a, b, c) arr_in[c][b][a]
#define bOut(a, b, c) arr_out[c][b][a]
__global__ void s3d_codegen5(bElem (*arr_in)[STRIDE1][STRIDE0], bElem (*arr_out)[STRIDE1][STRIDE0], bElem *dev_coeff) {
    long k = OFF2 + (blockIdx.z * TILE2);
    long j = OFF1 + (blockIdx.y * TILE1);
    long i = OFF0 + (blockIdx.x * VECSIZE);
    tile("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/s3d/intermediate_gen/laplacian5.py", VSVEC, (TILE2, TILE1, VECSIZE), ("k", "j", "i"), (1, 1, VECSIZE));
}
#undef bIn
#undef bOut
