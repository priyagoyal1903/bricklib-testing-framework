#include <omp.h>
#include "vecscatter.h"
#include "brick.h"
#include "../../../gen/consts.h"
#include "./poisson.h"
#include <brick-hip.h>

__global__ void poisson_naive2(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0]) {
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;
    out[k][j][i] = 2.666 * in[k][j][i] - 
        (0.166 * (in[k - 1][j][i] + in[k + 1][j][i] +
                in[k][j - 1][i] + in[k][j + 1][i] +
                in[k][j][i - 1] + in[k][j][i - 1])) -
        (0.0833 * (in[k - 1][j - 1][i] + in[k + 1][j - 1][i] +
                in[k - 1][j + 1][i] + in[k + 1][j + 1][i] +
                in[k - 1][j][i - 1] + in[k + 1][j][i - 1] +
                in[k][j - 1][i - 1] + in[k][j + 1][i - 1] +
                in[k - 1][j][i + 1] + in[k + 1][j][i + 1] +
                in[k][j - 1][i + 1] + in[k][j + 1][i + 1]));
}
#define in(a, b, c) in_arr[c][b][a]
#define out(a, b, c) out_arr[c][b][a]
__global__ void poisson_codegen2(bElem (*in_arr)[STRIDE1][STRIDE0], bElem (*out_arr)[STRIDE1][STRIDE0]) {
    long k = OFF2 + (blockIdx.z * TILE2);
    long j = OFF1 + (blockIdx.y * TILE1);
    long i = OFF0 + (blockIdx.x * TILE0);
    tile("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/poisson/intermediate_gen/poisson2.py", VSVEC, (TILE2, TILE1, VECSIZE), ("k", "j", "i"), (1, 1, VECSIZE));
}
#undef in
#undef out
__global__ void poisson_naive_bricks2(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType in, BType out) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;
    out[b][k][j][i] = 2.666 * in[b][k][j][i] - 
        (0.166 * (in[b][k - 1][j][i] + in[b][k + 1][j][i] +
                in[b][k][j - 1][i] + in[b][k][j + 1][i] +
                in[b][k][j][i - 1] + in[b][k][j][i - 1])) -
        (0.0833 * (in[b][k - 1][j - 1][i] + in[b][k + 1][j - 1][i] +
                in[b][k - 1][j + 1][i] + in[b][k + 1][j + 1][i] +
                in[b][k - 1][j][i - 1] + in[b][k + 1][j][i - 1] +
                in[b][k][j - 1][i - 1] + in[b][k][j + 1][i - 1] +
                in[b][k - 1][j][i + 1] + in[b][k + 1][j][i + 1] +
                in[b][k][j - 1][i + 1] + in[b][k][j + 1][i + 1]));
}
__global__ void poisson_codegen_bricks2(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType in, BType out) {
  unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
  brick("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/poisson/intermediate_gen/poisson2.py", VSVEC, (TILE2, TILE1, TILE0), (FOLD), b);
}
__global__ void poisson_naive3(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0]) {
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;
    out[k][j][i] = 2.666 * in[k][j][i] - 
        (0.166 * (in[k - 1][j][i] + in[k + 1][j][i] +
                in[k][j - 1][i] + in[k][j + 1][i] +
                in[k][j][i - 1] + in[k][j][i - 1])) -
        (0.0833 * (in[k - 1][j - 1][i] + in[k + 1][j - 1][i] +
                in[k - 1][j + 1][i] + in[k + 1][j + 1][i] +
                in[k - 1][j][i - 1] + in[k + 1][j][i - 1] +
                in[k][j - 1][i - 1] + in[k][j + 1][i - 1] +
                in[k - 1][j][i + 1] + in[k + 1][j][i + 1] +
                in[k][j - 1][i + 1] + in[k][j + 1][i + 1]));
}
#define in(a, b, c) in_arr[c][b][a]
#define out(a, b, c) out_arr[c][b][a]
__global__ void poisson_codegen3(bElem (*in_arr)[STRIDE1][STRIDE0], bElem (*out_arr)[STRIDE1][STRIDE0]) {
    long k = OFF2 + (blockIdx.z * TILE2);
    long j = OFF1 + (blockIdx.y * TILE1);
    long i = OFF0 + (blockIdx.x * TILE0);
    tile("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/poisson/intermediate_gen/poisson3.py", VSVEC, (TILE2, TILE1, VECSIZE), ("k", "j", "i"), (1, 1, VECSIZE));
}
#undef in
#undef out
__global__ void poisson_naive_bricks3(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType in, BType out) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;
    out[b][k][j][i] = 2.666 * in[b][k][j][i] - 
        (0.166 * (in[b][k - 1][j][i] + in[b][k + 1][j][i] +
                in[b][k][j - 1][i] + in[b][k][j + 1][i] +
                in[b][k][j][i - 1] + in[b][k][j][i - 1])) -
        (0.0833 * (in[b][k - 1][j - 1][i] + in[b][k + 1][j - 1][i] +
                in[b][k - 1][j + 1][i] + in[b][k + 1][j + 1][i] +
                in[b][k - 1][j][i - 1] + in[b][k + 1][j][i - 1] +
                in[b][k][j - 1][i - 1] + in[b][k][j + 1][i - 1] +
                in[b][k - 1][j][i + 1] + in[b][k + 1][j][i + 1] +
                in[b][k][j - 1][i + 1] + in[b][k][j + 1][i + 1]));
}
__global__ void poisson_codegen_bricks3(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType in, BType out) {
  unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
  brick("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/poisson/intermediate_gen/poisson3.py", VSVEC, (TILE2, TILE1, TILE0), (FOLD), b);
}
__global__ void poisson_naive5(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0]) {
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;
    out[k][j][i] = 2.666 * in[k][j][i] - 
        (0.166 * (in[k - 1][j][i] + in[k + 1][j][i] +
                in[k][j - 1][i] + in[k][j + 1][i] +
                in[k][j][i - 1] + in[k][j][i - 1])) -
        (0.0833 * (in[k - 1][j - 1][i] + in[k + 1][j - 1][i] +
                in[k - 1][j + 1][i] + in[k + 1][j + 1][i] +
                in[k - 1][j][i - 1] + in[k + 1][j][i - 1] +
                in[k][j - 1][i - 1] + in[k][j + 1][i - 1] +
                in[k - 1][j][i + 1] + in[k + 1][j][i + 1] +
                in[k][j - 1][i + 1] + in[k][j + 1][i + 1]));
}
#define in(a, b, c) in_arr[c][b][a]
#define out(a, b, c) out_arr[c][b][a]
__global__ void poisson_codegen5(bElem (*in_arr)[STRIDE1][STRIDE0], bElem (*out_arr)[STRIDE1][STRIDE0]) {
    long k = OFF2 + (blockIdx.z * TILE2);
    long j = OFF1 + (blockIdx.y * TILE1);
    long i = OFF0 + (blockIdx.x * TILE0);
    tile("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/poisson/intermediate_gen/poisson5.py", VSVEC, (TILE2, TILE1, VECSIZE), ("k", "j", "i"), (1, 1, VECSIZE));
}
#undef in
#undef out
__global__ void poisson_naive_bricks5(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType in, BType out) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;
    out[b][k][j][i] = 2.666 * in[b][k][j][i] - 
        (0.166 * (in[b][k - 1][j][i] + in[b][k + 1][j][i] +
                in[b][k][j - 1][i] + in[b][k][j + 1][i] +
                in[b][k][j][i - 1] + in[b][k][j][i - 1])) -
        (0.0833 * (in[b][k - 1][j - 1][i] + in[b][k + 1][j - 1][i] +
                in[b][k - 1][j + 1][i] + in[b][k + 1][j + 1][i] +
                in[b][k - 1][j][i - 1] + in[b][k + 1][j][i - 1] +
                in[b][k][j - 1][i - 1] + in[b][k][j + 1][i - 1] +
                in[b][k - 1][j][i + 1] + in[b][k + 1][j][i + 1] +
                in[b][k][j - 1][i + 1] + in[b][k][j + 1][i + 1]));
}
__global__ void poisson_codegen_bricks5(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType in, BType out) {
  unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
  brick("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/poisson/intermediate_gen/poisson5.py", VSVEC, (TILE2, TILE1, TILE0), (FOLD), b);
}
