#include <omp.h>
#include "vecscatter.h"
#include "brick.h"
#include "../../../gen/consts.h"
#include "./helmholtz4.h"
#include <brick-hip.h>

__global__ void helmholtz4_naive2(bElem (*x)[STRIDE1][STRIDE0], bElem (*alpha)[STRIDE1][STRIDE0], 
  bElem (*beta_i)[STRIDE1][STRIDE0], bElem (*beta_j)[STRIDE1][STRIDE0], bElem (*beta_k)[STRIDE1][STRIDE0], 
  bElem (*out)[STRIDE1][STRIDE0], bElem *c) {
    bElem c1 = c[0];
    bElem c2 = c[1];
    bElem h2inv = c[2];
    out[k][j][i] = c1 * alpha[k][j][i] * x[k][j][i]] -
        c2 * h2inv *
        (0.0833 * 
            (beta_i[k][j][i] * (15.0 * (x[k][j][i - 1] - x[k][j][i]) - 
                (x[k][j][i - 1] - x[k][j][i + 1])) + 
            beta_i[k][j][i + 1] * (15.0 * (x[k][j][i + 1] - x[k][j][i]) - 
                (x[k][j][i + 2] - x[k][j][i - 1])) +
            beta_j[k][j][i] * (15.0 * (x[k][j - 1][i] - x[k][j][i]) - 
                (x[k][j - 1][i] - x[k][j + 1][i])) +
            beta_j[k][j + 1][i] * (15.0 * (x[k][j + 1][i] - x[k][j][i]) -
                (x[k][j + 2][i] - x[k][j - 1][i])) +
            beta_k[k][j][i] * (15.0 * (x[k - 1][j][i] - x[k][j][i]) -
                (x[k - 2][j][i] - x[k + 1][j][i])) +
            beta_k[k + 1][j][i] * (15.0 * (x[k + 1][j][i] - x[k][j][i]) -
                (x[k + 2][j][i] - x[k - 1][j][i]))) +
        0.25 * 0.0833 * 
            ((beta_i[k][j + 1][i] - beta_i[k][j - 1][i]) *
                (x[k][j + 1][i - 1] - x[k][j + 1][i] -
                 x[k][j - 1][i - 1] + x[k][j - 1][i]) +
            (beta_i[k + 1][j][i] - beta_i[k - 1][j][i]) * 
                (x[k + 1][j][i - 1] - x[k + 1][j][i] -
                 x[k - 1][j][i - 1] + x[k - 1][j][i]) +
            (beta_j[k][j][i + 1] - beta_j[k][j][i - 1]) *
                (x[k][j - 1][i + 1] - x[k][j][i + 1] -
                 x[k][j - 1][i - 1] + x[k][j][i - 1]) +
            (beta_j[k + 1][j][i] - beta_j[k - 1][j][i]) *
                (x[k + 1][j - 1][i] - x[k + 1][j][i] -
                 x[k - 1][j - 1][i] + x[k - 1][j][i]) +
            (beta_k[k][j][i + 1] - beta_k[k][j][i - 1]) *
                (x[k - 1][j][i + 1] - x[k][j][i + 1] -
                 x[k - 1][j][i - 1] + x[k][j][i - 1]) +
            (beta_k[k][j + 1][i] - beta_k[k][j - 1][i]) *
                (x[k - 1][j + 1][i] - x[k][j + 1][i] -
                 x[k - 1][j - 1][i] + x[k][j - 1][i]) +
            (beta_i[k][j + 1][i + 1] - beta_i[k][j - 1][i + 1]) *
                (x[k][j + 1][i + 1] - x[k][j + 1][i] -
                 x[k][j - 1][i + 1] + x[k][j - 1][i]) + 
            (beta_i[k + 1][j][i + 1] - beta_i[k - 1][j][i + 1]) *
                (x[k + 1][j][i + 1] - x[k + 1][j][i] - 
                 x[k - 1][j][i + 1] + x[k - 1][j][i]) +
            (beta_j[k][j + 1][i + 1] - beta_j[k][j + 1][i - 1]) *
                (x[k][j + 1][i + 1] - x[k][j][i + 1] -
                 x[k][j + 1][i - 1] + x[k][j][i - 1]) +
            (beta_j[k + 1][j + 1][i] - beta_j[k - 1][j + 1][i]) *
                (x[k + 1][j + 1][i] - x[k + 1][j][i] -
                 x[k - 1][j + 1][i] + x[k - 1][j][i]) +
            (beta_k[k + 1][j][i + 1] - beta_k[k + 1][j][i - 1]) *
                (x[k + 1][j][i + 1] - x[k][j][i + 1] -
                 x[k + 1][j][i - 1] + x[k][j][i - 1]) +
            (beta_k[k + 1][j + 1][i] - beta_k[k + 1][j - 1][i]) *
                (x[k + 1][j + 1][i] - x[k][j + 1][i] -
                 x[k + 1][j - 1][i] + x[k][j - 1][i])
        ));
}
#define x(a, b, c) x_arr[c][b][a]
#define alpha(a, b, c) alpha_arr[c][b][a]
#define beta_i(a, b, c) beta_i_arr[c][b][a]
#define beta_j(a, b, c) beta_j_arr[c][b][a]
#define beta_k(a, b, c) beta_k_arr[c][b][a] 
#define out(a, b, c) out_arr[c][b][a]
__global__ void helmholtz4_codegen2(bElem (*x_arr)[STRIDE1][STRIDE0], bElem (*alpha_arr)[STRIDE1][STRIDE0], 
  bElem (*beta_i_arr)[STRIDE1][STRIDE0], bElem (*beta_j_arr)[STRIDE1][STRIDE0], bElem (*beta_k_arr)[STRIDE1][STRIDE0], 
  bElem (*out_arr)[STRIDE1][STRIDE0], bElem *c) {
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;
    tile("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/helmholtz4/intermediate_gen/helmholtz42.py", VSVEC, (TILE2, TILE1, VECSIZE), ("k", "j", "i"), (1, 1, VECSIZE));
}
#undef x
#undef alpha
#undef beta_i
#undef beta_j
#undef beta_k
#undef out
__global__ void helmholtz4_naive_bricks2(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType x, BType alpha,
  BType beta_i, BType beta_j, BType beta_k,
  BType out, bElem *c) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;
    bElem c1 = c[0];
    bElem c2 = c[1];
    bElem h2inv = c[2];
    out[b][k][j][i] = c1 * alpha[b][k][j][i] * x[b][k][j][i]] -
        c2 * h2inv *
        (0.0833 * 
            (beta_i[b][k][j][i] * (15.0 * (x[b][k][j][i - 1] - x[b][k][j][i]) - 
                (x[b][k][j][i - 1] - x[b][k][j][i + 1])) + 
            beta_i[b][k][j][i + 1] * (15.0 * (x[b][k][j][i + 1] - x[b][k][j][i]) - 
                (x[b][k][j][i + 2] - x[b][k][j][i - 1])) +
            beta_j[b][k][j][i] * (15.0 * (x[b][k][j - 1][i] - x[b][k][j][i]) - 
                (x[b][k][j - 1][i] - x[b][k][j + 1][i])) +
            beta_j[b][k][j + 1][i] * (15.0 * (x[b][k][j + 1][i] - x[b][k][j][i]) -
                (x[b][k][j + 2][i] - x[b][k][j - 1][i])) +
            beta_k[b][k][j][i] * (15.0 * (x[b][k - 1][j][i] - x[b][k][j][i]) -
                (x[b][k - 2][j][i] - x[b][k + 1][j][i])) +
            beta_k[b][k + 1][j][i] * (15.0 * (x[b][k + 1][j][i] - x[b][k][j][i]) -
                (x[b][k + 2][j][i] - x[b][k - 1][j][i]))) +
        0.25 * 0.0833 * 
            ((beta_i[b][k][j + 1][i] - beta_i[b][k][j - 1][i]) *
                (x[b][k][j + 1][i - 1] - x[b][k][j + 1][i] -
                 x[b][k][j - 1][i - 1] + x[b][k][j - 1][i]) +
            (beta_i[b][k + 1][j][i] - beta_i[b][k - 1][j][i]) * 
                (x[b][k + 1][j][i - 1] - x[b][k + 1][j][i] -
                 x[b][k - 1][j][i - 1] + x[b][k - 1][j][i]) +
            (beta_j[b][k][j][i + 1] - beta_j[b][k][j][i - 1]) *
                (x[b][k][j - 1][i + 1] - x[b][k][j][i + 1] -
                 x[b][k][j - 1][i - 1] + x[b][k][j][i - 1]) +
            (beta_j[b][k + 1][j][i] - beta_j[b][k - 1][j][i]) *
                (x[b][k + 1][j - 1][i] - x[b][k + 1][j][i] -
                 x[b][k - 1][j - 1][i] + x[b][k - 1][j][i]) +
            (beta_k[b][k][j][i + 1] - beta_k[b][k][j][i - 1]) *
                (x[b][k - 1][j][i + 1] - x[b][k][j][i + 1] -
                 x[b][k - 1][j][i - 1] + x[b][k][j][i - 1]) +
            (beta_k[b][k][j + 1][i] - beta_k[b][k][j - 1][i]) *
                (x[b][k - 1][j + 1][i] - x[b][k][j + 1][i] -
                 x[b][k - 1][j - 1][i] + x[b][k][j - 1][i]) +
            (beta_i[b][k][j + 1][i + 1] - beta_i[b][k][j - 1][i + 1]) *
                (x[b][k][j + 1][i + 1] - x[b][k][j + 1][i] -
                 x[b][k][j - 1][i + 1] + x[b][k][j - 1][i]) + 
            (beta_i[b][k + 1][j][i + 1] - beta_i[b][k - 1][j][i + 1]) *
                (x[b][k + 1][j][i + 1] - x[b][k + 1][j][i] - 
                 x[b][k - 1][j][i + 1] + x[b][k - 1][j][i]) +
            (beta_j[b][k][j + 1][i + 1] - beta_j[b][k][j + 1][i - 1]) *
                (x[b][k][j + 1][i + 1] - x[b][k][j][i + 1] -
                 x[b][k][j + 1][i - 1] + x[b][k][j][i - 1]) +
            (beta_j[b][k + 1][j + 1][i] - beta_j[b][k - 1][j + 1][i]) *
                (x[b][k + 1][j + 1][i] - x[b][k + 1][j][i] -
                 x[b][k - 1][j + 1][i] + x[b][k - 1][j][i]) +
            (beta_k[b][k + 1][j][i + 1] - beta_k[b][k + 1][j][i - 1]) *
                (x[b][k + 1][j][i + 1] - x[b][k][j][i + 1] -
                 x[b][k + 1][j][i - 1] + x[b][k][j][i - 1]) +
            (beta_k[b][k + 1][j + 1][i] - beta_k[b][k + 1][j - 1][i]) *
                (x[b][k + 1][j + 1][i] - x[b][k][j + 1][i] -
                 x[b][k + 1][j - 1][i] + x[b][k][j - 1][i])
        ));
} 
__global__ void helmholtz4_codegen_bricks2(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType x, BType alpha,
  BType beta_i, BType beta_j, BType beta_k,
  BType out, bElem *c) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    brick("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/helmholtz4/intermediate_gen/helmholtz42.py", VSVEC, (TILE2, TILE1, TILE0), (FOLD), b);
}
__global__ void helmholtz4_naive3(bElem (*x)[STRIDE1][STRIDE0], bElem (*alpha)[STRIDE1][STRIDE0], 
  bElem (*beta_i)[STRIDE1][STRIDE0], bElem (*beta_j)[STRIDE1][STRIDE0], bElem (*beta_k)[STRIDE1][STRIDE0], 
  bElem (*out)[STRIDE1][STRIDE0], bElem *c) {
    bElem c1 = c[0];
    bElem c2 = c[1];
    bElem h2inv = c[2];
    out[k][j][i] = c1 * alpha[k][j][i] * x[k][j][i]] -
        c2 * h2inv *
        (0.0833 * 
            (beta_i[k][j][i] * (15.0 * (x[k][j][i - 1] - x[k][j][i]) - 
                (x[k][j][i - 1] - x[k][j][i + 1])) + 
            beta_i[k][j][i + 1] * (15.0 * (x[k][j][i + 1] - x[k][j][i]) - 
                (x[k][j][i + 2] - x[k][j][i - 1])) +
            beta_j[k][j][i] * (15.0 * (x[k][j - 1][i] - x[k][j][i]) - 
                (x[k][j - 1][i] - x[k][j + 1][i])) +
            beta_j[k][j + 1][i] * (15.0 * (x[k][j + 1][i] - x[k][j][i]) -
                (x[k][j + 2][i] - x[k][j - 1][i])) +
            beta_k[k][j][i] * (15.0 * (x[k - 1][j][i] - x[k][j][i]) -
                (x[k - 2][j][i] - x[k + 1][j][i])) +
            beta_k[k + 1][j][i] * (15.0 * (x[k + 1][j][i] - x[k][j][i]) -
                (x[k + 2][j][i] - x[k - 1][j][i]))) +
        0.25 * 0.0833 * 
            ((beta_i[k][j + 1][i] - beta_i[k][j - 1][i]) *
                (x[k][j + 1][i - 1] - x[k][j + 1][i] -
                 x[k][j - 1][i - 1] + x[k][j - 1][i]) +
            (beta_i[k + 1][j][i] - beta_i[k - 1][j][i]) * 
                (x[k + 1][j][i - 1] - x[k + 1][j][i] -
                 x[k - 1][j][i - 1] + x[k - 1][j][i]) +
            (beta_j[k][j][i + 1] - beta_j[k][j][i - 1]) *
                (x[k][j - 1][i + 1] - x[k][j][i + 1] -
                 x[k][j - 1][i - 1] + x[k][j][i - 1]) +
            (beta_j[k + 1][j][i] - beta_j[k - 1][j][i]) *
                (x[k + 1][j - 1][i] - x[k + 1][j][i] -
                 x[k - 1][j - 1][i] + x[k - 1][j][i]) +
            (beta_k[k][j][i + 1] - beta_k[k][j][i - 1]) *
                (x[k - 1][j][i + 1] - x[k][j][i + 1] -
                 x[k - 1][j][i - 1] + x[k][j][i - 1]) +
            (beta_k[k][j + 1][i] - beta_k[k][j - 1][i]) *
                (x[k - 1][j + 1][i] - x[k][j + 1][i] -
                 x[k - 1][j - 1][i] + x[k][j - 1][i]) +
            (beta_i[k][j + 1][i + 1] - beta_i[k][j - 1][i + 1]) *
                (x[k][j + 1][i + 1] - x[k][j + 1][i] -
                 x[k][j - 1][i + 1] + x[k][j - 1][i]) + 
            (beta_i[k + 1][j][i + 1] - beta_i[k - 1][j][i + 1]) *
                (x[k + 1][j][i + 1] - x[k + 1][j][i] - 
                 x[k - 1][j][i + 1] + x[k - 1][j][i]) +
            (beta_j[k][j + 1][i + 1] - beta_j[k][j + 1][i - 1]) *
                (x[k][j + 1][i + 1] - x[k][j][i + 1] -
                 x[k][j + 1][i - 1] + x[k][j][i - 1]) +
            (beta_j[k + 1][j + 1][i] - beta_j[k - 1][j + 1][i]) *
                (x[k + 1][j + 1][i] - x[k + 1][j][i] -
                 x[k - 1][j + 1][i] + x[k - 1][j][i]) +
            (beta_k[k + 1][j][i + 1] - beta_k[k + 1][j][i - 1]) *
                (x[k + 1][j][i + 1] - x[k][j][i + 1] -
                 x[k + 1][j][i - 1] + x[k][j][i - 1]) +
            (beta_k[k + 1][j + 1][i] - beta_k[k + 1][j - 1][i]) *
                (x[k + 1][j + 1][i] - x[k][j + 1][i] -
                 x[k + 1][j - 1][i] + x[k][j - 1][i])
        ));
}
#define x(a, b, c) x_arr[c][b][a]
#define alpha(a, b, c) alpha_arr[c][b][a]
#define beta_i(a, b, c) beta_i_arr[c][b][a]
#define beta_j(a, b, c) beta_j_arr[c][b][a]
#define beta_k(a, b, c) beta_k_arr[c][b][a] 
#define out(a, b, c) out_arr[c][b][a]
__global__ void helmholtz4_codegen3(bElem (*x_arr)[STRIDE1][STRIDE0], bElem (*alpha_arr)[STRIDE1][STRIDE0], 
  bElem (*beta_i_arr)[STRIDE1][STRIDE0], bElem (*beta_j_arr)[STRIDE1][STRIDE0], bElem (*beta_k_arr)[STRIDE1][STRIDE0], 
  bElem (*out_arr)[STRIDE1][STRIDE0], bElem *c) {
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;
    tile("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/helmholtz4/intermediate_gen/helmholtz43.py", VSVEC, (TILE2, TILE1, VECSIZE), ("k", "j", "i"), (1, 1, VECSIZE));
}
#undef x
#undef alpha
#undef beta_i
#undef beta_j
#undef beta_k
#undef out
__global__ void helmholtz4_naive_bricks3(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType x, BType alpha,
  BType beta_i, BType beta_j, BType beta_k,
  BType out, bElem *c) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;
    bElem c1 = c[0];
    bElem c2 = c[1];
    bElem h2inv = c[2];
    out[b][k][j][i] = c1 * alpha[b][k][j][i] * x[b][k][j][i]] -
        c2 * h2inv *
        (0.0833 * 
            (beta_i[b][k][j][i] * (15.0 * (x[b][k][j][i - 1] - x[b][k][j][i]) - 
                (x[b][k][j][i - 1] - x[b][k][j][i + 1])) + 
            beta_i[b][k][j][i + 1] * (15.0 * (x[b][k][j][i + 1] - x[b][k][j][i]) - 
                (x[b][k][j][i + 2] - x[b][k][j][i - 1])) +
            beta_j[b][k][j][i] * (15.0 * (x[b][k][j - 1][i] - x[b][k][j][i]) - 
                (x[b][k][j - 1][i] - x[b][k][j + 1][i])) +
            beta_j[b][k][j + 1][i] * (15.0 * (x[b][k][j + 1][i] - x[b][k][j][i]) -
                (x[b][k][j + 2][i] - x[b][k][j - 1][i])) +
            beta_k[b][k][j][i] * (15.0 * (x[b][k - 1][j][i] - x[b][k][j][i]) -
                (x[b][k - 2][j][i] - x[b][k + 1][j][i])) +
            beta_k[b][k + 1][j][i] * (15.0 * (x[b][k + 1][j][i] - x[b][k][j][i]) -
                (x[b][k + 2][j][i] - x[b][k - 1][j][i]))) +
        0.25 * 0.0833 * 
            ((beta_i[b][k][j + 1][i] - beta_i[b][k][j - 1][i]) *
                (x[b][k][j + 1][i - 1] - x[b][k][j + 1][i] -
                 x[b][k][j - 1][i - 1] + x[b][k][j - 1][i]) +
            (beta_i[b][k + 1][j][i] - beta_i[b][k - 1][j][i]) * 
                (x[b][k + 1][j][i - 1] - x[b][k + 1][j][i] -
                 x[b][k - 1][j][i - 1] + x[b][k - 1][j][i]) +
            (beta_j[b][k][j][i + 1] - beta_j[b][k][j][i - 1]) *
                (x[b][k][j - 1][i + 1] - x[b][k][j][i + 1] -
                 x[b][k][j - 1][i - 1] + x[b][k][j][i - 1]) +
            (beta_j[b][k + 1][j][i] - beta_j[b][k - 1][j][i]) *
                (x[b][k + 1][j - 1][i] - x[b][k + 1][j][i] -
                 x[b][k - 1][j - 1][i] + x[b][k - 1][j][i]) +
            (beta_k[b][k][j][i + 1] - beta_k[b][k][j][i - 1]) *
                (x[b][k - 1][j][i + 1] - x[b][k][j][i + 1] -
                 x[b][k - 1][j][i - 1] + x[b][k][j][i - 1]) +
            (beta_k[b][k][j + 1][i] - beta_k[b][k][j - 1][i]) *
                (x[b][k - 1][j + 1][i] - x[b][k][j + 1][i] -
                 x[b][k - 1][j - 1][i] + x[b][k][j - 1][i]) +
            (beta_i[b][k][j + 1][i + 1] - beta_i[b][k][j - 1][i + 1]) *
                (x[b][k][j + 1][i + 1] - x[b][k][j + 1][i] -
                 x[b][k][j - 1][i + 1] + x[b][k][j - 1][i]) + 
            (beta_i[b][k + 1][j][i + 1] - beta_i[b][k - 1][j][i + 1]) *
                (x[b][k + 1][j][i + 1] - x[b][k + 1][j][i] - 
                 x[b][k - 1][j][i + 1] + x[b][k - 1][j][i]) +
            (beta_j[b][k][j + 1][i + 1] - beta_j[b][k][j + 1][i - 1]) *
                (x[b][k][j + 1][i + 1] - x[b][k][j][i + 1] -
                 x[b][k][j + 1][i - 1] + x[b][k][j][i - 1]) +
            (beta_j[b][k + 1][j + 1][i] - beta_j[b][k - 1][j + 1][i]) *
                (x[b][k + 1][j + 1][i] - x[b][k + 1][j][i] -
                 x[b][k - 1][j + 1][i] + x[b][k - 1][j][i]) +
            (beta_k[b][k + 1][j][i + 1] - beta_k[b][k + 1][j][i - 1]) *
                (x[b][k + 1][j][i + 1] - x[b][k][j][i + 1] -
                 x[b][k + 1][j][i - 1] + x[b][k][j][i - 1]) +
            (beta_k[b][k + 1][j + 1][i] - beta_k[b][k + 1][j - 1][i]) *
                (x[b][k + 1][j + 1][i] - x[b][k][j + 1][i] -
                 x[b][k + 1][j - 1][i] + x[b][k][j - 1][i])
        ));
} 
__global__ void helmholtz4_codegen_bricks3(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType x, BType alpha,
  BType beta_i, BType beta_j, BType beta_k,
  BType out, bElem *c) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    brick("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/helmholtz4/intermediate_gen/helmholtz43.py", VSVEC, (TILE2, TILE1, TILE0), (FOLD), b);
}
__global__ void helmholtz4_naive5(bElem (*x)[STRIDE1][STRIDE0], bElem (*alpha)[STRIDE1][STRIDE0], 
  bElem (*beta_i)[STRIDE1][STRIDE0], bElem (*beta_j)[STRIDE1][STRIDE0], bElem (*beta_k)[STRIDE1][STRIDE0], 
  bElem (*out)[STRIDE1][STRIDE0], bElem *c) {
    bElem c1 = c[0];
    bElem c2 = c[1];
    bElem h2inv = c[2];
    out[k][j][i] = c1 * alpha[k][j][i] * x[k][j][i]] -
        c2 * h2inv *
        (0.0833 * 
            (beta_i[k][j][i] * (15.0 * (x[k][j][i - 1] - x[k][j][i]) - 
                (x[k][j][i - 1] - x[k][j][i + 1])) + 
            beta_i[k][j][i + 1] * (15.0 * (x[k][j][i + 1] - x[k][j][i]) - 
                (x[k][j][i + 2] - x[k][j][i - 1])) +
            beta_j[k][j][i] * (15.0 * (x[k][j - 1][i] - x[k][j][i]) - 
                (x[k][j - 1][i] - x[k][j + 1][i])) +
            beta_j[k][j + 1][i] * (15.0 * (x[k][j + 1][i] - x[k][j][i]) -
                (x[k][j + 2][i] - x[k][j - 1][i])) +
            beta_k[k][j][i] * (15.0 * (x[k - 1][j][i] - x[k][j][i]) -
                (x[k - 2][j][i] - x[k + 1][j][i])) +
            beta_k[k + 1][j][i] * (15.0 * (x[k + 1][j][i] - x[k][j][i]) -
                (x[k + 2][j][i] - x[k - 1][j][i]))) +
        0.25 * 0.0833 * 
            ((beta_i[k][j + 1][i] - beta_i[k][j - 1][i]) *
                (x[k][j + 1][i - 1] - x[k][j + 1][i] -
                 x[k][j - 1][i - 1] + x[k][j - 1][i]) +
            (beta_i[k + 1][j][i] - beta_i[k - 1][j][i]) * 
                (x[k + 1][j][i - 1] - x[k + 1][j][i] -
                 x[k - 1][j][i - 1] + x[k - 1][j][i]) +
            (beta_j[k][j][i + 1] - beta_j[k][j][i - 1]) *
                (x[k][j - 1][i + 1] - x[k][j][i + 1] -
                 x[k][j - 1][i - 1] + x[k][j][i - 1]) +
            (beta_j[k + 1][j][i] - beta_j[k - 1][j][i]) *
                (x[k + 1][j - 1][i] - x[k + 1][j][i] -
                 x[k - 1][j - 1][i] + x[k - 1][j][i]) +
            (beta_k[k][j][i + 1] - beta_k[k][j][i - 1]) *
                (x[k - 1][j][i + 1] - x[k][j][i + 1] -
                 x[k - 1][j][i - 1] + x[k][j][i - 1]) +
            (beta_k[k][j + 1][i] - beta_k[k][j - 1][i]) *
                (x[k - 1][j + 1][i] - x[k][j + 1][i] -
                 x[k - 1][j - 1][i] + x[k][j - 1][i]) +
            (beta_i[k][j + 1][i + 1] - beta_i[k][j - 1][i + 1]) *
                (x[k][j + 1][i + 1] - x[k][j + 1][i] -
                 x[k][j - 1][i + 1] + x[k][j - 1][i]) + 
            (beta_i[k + 1][j][i + 1] - beta_i[k - 1][j][i + 1]) *
                (x[k + 1][j][i + 1] - x[k + 1][j][i] - 
                 x[k - 1][j][i + 1] + x[k - 1][j][i]) +
            (beta_j[k][j + 1][i + 1] - beta_j[k][j + 1][i - 1]) *
                (x[k][j + 1][i + 1] - x[k][j][i + 1] -
                 x[k][j + 1][i - 1] + x[k][j][i - 1]) +
            (beta_j[k + 1][j + 1][i] - beta_j[k - 1][j + 1][i]) *
                (x[k + 1][j + 1][i] - x[k + 1][j][i] -
                 x[k - 1][j + 1][i] + x[k - 1][j][i]) +
            (beta_k[k + 1][j][i + 1] - beta_k[k + 1][j][i - 1]) *
                (x[k + 1][j][i + 1] - x[k][j][i + 1] -
                 x[k + 1][j][i - 1] + x[k][j][i - 1]) +
            (beta_k[k + 1][j + 1][i] - beta_k[k + 1][j - 1][i]) *
                (x[k + 1][j + 1][i] - x[k][j + 1][i] -
                 x[k + 1][j - 1][i] + x[k][j - 1][i])
        ));
}
#define x(a, b, c) x_arr[c][b][a]
#define alpha(a, b, c) alpha_arr[c][b][a]
#define beta_i(a, b, c) beta_i_arr[c][b][a]
#define beta_j(a, b, c) beta_j_arr[c][b][a]
#define beta_k(a, b, c) beta_k_arr[c][b][a] 
#define out(a, b, c) out_arr[c][b][a]
__global__ void helmholtz4_codegen5(bElem (*x_arr)[STRIDE1][STRIDE0], bElem (*alpha_arr)[STRIDE1][STRIDE0], 
  bElem (*beta_i_arr)[STRIDE1][STRIDE0], bElem (*beta_j_arr)[STRIDE1][STRIDE0], bElem (*beta_k_arr)[STRIDE1][STRIDE0], 
  bElem (*out_arr)[STRIDE1][STRIDE0], bElem *c) {
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;
    tile("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/helmholtz4/intermediate_gen/helmholtz45.py", VSVEC, (TILE2, TILE1, VECSIZE), ("k", "j", "i"), (1, 1, VECSIZE));
}
#undef x
#undef alpha
#undef beta_i
#undef beta_j
#undef beta_k
#undef out
__global__ void helmholtz4_naive_bricks5(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType x, BType alpha,
  BType beta_i, BType beta_j, BType beta_k,
  BType out, bElem *c) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;
    bElem c1 = c[0];
    bElem c2 = c[1];
    bElem h2inv = c[2];
    out[b][k][j][i] = c1 * alpha[b][k][j][i] * x[b][k][j][i]] -
        c2 * h2inv *
        (0.0833 * 
            (beta_i[b][k][j][i] * (15.0 * (x[b][k][j][i - 1] - x[b][k][j][i]) - 
                (x[b][k][j][i - 1] - x[b][k][j][i + 1])) + 
            beta_i[b][k][j][i + 1] * (15.0 * (x[b][k][j][i + 1] - x[b][k][j][i]) - 
                (x[b][k][j][i + 2] - x[b][k][j][i - 1])) +
            beta_j[b][k][j][i] * (15.0 * (x[b][k][j - 1][i] - x[b][k][j][i]) - 
                (x[b][k][j - 1][i] - x[b][k][j + 1][i])) +
            beta_j[b][k][j + 1][i] * (15.0 * (x[b][k][j + 1][i] - x[b][k][j][i]) -
                (x[b][k][j + 2][i] - x[b][k][j - 1][i])) +
            beta_k[b][k][j][i] * (15.0 * (x[b][k - 1][j][i] - x[b][k][j][i]) -
                (x[b][k - 2][j][i] - x[b][k + 1][j][i])) +
            beta_k[b][k + 1][j][i] * (15.0 * (x[b][k + 1][j][i] - x[b][k][j][i]) -
                (x[b][k + 2][j][i] - x[b][k - 1][j][i]))) +
        0.25 * 0.0833 * 
            ((beta_i[b][k][j + 1][i] - beta_i[b][k][j - 1][i]) *
                (x[b][k][j + 1][i - 1] - x[b][k][j + 1][i] -
                 x[b][k][j - 1][i - 1] + x[b][k][j - 1][i]) +
            (beta_i[b][k + 1][j][i] - beta_i[b][k - 1][j][i]) * 
                (x[b][k + 1][j][i - 1] - x[b][k + 1][j][i] -
                 x[b][k - 1][j][i - 1] + x[b][k - 1][j][i]) +
            (beta_j[b][k][j][i + 1] - beta_j[b][k][j][i - 1]) *
                (x[b][k][j - 1][i + 1] - x[b][k][j][i + 1] -
                 x[b][k][j - 1][i - 1] + x[b][k][j][i - 1]) +
            (beta_j[b][k + 1][j][i] - beta_j[b][k - 1][j][i]) *
                (x[b][k + 1][j - 1][i] - x[b][k + 1][j][i] -
                 x[b][k - 1][j - 1][i] + x[b][k - 1][j][i]) +
            (beta_k[b][k][j][i + 1] - beta_k[b][k][j][i - 1]) *
                (x[b][k - 1][j][i + 1] - x[b][k][j][i + 1] -
                 x[b][k - 1][j][i - 1] + x[b][k][j][i - 1]) +
            (beta_k[b][k][j + 1][i] - beta_k[b][k][j - 1][i]) *
                (x[b][k - 1][j + 1][i] - x[b][k][j + 1][i] -
                 x[b][k - 1][j - 1][i] + x[b][k][j - 1][i]) +
            (beta_i[b][k][j + 1][i + 1] - beta_i[b][k][j - 1][i + 1]) *
                (x[b][k][j + 1][i + 1] - x[b][k][j + 1][i] -
                 x[b][k][j - 1][i + 1] + x[b][k][j - 1][i]) + 
            (beta_i[b][k + 1][j][i + 1] - beta_i[b][k - 1][j][i + 1]) *
                (x[b][k + 1][j][i + 1] - x[b][k + 1][j][i] - 
                 x[b][k - 1][j][i + 1] + x[b][k - 1][j][i]) +
            (beta_j[b][k][j + 1][i + 1] - beta_j[b][k][j + 1][i - 1]) *
                (x[b][k][j + 1][i + 1] - x[b][k][j][i + 1] -
                 x[b][k][j + 1][i - 1] + x[b][k][j][i - 1]) +
            (beta_j[b][k + 1][j + 1][i] - beta_j[b][k - 1][j + 1][i]) *
                (x[b][k + 1][j + 1][i] - x[b][k + 1][j][i] -
                 x[b][k - 1][j + 1][i] + x[b][k - 1][j][i]) +
            (beta_k[b][k + 1][j][i + 1] - beta_k[b][k + 1][j][i - 1]) *
                (x[b][k + 1][j][i + 1] - x[b][k][j][i + 1] -
                 x[b][k + 1][j][i - 1] + x[b][k][j][i - 1]) +
            (beta_k[b][k + 1][j + 1][i] - beta_k[b][k + 1][j - 1][i]) *
                (x[b][k + 1][j + 1][i] - x[b][k][j + 1][i] -
                 x[b][k + 1][j - 1][i] + x[b][k][j - 1][i])
        ));
} 
__global__ void helmholtz4_codegen_bricks5(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType x, BType alpha,
  BType beta_i, BType beta_j, BType beta_k,
  BType out, bElem *c) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    brick("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/helmholtz4/intermediate_gen/helmholtz45.py", VSVEC, (TILE2, TILE1, TILE0), (FOLD), b);
}
