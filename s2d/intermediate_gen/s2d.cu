#include <omp.h>
#include "vecscatter.h"
#include "brick.h"
// #include "../out/laplacian-stencils.h"
#include "../../../gen/consts.h"
#include "./s2d.h"
#include <brick-hip.h>

__global__ void s2d_naive2(bElem (*in)[STRIDE0], bElem (*out)[STRIDE0], bElem* dev_coeff) {
    const size_t radius = 2;
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    bElem temp = dev_coeff[0] * in[j][i];
    #pragma unroll
    for (int a = 1; a <= radius; a++) {
        temp += dev_coeff[a] * (
            in[j][i + a] + in[j + a][i] +
            in[j][i - a] + in[j - a][i]);
    }
    out[j][i] = temp;
}
__global__ void s2d_naive_bricks2(unsigned (*grid)[NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem *dev_coeff) {
    unsigned b = grid[blockIdx.y + GB1][blockIdx.x + GB0];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    bOut[b][j][i] = dev_coeff[0] * bIn[b][j][i];
    const size_t radius = 2;
    #pragma unroll
    for (int a = 1; a <= radius; a++) {
        bOut[b][j][i] += dev_coeff[a] * (
            bIn[b][j][i + a] + bIn[b][j + a][i] +
            bIn[b][j][i - a] + bIn[b][j - a][i]
        );
    }
}
__global__ void s2d_codegen_bricks2(unsigned (*grid)[NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem *dev_coeff) {
    unsigned b = grid[blockIdx.y + GB1][blockIdx.x + GB0];
    brick("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/s2d/intermediate_gen/laplacian2d2.py", VSVEC, (TILE1, TILE0), (FOLD), b);
}
#define bIn(a, b) arr_in[c][b]
#define bOut(a, b) arr_out[c][b]
__global__ void laplacian_codegen(bElem (*arr_in)[STRIDE0], bElem (*arr_out)[STRIDE0], bElem *dev_coeff) {
    long j = OFF1 + (blockIdx.y * TILE1);
    long i = OFF0 + (blockIdx.x * VECSIZE);
    tile("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/s2d/intermediate_gen/laplacian2d2.py", VSVEC, (TILE1, VECSIZE), ("j", "i"), (1, VECSIZE));
}
#undef bIn
#undef bOut
