#include <omp.h>
#include "vecscatter.h"
#include "brick.h"
#include "../../../gen/consts.h"
#include "./chebyshev.h"
#include <brick-hip.h>

__global__ void chebyshev_naive1(bElem (*Ac)[STRIDE1][STRIDE0], bElem (*Ap)[STRIDE1][STRIDE0], bElem (*Dinv)[STRIDE1][STRIDE0], bElem (*RHS)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0], bElem *c) {
  unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
  unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
  unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;
  out[k][j][i] = 
    Ac[k][j][i] + c[0] * (Ac[k][j][i] + Ap[k][j][i]) +
    c[1] * Dinv[k][j][i] *
    (RHS[k][j][i] +
    (Ac[k][j][i] + 
    c[2] * 
      (
        0.03 * (Ac[k - 1][j - 1][i - 1] + Ac[k - 1][j - 1][i + 1] +
                Ac[k - 1][j + 1][i - 1] + Ac[k - 1][j + 1][i + 1] +
                Ac[k + 1][j - 1][i - 1] + Ac[k + 1][j - 1][i + 1] +
                Ac[k + 1][j + 1][i - 1] + Ac[k + 1][j + 1][i + 1]) +
        0.1 * (Ac[k - 1][j - 1][i] + Ac[k - 1][j][i - 1] +
               Ac[k - 1][j][i + 1] + Ac[k - 1][j + 1][i] +
               Ac[k][j - 1][i - 1] + Ac[k][j - 1][i + 1] +
               Ac[k][j + 1][i - 1] + Ac[k][j + 1][i + 1] +
               Ac[k + 1][j - 1][i] + Ac[k + 1][j][i - 1] +
               Ac[k + 1][j][i + 1] + Ac[k + 1][j + 1][i]) +
        0.46 * (Ac[k - 1][j][i] + Ac[k][j - 1][i] + Ac[k][j][i - 1] +
                Ac[k + 1][j][i] + Ac[k][j + 1][i] + Ac[k][j][i + 1]) +
        4.26 * Ac[k][j][i]
               )));
}
__global__ void chebyshev_naive_bricks1(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType Ac, BType Ap, BType Dinv, BType RHS, BType out, bElem *c) {
  unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
  unsigned i = threadIdx.x;
  unsigned j = threadIdx.y;
  unsigned k = threadIdx.z;
  out[b][k][j][i] = 
    Ac[b][k][j][i] + c[0] * (Ac[b][k][j][i] + Ap[b][k][j][i]) +
    c[1] * Dinv[b][k][j][i] *
    (RHS[b][k][j][i] +
    (Ac[b][k][j][i] + 
    c[2] * 
      (
        0.03 * (Ac[b][k - 1][j - 1][i - 1] + Ac[b][k - 1][j - 1][i + 1] +
                Ac[b][k - 1][j + 1][i - 1] + Ac[b][k - 1][j + 1][i + 1] +
                Ac[b][k + 1][j - 1][i - 1] + Ac[b][k + 1][j - 1][i + 1] +
                Ac[b][k + 1][j + 1][i - 1] + Ac[b][k + 1][j + 1][i + 1]) +
        0.1 * (Ac[b][k - 1][j - 1][i] + Ac[b][k - 1][j][i - 1] +
               Ac[b][k - 1][j][i + 1] + Ac[b][k - 1][j + 1][i] +
               Ac[b][k][j - 1][i - 1] + Ac[b][k][j - 1][i + 1] +
               Ac[b][k][j + 1][i - 1] + Ac[b][k][j + 1][i + 1] +
               Ac[b][k + 1][j - 1][i] + Ac[b][k + 1][j][i - 1] +
               Ac[b][k + 1][j][i + 1] + Ac[b][k + 1][j + 1][i]) +
        0.46 * (Ac[b][k - 1][j][i] + Ac[b][k][j - 1][i] + Ac[b][k][j][i - 1] +
                Ac[b][k + 1][j][i] + Ac[b][k][j + 1][i] + Ac[b][k][j][i + 1]) +
        4.26 * Ac[b][k][j][i]
               )));
}
}
__global__ void chebyshev_codegen_bricks1(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType Ac, BType Ap, BType Dinv, BType RHS, BType out, bElem *c) {
  unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
  brick("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/chebyshev/intermediate_gen/chebyshev1.py", VSVEC, (TILE2, TILE1, TILE0), (FOLD), b);
}
#define Ac(a, b, c) Ac_arr[c][b][a]
#define Ap(a, b, c) Ap_arr[c][b][a]
#define Dinv(a, b, c) Dinv_arr[c][b][a]
#define RHS(a, b, c) RHS_arr[c][b][a]
#define out(a, b, c) out_arr[c][b][a]
__global__ void chebyshev_codegen1(bElem (*Ac_arr)[STRIDE1][STRIDE0], bElem (*Ap_arr)[STRIDE1][STRIDE0], bElem (*Dinv_arr)[STRIDE1][STRIDE0], bElem (*RHS_arr)[STRIDE1][STRIDE0], bElem (*out_arr)[STRIDE1][STRIDE0], bElem *c) {
  long k = OFF2 + (blockIdx.z * TILE2);
  long j = OFF1 + (blockIdx.y * TILE1);
  long i = OFF0 + (blockIdx.x * TILE0);
  tile("/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/chebyshev/intermediate_gen/chebyshev1.py", VSVEC, (TILE2, TILE1, VECSIZE), ("k", "j", "i"), (1, 1, VECSIZE));
}
#undef Ac
#undef Ap
#undef Dinv
#undef RHS
#undef out
