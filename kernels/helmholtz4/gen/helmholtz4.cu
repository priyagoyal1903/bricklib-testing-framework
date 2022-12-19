# 1 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/helmholtz4/intermediate_gen/helmholtz4.cu"
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
# 1 "VSTile-helmholtz42.py-HIP-8x8x64" 1
{
  bElem buf0[64];
  bElem buf1[64];
  bElem buf2[64];
  bElem buf3[64];
  bElem buf4[64];
  bElem buf5[64];
  bElem buf6[64];
  bElem buf7[64];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf0[0 + rel] = 0;
                buf1[0 + rel] = 0;
                buf2[0 + rel] = 0;
                buf3[0 + rel] = 0;
                buf4[0 + rel] = 0;
                buf5[0 + rel] = 0;
                buf6[0 + rel] = 0;
                buf7[0 + rel] = 0;
              }
              _cg_rel1 += 1;
            }
          }
          _cg_rel2 += 8;
        }
      }
    }
    {
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 7; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf2[0 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + -2);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf2[7] -= x(i + hipThreadIdx_x, j + 7, k + -2);
      }
      {
        buf5[0] -= x(i + hipThreadIdx_x + -1, j, k + -1);
      }
      {
        buf5[1] -= x(i + hipThreadIdx_x + -1, j + 1, k + -1);
        buf7[0] -= x(i + hipThreadIdx_x, j + -1, k + -1);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf2[0 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf2[8 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf3[0 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf5[0 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf5[2 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + -1);
            buf7[0 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf7[1 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf2[6] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + -1);
        buf2[14] -= x(i + hipThreadIdx_x, j + 6, k + -1);
        buf3[6] += x(i + hipThreadIdx_x, j + 6, k + -1);
        buf5[6] += x(i + hipThreadIdx_x, j + 6, k + -1);
        buf7[6] += x(i + hipThreadIdx_x, j + 6, k + -1);
        buf7[7] -= x(i + hipThreadIdx_x, j + 6, k + -1);
      }
      {
        buf2[7] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + -1);
        buf2[15] -= x(i + hipThreadIdx_x, j + 7, k + -1);
        buf3[7] += x(i + hipThreadIdx_x, j + 7, k + -1);
        buf5[7] += x(i + hipThreadIdx_x, j + 7, k + -1);
        buf7[7] += x(i + hipThreadIdx_x, j + 7, k + -1);
      }
      {
        buf4[0] -= x(i + hipThreadIdx_x + -1, j + -1, k);
        buf6[0] -= x(i + hipThreadIdx_x + -1, j + -1, k);
      }
      {
        buf4[1] -= x(i + hipThreadIdx_x + -1, j, k);
        buf6[1] -= x(i + hipThreadIdx_x + -1, j, k);
        buf6[0] += x(i + hipThreadIdx_x + -1, j, k);
        buf5[8] -= x(i + hipThreadIdx_x + -1, j, k);
      }
      {
        buf0[0] += 15.0 * x(i + hipThreadIdx_x, j + -1, k);
        buf0[0] -= x(i + hipThreadIdx_x, j + -1, k);
        buf1[0] += x(i + hipThreadIdx_x, j + -1, k);
        buf4[0] += x(i + hipThreadIdx_x + -1, j + 1, k);
        buf4[0] += x(i + hipThreadIdx_x, j + -1, k);
        buf4[2] -= x(i + hipThreadIdx_x + -1, j + 1, k);
        buf6[0] += x(i + hipThreadIdx_x + 1, j + -1, k);
        buf6[2] -= x(i + hipThreadIdx_x + -1, j + 1, k);
        buf6[1] += x(i + hipThreadIdx_x + -1, j + 1, k);
        buf5[9] -= x(i + hipThreadIdx_x + -1, j + 1, k);
        buf7[8] -= x(i + hipThreadIdx_x, j + -1, k);
      }
      {
        buf0[1] += 15.0 * x(i + hipThreadIdx_x, j, k);
        buf0[1] -= x(i + hipThreadIdx_x, j, k);
        buf0[0] -= 15.0 * x(i + hipThreadIdx_x, j, k);
        buf1[1] += x(i + hipThreadIdx_x, j, k);
        buf1[0] -= 15.0 * x(i + hipThreadIdx_x, j, k);
        buf4[1] += x(i + hipThreadIdx_x + -1, j + 2, k);
        buf4[1] += x(i + hipThreadIdx_x, j, k);
        buf4[3] -= x(i + hipThreadIdx_x + -1, j + 2, k);
        buf6[1] += x(i + hipThreadIdx_x + 1, j, k);
        buf6[0] -= x(i + hipThreadIdx_x + 1, j, k);
        buf6[3] -= x(i + hipThreadIdx_x + -1, j + 2, k);
        buf6[2] += x(i + hipThreadIdx_x + -1, j + 2, k);
        buf2[0] -= 15.0 * x(i + hipThreadIdx_x, j, k);
        buf2[8] += 15.0 * x(i + hipThreadIdx_x, j, k);
        buf2[16] -= x(i + hipThreadIdx_x, j, k);
        buf3[0] -= 15.0 * x(i + hipThreadIdx_x, j, k);
        buf3[8] += x(i + hipThreadIdx_x, j, k);
        buf5[8] += x(i + hipThreadIdx_x, j, k);
        buf5[10] -= x(i + hipThreadIdx_x + -1, j + 2, k);
        buf7[8] += x(i + hipThreadIdx_x, j, k);
        buf7[9] -= x(i + hipThreadIdx_x, j, k);
      }
      {
        buf0[2] += 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf0[2] -= x(i + hipThreadIdx_x, j + 1, k);
        buf0[1] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf0[0] += x(i + hipThreadIdx_x, j + 1, k);
        buf1[2] += x(i + hipThreadIdx_x, j + 1, k);
        buf1[1] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf1[0] += 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf4[2] += x(i + hipThreadIdx_x + -1, j + 3, k);
        buf4[2] += x(i + hipThreadIdx_x, j + 1, k);
        buf4[0] -= x(i + hipThreadIdx_x, j + 1, k);
        buf4[4] -= x(i + hipThreadIdx_x + -1, j + 3, k);
        buf6[2] += x(i + hipThreadIdx_x + 1, j + 1, k);
        buf6[1] -= x(i + hipThreadIdx_x + 1, j + 1, k);
        buf6[4] -= x(i + hipThreadIdx_x + -1, j + 3, k);
        buf6[3] += x(i + hipThreadIdx_x + -1, j + 3, k);
        buf2[1] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf2[9] += 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf2[17] -= x(i + hipThreadIdx_x, j + 1, k);
        buf3[1] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf3[9] += x(i + hipThreadIdx_x, j + 1, k);
        buf5[9] += x(i + hipThreadIdx_x, j + 1, k);
        buf5[11] -= x(i + hipThreadIdx_x + -1, j + 3, k);
        buf7[9] += x(i + hipThreadIdx_x, j + 1, k);
        buf7[10] -= x(i + hipThreadIdx_x, j + 1, k);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 3; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[3 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf0[3 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf0[2 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf0[1 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf1[3 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf1[2 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf1[1 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf1[0 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf4[3 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k);
            buf4[3 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf4[1 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf4[5 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k);
            buf6[3 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k);
            buf6[2 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k);
            buf6[5 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k);
            buf6[4 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k);
            buf2[2 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf2[10 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf2[18 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf3[2 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf3[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf5[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf5[12 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k);
            buf7[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf7[11 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[6] += 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf0[6] -= x(i + hipThreadIdx_x, j + 5, k);
        buf0[5] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf0[4] += x(i + hipThreadIdx_x, j + 5, k);
        buf1[6] += x(i + hipThreadIdx_x, j + 5, k);
        buf1[5] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf1[4] += 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf1[3] -= x(i + hipThreadIdx_x, j + 5, k);
        buf4[6] += x(i + hipThreadIdx_x + -1, j + 7, k);
        buf4[6] += x(i + hipThreadIdx_x, j + 5, k);
        buf4[4] -= x(i + hipThreadIdx_x, j + 5, k);
        buf6[6] += x(i + hipThreadIdx_x + 1, j + 5, k);
        buf6[5] -= x(i + hipThreadIdx_x + 1, j + 5, k);
        buf6[7] += x(i + hipThreadIdx_x + -1, j + 7, k);
        buf2[5] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf2[13] += 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf2[21] -= x(i + hipThreadIdx_x, j + 5, k);
        buf3[5] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf3[13] += x(i + hipThreadIdx_x, j + 5, k);
        buf5[13] += x(i + hipThreadIdx_x, j + 5, k);
        buf5[15] -= x(i + hipThreadIdx_x + -1, j + 7, k);
        buf7[13] += x(i + hipThreadIdx_x, j + 5, k);
        buf7[14] -= x(i + hipThreadIdx_x, j + 5, k);
      }
      {
        buf0[7] += 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf0[7] -= x(i + hipThreadIdx_x, j + 6, k);
        buf0[6] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf0[5] += x(i + hipThreadIdx_x, j + 6, k);
        buf1[7] += x(i + hipThreadIdx_x, j + 6, k);
        buf1[6] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf1[5] += 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf1[4] -= x(i + hipThreadIdx_x, j + 6, k);
        buf4[7] += x(i + hipThreadIdx_x + -1, j + 8, k);
        buf4[7] += x(i + hipThreadIdx_x, j + 6, k);
        buf4[5] -= x(i + hipThreadIdx_x, j + 6, k);
        buf6[7] += x(i + hipThreadIdx_x + 1, j + 6, k);
        buf6[6] -= x(i + hipThreadIdx_x + 1, j + 6, k);
        buf2[6] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf2[14] += 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf2[22] -= x(i + hipThreadIdx_x, j + 6, k);
        buf3[6] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf3[14] += x(i + hipThreadIdx_x, j + 6, k);
        buf5[14] += x(i + hipThreadIdx_x, j + 6, k);
        buf7[14] += x(i + hipThreadIdx_x, j + 6, k);
        buf7[15] -= x(i + hipThreadIdx_x, j + 6, k);
      }
      {
        buf0[7] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k);
        buf0[6] += x(i + hipThreadIdx_x, j + 7, k);
        buf1[7] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k);
        buf1[6] += 15.0 * x(i + hipThreadIdx_x, j + 7, k);
        buf1[5] -= x(i + hipThreadIdx_x, j + 7, k);
        buf2[7] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k);
        buf2[15] += 15.0 * x(i + hipThreadIdx_x, j + 7, k);
        buf2[23] -= x(i + hipThreadIdx_x, j + 7, k);
        buf3[7] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k);
        buf3[15] += x(i + hipThreadIdx_x, j + 7, k);
        buf6[7] -= x(i + hipThreadIdx_x + 1, j + 7, k);
        buf4[6] -= x(i + hipThreadIdx_x, j + 7, k);
        buf5[15] += x(i + hipThreadIdx_x, j + 7, k);
        buf7[15] += x(i + hipThreadIdx_x, j + 7, k);
      }
      {
        buf0[7] += x(i + hipThreadIdx_x, j + 8, k);
        buf1[7] += 15.0 * x(i + hipThreadIdx_x, j + 8, k);
        buf1[6] -= x(i + hipThreadIdx_x, j + 8, k);
        buf4[7] -= x(i + hipThreadIdx_x, j + 8, k);
      }
      {
        buf1[7] -= x(i + hipThreadIdx_x, j + 9, k);
      }
      {
        buf4[8] -= x(i + hipThreadIdx_x + -1, j + -1, k + 1);
        buf6[8] -= x(i + hipThreadIdx_x + -1, j + -1, k + 1);
      }
      {
        buf4[9] -= x(i + hipThreadIdx_x + -1, j, k + 1);
        buf6[9] -= x(i + hipThreadIdx_x + -1, j, k + 1);
        buf6[8] += x(i + hipThreadIdx_x + -1, j, k + 1);
        buf5[0] += x(i + hipThreadIdx_x + -1, j, k + 1);
        buf5[16] -= x(i + hipThreadIdx_x + -1, j, k + 1);
      }
      {
        buf0[8] += 15.0 * x(i + hipThreadIdx_x, j + -1, k + 1);
        buf0[8] -= x(i + hipThreadIdx_x, j + -1, k + 1);
        buf1[8] += x(i + hipThreadIdx_x, j + -1, k + 1);
        buf4[8] += x(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf4[8] += x(i + hipThreadIdx_x, j + -1, k + 1);
        buf4[10] -= x(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf6[8] += x(i + hipThreadIdx_x + 1, j + -1, k + 1);
        buf6[10] -= x(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf6[9] += x(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf5[1] += x(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf5[17] -= x(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf7[0] += x(i + hipThreadIdx_x, j + -1, k + 1);
        buf7[16] -= x(i + hipThreadIdx_x, j + -1, k + 1);
      }
      {
        buf0[9] += 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf0[9] -= x(i + hipThreadIdx_x, j, k + 1);
        buf0[8] -= 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf1[9] += x(i + hipThreadIdx_x, j, k + 1);
        buf1[8] -= 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf4[9] += x(i + hipThreadIdx_x + -1, j + 2, k + 1);
        buf4[9] += x(i + hipThreadIdx_x, j, k + 1);
        buf4[11] -= x(i + hipThreadIdx_x + -1, j + 2, k + 1);
        buf6[9] += x(i + hipThreadIdx_x + 1, j, k + 1);
        buf6[8] -= x(i + hipThreadIdx_x + 1, j, k + 1);
        buf6[11] -= x(i + hipThreadIdx_x + -1, j + 2, k + 1);
        buf6[10] += x(i + hipThreadIdx_x + -1, j + 2, k + 1);
        buf2[8] -= 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf2[16] += 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf2[24] -= x(i + hipThreadIdx_x, j, k + 1);
        buf2[0] += x(i + hipThreadIdx_x, j, k + 1);
        buf3[8] -= 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf3[16] += x(i + hipThreadIdx_x, j, k + 1);
        buf3[0] += 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf5[16] += x(i + hipThreadIdx_x, j, k + 1);
        buf5[0] -= x(i + hipThreadIdx_x, j, k + 1);
        buf5[2] += x(i + hipThreadIdx_x + -1, j + 2, k + 1);
        buf5[18] -= x(i + hipThreadIdx_x + -1, j + 2, k + 1);
        buf7[16] += x(i + hipThreadIdx_x, j, k + 1);
        buf7[0] -= x(i + hipThreadIdx_x, j, k + 1);
        buf7[1] += x(i + hipThreadIdx_x, j, k + 1);
        buf7[17] -= x(i + hipThreadIdx_x, j, k + 1);
      }
      {
        buf0[10] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[10] -= x(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[9] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[8] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf1[10] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf1[9] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf1[8] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf4[10] += x(i + hipThreadIdx_x + -1, j + 3, k + 1);
        buf4[10] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf4[8] -= x(i + hipThreadIdx_x, j + 1, k + 1);
        buf4[12] -= x(i + hipThreadIdx_x + -1, j + 3, k + 1);
        buf6[10] += x(i + hipThreadIdx_x + 1, j + 1, k + 1);
        buf6[9] -= x(i + hipThreadIdx_x + 1, j + 1, k + 1);
        buf6[12] -= x(i + hipThreadIdx_x + -1, j + 3, k + 1);
        buf6[11] += x(i + hipThreadIdx_x + -1, j + 3, k + 1);
        buf2[9] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf2[17] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf2[25] -= x(i + hipThreadIdx_x, j + 1, k + 1);
        buf2[1] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf3[9] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf3[17] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf3[1] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf5[17] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf5[1] -= x(i + hipThreadIdx_x, j + 1, k + 1);
        buf5[3] += x(i + hipThreadIdx_x + -1, j + 3, k + 1);
        buf5[19] -= x(i + hipThreadIdx_x + -1, j + 3, k + 1);
        buf7[17] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf7[1] -= x(i + hipThreadIdx_x, j + 1, k + 1);
        buf7[2] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf7[18] -= x(i + hipThreadIdx_x, j + 1, k + 1);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 3; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[11 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf0[11 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf0[10 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf0[9 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf1[11 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf1[10 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf1[9 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf1[8 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf4[11 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 1);
            buf4[11 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf4[9 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf4[13 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 1);
            buf6[11 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1);
            buf6[10 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1);
            buf6[13 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 1);
            buf6[12 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 1);
            buf2[10 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf2[18 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf2[26 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf2[2 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf3[10 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf3[18 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf3[2 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf5[18 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf5[2 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf5[4 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 1);
            buf5[20 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 1);
            buf7[18 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf7[2 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf7[3 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf7[19 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[14] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[14] -= x(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[13] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[12] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf1[14] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf1[13] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf1[12] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf1[11] -= x(i + hipThreadIdx_x, j + 5, k + 1);
        buf4[14] += x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf4[14] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf4[12] -= x(i + hipThreadIdx_x, j + 5, k + 1);
        buf6[14] += x(i + hipThreadIdx_x + 1, j + 5, k + 1);
        buf6[13] -= x(i + hipThreadIdx_x + 1, j + 5, k + 1);
        buf6[15] += x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf2[13] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf2[21] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf2[29] -= x(i + hipThreadIdx_x, j + 5, k + 1);
        buf2[5] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf3[13] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf3[21] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf3[5] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf5[21] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf5[5] -= x(i + hipThreadIdx_x, j + 5, k + 1);
        buf5[7] += x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf5[23] -= x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf7[21] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf7[5] -= x(i + hipThreadIdx_x, j + 5, k + 1);
        buf7[6] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf7[22] -= x(i + hipThreadIdx_x, j + 5, k + 1);
      }
      {
        buf0[15] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[15] -= x(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[14] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[13] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf1[15] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf1[14] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf1[13] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf1[12] -= x(i + hipThreadIdx_x, j + 6, k + 1);
        buf4[15] += x(i + hipThreadIdx_x + -1, j + 8, k + 1);
        buf4[15] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf4[13] -= x(i + hipThreadIdx_x, j + 6, k + 1);
        buf6[15] += x(i + hipThreadIdx_x + 1, j + 6, k + 1);
        buf6[14] -= x(i + hipThreadIdx_x + 1, j + 6, k + 1);
        buf2[14] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf2[22] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf2[30] -= x(i + hipThreadIdx_x, j + 6, k + 1);
        buf2[6] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf3[14] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf3[22] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf3[6] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf5[22] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf5[6] -= x(i + hipThreadIdx_x, j + 6, k + 1);
        buf7[22] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf7[6] -= x(i + hipThreadIdx_x, j + 6, k + 1);
        buf7[7] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf7[23] -= x(i + hipThreadIdx_x, j + 6, k + 1);
      }
      {
        buf0[15] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[14] += x(i + hipThreadIdx_x, j + 7, k + 1);
        buf1[15] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf1[14] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf1[13] -= x(i + hipThreadIdx_x, j + 7, k + 1);
        buf2[15] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf2[23] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf2[31] -= x(i + hipThreadIdx_x, j + 7, k + 1);
        buf2[7] += x(i + hipThreadIdx_x, j + 7, k + 1);
        buf3[15] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf3[23] += x(i + hipThreadIdx_x, j + 7, k + 1);
        buf3[7] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf6[15] -= x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf4[14] -= x(i + hipThreadIdx_x, j + 7, k + 1);
        buf5[23] += x(i + hipThreadIdx_x, j + 7, k + 1);
        buf5[7] -= x(i + hipThreadIdx_x, j + 7, k + 1);
        buf7[23] += x(i + hipThreadIdx_x, j + 7, k + 1);
        buf7[7] -= x(i + hipThreadIdx_x, j + 7, k + 1);
      }
      {
        buf0[15] += x(i + hipThreadIdx_x, j + 8, k + 1);
        buf1[15] += 15.0 * x(i + hipThreadIdx_x, j + 8, k + 1);
        buf1[14] -= x(i + hipThreadIdx_x, j + 8, k + 1);
        buf4[15] -= x(i + hipThreadIdx_x, j + 8, k + 1);
      }
      {
        buf1[15] -= x(i + hipThreadIdx_x, j + 9, k + 1);
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 4; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            buf4[16 + rel] -= x(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 2);
            buf6[16 + rel] -= x(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 2);
          }
          {
            buf4[17 + rel] -= x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
            buf6[17 + rel] -= x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
            buf6[16 + rel] += x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
            buf5[8 + rel] += x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
            buf5[24 + rel] -= x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
          }
          {
            buf0[16 + rel] += 15.0 * x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2);
            buf0[16 + rel] -= x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2);
            buf1[16 + rel] += x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2);
            buf4[16 + rel] += x(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2);
            buf4[16 + rel] += x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2);
            buf4[18 + rel] -= x(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2);
            buf6[16 + rel] += x(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 2);
            buf6[18 + rel] -= x(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2);
            buf6[17 + rel] += x(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2);
            buf5[9 + rel] += x(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2);
            buf5[25 + rel] -= x(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2);
            buf7[8 + rel] += x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2);
            buf7[24 + rel] -= x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2);
          }
          {
            buf0[17 + rel] += 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf0[17 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf0[16 + rel] -= 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf1[17 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf1[16 + rel] -= 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf4[17 + rel] += x(i + hipThreadIdx_x + -1, j + 2, k + _cg_idx2 + 2);
            buf4[17 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf4[19 + rel] -= x(i + hipThreadIdx_x + -1, j + 2, k + _cg_idx2 + 2);
            buf6[17 + rel] += x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf6[16 + rel] -= x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf6[19 + rel] -= x(i + hipThreadIdx_x + -1, j + 2, k + _cg_idx2 + 2);
            buf6[18 + rel] += x(i + hipThreadIdx_x + -1, j + 2, k + _cg_idx2 + 2);
            buf2[16 + rel] -= 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf2[24 + rel] += 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf2[32 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf2[8 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf3[16 + rel] -= 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf3[24 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf3[8 + rel] += 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf3[0 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf5[24 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf5[8 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf5[10 + rel] += x(i + hipThreadIdx_x + -1, j + 2, k + _cg_idx2 + 2);
            buf5[26 + rel] -= x(i + hipThreadIdx_x + -1, j + 2, k + _cg_idx2 + 2);
            buf7[24 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf7[8 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf7[9 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf7[25 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
          }
          {
            buf0[18 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf0[18 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf0[17 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf0[16 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf1[18 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf1[17 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf1[16 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf4[18 + rel] += x(i + hipThreadIdx_x + -1, j + 3, k + _cg_idx2 + 2);
            buf4[18 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf4[16 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf4[20 + rel] -= x(i + hipThreadIdx_x + -1, j + 3, k + _cg_idx2 + 2);
            buf6[18 + rel] += x(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2);
            buf6[17 + rel] -= x(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2);
            buf6[20 + rel] -= x(i + hipThreadIdx_x + -1, j + 3, k + _cg_idx2 + 2);
            buf6[19 + rel] += x(i + hipThreadIdx_x + -1, j + 3, k + _cg_idx2 + 2);
            buf2[17 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf2[25 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf2[33 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf2[9 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf3[17 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf3[25 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf3[9 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf3[1 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf5[25 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf5[9 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf5[11 + rel] += x(i + hipThreadIdx_x + -1, j + 3, k + _cg_idx2 + 2);
            buf5[27 + rel] -= x(i + hipThreadIdx_x + -1, j + 3, k + _cg_idx2 + 2);
            buf7[25 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf7[9 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf7[10 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf7[26 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
          }
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 3; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf0[19 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[19 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[18 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[17 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf1[19 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf1[18 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf1[17 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf1[16 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf4[19 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + _cg_idx2 + 2);
                buf4[19 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf4[17 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf4[21 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + _cg_idx2 + 2);
                buf6[19 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf6[18 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf6[21 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + _cg_idx2 + 2);
                buf6[20 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + _cg_idx2 + 2);
                buf2[18 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf2[26 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf2[34 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf2[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf3[18 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf3[26 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf3[10 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf3[2 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf5[26 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf5[10 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf5[12 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + _cg_idx2 + 2);
                buf5[28 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + _cg_idx2 + 2);
                buf7[26 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf7[10 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf7[11 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf7[27 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
              }
              _cg_rel1 += 1;
            }
          }
          {
            buf0[22 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf0[22 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf0[21 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf0[20 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf1[22 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf1[21 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf1[20 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf1[19 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf4[22 + rel] += x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf4[22 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf4[20 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf6[22 + rel] += x(i + hipThreadIdx_x + 1, j + 5, k + _cg_idx2 + 2);
            buf6[21 + rel] -= x(i + hipThreadIdx_x + 1, j + 5, k + _cg_idx2 + 2);
            buf6[23 + rel] += x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf2[21 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf2[29 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf2[37 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf2[13 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf3[21 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf3[29 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf3[13 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf3[5 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf5[29 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf5[13 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf5[15 + rel] += x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf5[31 + rel] -= x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf7[29 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf7[13 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf7[14 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf7[30 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
          }
          {
            buf0[23 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf0[23 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf0[22 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf0[21 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf1[23 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf1[22 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf1[21 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf1[20 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf4[23 + rel] += x(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 2);
            buf4[23 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf4[21 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf6[23 + rel] += x(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2);
            buf6[22 + rel] -= x(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2);
            buf2[22 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf2[30 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf2[38 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf2[14 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf3[22 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf3[30 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf3[14 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf3[6 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf5[30 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf5[14 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf7[30 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf7[14 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf7[15 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf7[31 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
          }
          {
            buf0[23 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf0[22 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf1[23 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf1[22 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf1[21 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf2[23 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf2[31 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf2[39 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf2[15 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf3[23 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf3[31 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf3[15 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf3[7 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf6[23 + rel] -= x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf4[22 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf5[31 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf5[15 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf7[31 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf7[15 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
          }
          {
            buf0[23 + rel] += x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2);
            buf1[23 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2);
            buf1[22 + rel] -= x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2);
            buf4[23 + rel] -= x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2);
          }
          {
            buf1[23 + rel] -= x(i + hipThreadIdx_x, j + 9, k + _cg_idx2 + 2);
          }
          _cg_rel2 += 8;
        }
      }
      {
        buf4[48] -= x(i + hipThreadIdx_x + -1, j + -1, k + 6);
        buf6[48] -= x(i + hipThreadIdx_x + -1, j + -1, k + 6);
      }
      {
        buf4[49] -= x(i + hipThreadIdx_x + -1, j, k + 6);
        buf6[49] -= x(i + hipThreadIdx_x + -1, j, k + 6);
        buf6[48] += x(i + hipThreadIdx_x + -1, j, k + 6);
        buf5[40] += x(i + hipThreadIdx_x + -1, j, k + 6);
        buf5[56] -= x(i + hipThreadIdx_x + -1, j, k + 6);
      }
      {
        buf0[48] += 15.0 * x(i + hipThreadIdx_x, j + -1, k + 6);
        buf0[48] -= x(i + hipThreadIdx_x, j + -1, k + 6);
        buf1[48] += x(i + hipThreadIdx_x, j + -1, k + 6);
        buf4[48] += x(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf4[48] += x(i + hipThreadIdx_x, j + -1, k + 6);
        buf4[50] -= x(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf6[48] += x(i + hipThreadIdx_x + 1, j + -1, k + 6);
        buf6[50] -= x(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf6[49] += x(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf5[41] += x(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf5[57] -= x(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf7[40] += x(i + hipThreadIdx_x, j + -1, k + 6);
        buf7[56] -= x(i + hipThreadIdx_x, j + -1, k + 6);
      }
      {
        buf0[49] += 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf0[49] -= x(i + hipThreadIdx_x, j, k + 6);
        buf0[48] -= 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf1[49] += x(i + hipThreadIdx_x, j, k + 6);
        buf1[48] -= 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf4[49] += x(i + hipThreadIdx_x + -1, j + 2, k + 6);
        buf4[49] += x(i + hipThreadIdx_x, j, k + 6);
        buf4[51] -= x(i + hipThreadIdx_x + -1, j + 2, k + 6);
        buf6[49] += x(i + hipThreadIdx_x + 1, j, k + 6);
        buf6[48] -= x(i + hipThreadIdx_x + 1, j, k + 6);
        buf6[51] -= x(i + hipThreadIdx_x + -1, j + 2, k + 6);
        buf6[50] += x(i + hipThreadIdx_x + -1, j + 2, k + 6);
        buf2[48] -= 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf2[56] += 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf2[40] += x(i + hipThreadIdx_x, j, k + 6);
        buf3[48] -= 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf3[56] += x(i + hipThreadIdx_x, j, k + 6);
        buf3[40] += 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf3[32] -= x(i + hipThreadIdx_x, j, k + 6);
        buf5[56] += x(i + hipThreadIdx_x, j, k + 6);
        buf5[40] -= x(i + hipThreadIdx_x, j, k + 6);
        buf5[42] += x(i + hipThreadIdx_x + -1, j + 2, k + 6);
        buf5[58] -= x(i + hipThreadIdx_x + -1, j + 2, k + 6);
        buf7[56] += x(i + hipThreadIdx_x, j, k + 6);
        buf7[40] -= x(i + hipThreadIdx_x, j, k + 6);
        buf7[41] += x(i + hipThreadIdx_x, j, k + 6);
        buf7[57] -= x(i + hipThreadIdx_x, j, k + 6);
      }
      {
        buf0[50] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[50] -= x(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[49] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[48] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf1[50] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf1[49] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf1[48] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf4[50] += x(i + hipThreadIdx_x + -1, j + 3, k + 6);
        buf4[50] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf4[48] -= x(i + hipThreadIdx_x, j + 1, k + 6);
        buf4[52] -= x(i + hipThreadIdx_x + -1, j + 3, k + 6);
        buf6[50] += x(i + hipThreadIdx_x + 1, j + 1, k + 6);
        buf6[49] -= x(i + hipThreadIdx_x + 1, j + 1, k + 6);
        buf6[52] -= x(i + hipThreadIdx_x + -1, j + 3, k + 6);
        buf6[51] += x(i + hipThreadIdx_x + -1, j + 3, k + 6);
        buf2[49] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf2[57] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf2[41] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf3[49] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf3[57] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf3[41] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf3[33] -= x(i + hipThreadIdx_x, j + 1, k + 6);
        buf5[57] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf5[41] -= x(i + hipThreadIdx_x, j + 1, k + 6);
        buf5[43] += x(i + hipThreadIdx_x + -1, j + 3, k + 6);
        buf5[59] -= x(i + hipThreadIdx_x + -1, j + 3, k + 6);
        buf7[57] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf7[41] -= x(i + hipThreadIdx_x, j + 1, k + 6);
        buf7[42] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf7[58] -= x(i + hipThreadIdx_x, j + 1, k + 6);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 3; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[51 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf0[51 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf0[50 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf0[49 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf1[51 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf1[50 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf1[49 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf1[48 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf4[51 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 6);
            buf4[51 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf4[49 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf4[53 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 6);
            buf6[51 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6);
            buf6[50 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6);
            buf6[53 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 6);
            buf6[52 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 6);
            buf2[50 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf2[58 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf2[42 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf3[50 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf3[58 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf3[42 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf3[34 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf5[58 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf5[42 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf5[44 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 6);
            buf5[60 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 6);
            buf7[58 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf7[42 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf7[43 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf7[59 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[54] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[54] -= x(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[53] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[52] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf1[54] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf1[53] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf1[52] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf1[51] -= x(i + hipThreadIdx_x, j + 5, k + 6);
        buf4[54] += x(i + hipThreadIdx_x + -1, j + 7, k + 6);
        buf4[54] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf4[52] -= x(i + hipThreadIdx_x, j + 5, k + 6);
        buf6[54] += x(i + hipThreadIdx_x + 1, j + 5, k + 6);
        buf6[53] -= x(i + hipThreadIdx_x + 1, j + 5, k + 6);
        buf6[55] += x(i + hipThreadIdx_x + -1, j + 7, k + 6);
        buf2[53] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf2[61] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf2[45] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf3[53] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf3[61] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf3[45] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf3[37] -= x(i + hipThreadIdx_x, j + 5, k + 6);
        buf5[61] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf5[45] -= x(i + hipThreadIdx_x, j + 5, k + 6);
        buf5[47] += x(i + hipThreadIdx_x + -1, j + 7, k + 6);
        buf5[63] -= x(i + hipThreadIdx_x + -1, j + 7, k + 6);
        buf7[61] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf7[45] -= x(i + hipThreadIdx_x, j + 5, k + 6);
        buf7[46] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf7[62] -= x(i + hipThreadIdx_x, j + 5, k + 6);
      }
      {
        buf0[55] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[55] -= x(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[54] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[53] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf1[55] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf1[54] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf1[53] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf1[52] -= x(i + hipThreadIdx_x, j + 6, k + 6);
        buf4[55] += x(i + hipThreadIdx_x + -1, j + 8, k + 6);
        buf4[55] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf4[53] -= x(i + hipThreadIdx_x, j + 6, k + 6);
        buf6[55] += x(i + hipThreadIdx_x + 1, j + 6, k + 6);
        buf6[54] -= x(i + hipThreadIdx_x + 1, j + 6, k + 6);
        buf2[54] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf2[62] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf2[46] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf3[54] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf3[62] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf3[46] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf3[38] -= x(i + hipThreadIdx_x, j + 6, k + 6);
        buf5[62] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf5[46] -= x(i + hipThreadIdx_x, j + 6, k + 6);
        buf7[62] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf7[46] -= x(i + hipThreadIdx_x, j + 6, k + 6);
        buf7[47] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf7[63] -= x(i + hipThreadIdx_x, j + 6, k + 6);
      }
      {
        buf0[55] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[54] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf1[55] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf1[54] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf1[53] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf2[55] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf2[63] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf2[47] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf3[55] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf3[63] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf3[47] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf3[39] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf6[55] -= x(i + hipThreadIdx_x + 1, j + 7, k + 6);
        buf4[54] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf5[63] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf5[47] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf7[63] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf7[47] -= x(i + hipThreadIdx_x, j + 7, k + 6);
      }
      {
        buf0[55] += x(i + hipThreadIdx_x, j + 8, k + 6);
        buf1[55] += 15.0 * x(i + hipThreadIdx_x, j + 8, k + 6);
        buf1[54] -= x(i + hipThreadIdx_x, j + 8, k + 6);
        buf4[55] -= x(i + hipThreadIdx_x, j + 8, k + 6);
      }
      {
        buf1[55] -= x(i + hipThreadIdx_x, j + 9, k + 6);
      }
      {
        buf4[56] -= x(i + hipThreadIdx_x + -1, j + -1, k + 7);
        buf6[56] -= x(i + hipThreadIdx_x + -1, j + -1, k + 7);
      }
      {
        buf4[57] -= x(i + hipThreadIdx_x + -1, j, k + 7);
        buf6[57] -= x(i + hipThreadIdx_x + -1, j, k + 7);
        buf6[56] += x(i + hipThreadIdx_x + -1, j, k + 7);
        buf5[48] += x(i + hipThreadIdx_x + -1, j, k + 7);
      }
      {
        buf0[56] += 15.0 * x(i + hipThreadIdx_x, j + -1, k + 7);
        buf0[56] -= x(i + hipThreadIdx_x, j + -1, k + 7);
        buf1[56] += x(i + hipThreadIdx_x, j + -1, k + 7);
        buf4[56] += x(i + hipThreadIdx_x + -1, j + 1, k + 7);
        buf4[56] += x(i + hipThreadIdx_x, j + -1, k + 7);
        buf4[58] -= x(i + hipThreadIdx_x + -1, j + 1, k + 7);
        buf6[56] += x(i + hipThreadIdx_x + 1, j + -1, k + 7);
        buf6[58] -= x(i + hipThreadIdx_x + -1, j + 1, k + 7);
        buf6[57] += x(i + hipThreadIdx_x + -1, j + 1, k + 7);
        buf5[49] += x(i + hipThreadIdx_x + -1, j + 1, k + 7);
        buf7[48] += x(i + hipThreadIdx_x, j + -1, k + 7);
      }
      {
        buf0[57] += 15.0 * x(i + hipThreadIdx_x, j, k + 7);
        buf0[57] -= x(i + hipThreadIdx_x, j, k + 7);
        buf0[56] -= 15.0 * x(i + hipThreadIdx_x, j, k + 7);
        buf1[57] += x(i + hipThreadIdx_x, j, k + 7);
        buf1[56] -= 15.0 * x(i + hipThreadIdx_x, j, k + 7);
        buf4[57] += x(i + hipThreadIdx_x + -1, j + 2, k + 7);
        buf4[57] += x(i + hipThreadIdx_x, j, k + 7);
        buf4[59] -= x(i + hipThreadIdx_x + -1, j + 2, k + 7);
        buf6[57] += x(i + hipThreadIdx_x + 1, j, k + 7);
        buf6[56] -= x(i + hipThreadIdx_x + 1, j, k + 7);
        buf6[59] -= x(i + hipThreadIdx_x + -1, j + 2, k + 7);
        buf6[58] += x(i + hipThreadIdx_x + -1, j + 2, k + 7);
        buf2[56] -= 15.0 * x(i + hipThreadIdx_x, j, k + 7);
        buf2[48] += x(i + hipThreadIdx_x, j, k + 7);
        buf3[56] -= 15.0 * x(i + hipThreadIdx_x, j, k + 7);
        buf3[48] += 15.0 * x(i + hipThreadIdx_x, j, k + 7);
        buf3[40] -= x(i + hipThreadIdx_x, j, k + 7);
        buf5[48] -= x(i + hipThreadIdx_x, j, k + 7);
        buf5[50] += x(i + hipThreadIdx_x + -1, j + 2, k + 7);
        buf7[48] -= x(i + hipThreadIdx_x, j, k + 7);
        buf7[49] += x(i + hipThreadIdx_x, j, k + 7);
      }
      {
        buf0[58] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[58] -= x(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[57] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[56] += x(i + hipThreadIdx_x, j + 1, k + 7);
        buf1[58] += x(i + hipThreadIdx_x, j + 1, k + 7);
        buf1[57] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf1[56] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf4[58] += x(i + hipThreadIdx_x + -1, j + 3, k + 7);
        buf4[58] += x(i + hipThreadIdx_x, j + 1, k + 7);
        buf4[56] -= x(i + hipThreadIdx_x, j + 1, k + 7);
        buf4[60] -= x(i + hipThreadIdx_x + -1, j + 3, k + 7);
        buf6[58] += x(i + hipThreadIdx_x + 1, j + 1, k + 7);
        buf6[57] -= x(i + hipThreadIdx_x + 1, j + 1, k + 7);
        buf6[60] -= x(i + hipThreadIdx_x + -1, j + 3, k + 7);
        buf6[59] += x(i + hipThreadIdx_x + -1, j + 3, k + 7);
        buf2[57] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf2[49] += x(i + hipThreadIdx_x, j + 1, k + 7);
        buf3[57] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf3[49] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf3[41] -= x(i + hipThreadIdx_x, j + 1, k + 7);
        buf5[49] -= x(i + hipThreadIdx_x, j + 1, k + 7);
        buf5[51] += x(i + hipThreadIdx_x + -1, j + 3, k + 7);
        buf7[49] -= x(i + hipThreadIdx_x, j + 1, k + 7);
        buf7[50] += x(i + hipThreadIdx_x, j + 1, k + 7);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 3; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[59 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf0[59 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf0[58 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf0[57 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf1[59 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf1[58 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf1[57 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf1[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf4[59 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 7);
            buf4[59 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf4[57 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf4[61 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 7);
            buf6[59 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7);
            buf6[58 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7);
            buf6[61 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 7);
            buf6[60 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 7);
            buf2[58 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf2[50 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf3[58 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf3[50 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf3[42 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf5[50 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf5[52 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 7);
            buf7[50 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf7[51 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[62] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[62] -= x(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[61] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[60] += x(i + hipThreadIdx_x, j + 5, k + 7);
        buf1[62] += x(i + hipThreadIdx_x, j + 5, k + 7);
        buf1[61] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf1[60] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf1[59] -= x(i + hipThreadIdx_x, j + 5, k + 7);
        buf4[62] += x(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf4[62] += x(i + hipThreadIdx_x, j + 5, k + 7);
        buf4[60] -= x(i + hipThreadIdx_x, j + 5, k + 7);
        buf6[62] += x(i + hipThreadIdx_x + 1, j + 5, k + 7);
        buf6[61] -= x(i + hipThreadIdx_x + 1, j + 5, k + 7);
        buf6[63] += x(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf2[61] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf2[53] += x(i + hipThreadIdx_x, j + 5, k + 7);
        buf3[61] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf3[53] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf3[45] -= x(i + hipThreadIdx_x, j + 5, k + 7);
        buf5[53] -= x(i + hipThreadIdx_x, j + 5, k + 7);
        buf5[55] += x(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf7[53] -= x(i + hipThreadIdx_x, j + 5, k + 7);
        buf7[54] += x(i + hipThreadIdx_x, j + 5, k + 7);
      }
      {
        buf0[63] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[63] -= x(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[62] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[61] += x(i + hipThreadIdx_x, j + 6, k + 7);
        buf1[63] += x(i + hipThreadIdx_x, j + 6, k + 7);
        buf1[62] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf1[61] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf1[60] -= x(i + hipThreadIdx_x, j + 6, k + 7);
        buf4[63] += x(i + hipThreadIdx_x + -1, j + 8, k + 7);
        buf4[63] += x(i + hipThreadIdx_x, j + 6, k + 7);
        buf4[61] -= x(i + hipThreadIdx_x, j + 6, k + 7);
        buf6[63] += x(i + hipThreadIdx_x + 1, j + 6, k + 7);
        buf6[62] -= x(i + hipThreadIdx_x + 1, j + 6, k + 7);
        buf2[62] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf2[54] += x(i + hipThreadIdx_x, j + 6, k + 7);
        buf3[62] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf3[54] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf3[46] -= x(i + hipThreadIdx_x, j + 6, k + 7);
        buf5[54] -= x(i + hipThreadIdx_x, j + 6, k + 7);
        buf7[54] -= x(i + hipThreadIdx_x, j + 6, k + 7);
        buf7[55] += x(i + hipThreadIdx_x, j + 6, k + 7);
      }
      {
        buf0[63] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[62] += x(i + hipThreadIdx_x, j + 7, k + 7);
        buf1[63] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 7);
        buf1[62] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 7);
        buf1[61] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf2[63] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 7);
        buf2[55] += x(i + hipThreadIdx_x, j + 7, k + 7);
        buf3[63] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 7);
        buf3[55] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 7);
        buf3[47] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf6[63] -= x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf4[62] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf5[55] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf7[55] -= x(i + hipThreadIdx_x, j + 7, k + 7);
      }
      {
        buf0[63] += x(i + hipThreadIdx_x, j + 8, k + 7);
        buf1[63] += 15.0 * x(i + hipThreadIdx_x, j + 8, k + 7);
        buf1[62] -= x(i + hipThreadIdx_x, j + 8, k + 7);
        buf4[63] -= x(i + hipThreadIdx_x, j + 8, k + 7);
      }
      {
        buf1[63] -= x(i + hipThreadIdx_x, j + 9, k + 7);
      }
      {
        buf5[56] += x(i + hipThreadIdx_x + -1, j, k + 8);
      }
      {
        buf5[57] += x(i + hipThreadIdx_x + -1, j + 1, k + 8);
        buf7[56] += x(i + hipThreadIdx_x, j + -1, k + 8);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf2[56 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf3[56 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf3[48 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf5[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf5[58 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 8);
            buf7[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf7[57 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf2[62] += x(i + hipThreadIdx_x, j + 6, k + 8);
        buf3[62] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 8);
        buf3[54] -= x(i + hipThreadIdx_x, j + 6, k + 8);
        buf5[62] -= x(i + hipThreadIdx_x, j + 6, k + 8);
        buf7[62] -= x(i + hipThreadIdx_x, j + 6, k + 8);
        buf7[63] += x(i + hipThreadIdx_x, j + 6, k + 8);
      }
      {
        buf2[63] += x(i + hipThreadIdx_x, j + 7, k + 8);
        buf3[63] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 8);
        buf3[55] -= x(i + hipThreadIdx_x, j + 7, k + 8);
        buf5[63] -= x(i + hipThreadIdx_x, j + 7, k + 8);
        buf7[63] -= x(i + hipThreadIdx_x, j + 7, k + 8);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 7; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf3[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + 9);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf3[63] -= x(i + hipThreadIdx_x, j + 7, k + 9);
      }
    }
  }
  bElem buf8[64];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf8[0 + rel] = 0;
              }
              _cg_rel1 += 1;
            }
          }
          _cg_rel2 += 8;
        }
      }
    }
    {
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf8[0 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + -1);
            buf8[0 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + -1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 7; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf8[8 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2);
                buf8[8 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + _cg_idx2);
                buf8[0 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2);
                buf8[0 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + _cg_idx2);
              }
              _cg_rel1 += 1;
            }
          }
          _cg_rel2 += 8;
        }
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf8[56 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + 7);
            buf8[56 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + 7);
          }
          _cg_rel1 += 1;
        }
      }
    }
  }
  bElem buf9[64];
  bElem buf10[64];
  bElem buf11[64];
  bElem buf12[64];
  bElem buf13[64];
  bElem buf14[64];
  bElem buf15[64];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf9[0 + rel] = 0;
                buf10[0 + rel] = 0;
                buf11[0 + rel] = 0;
                buf12[0 + rel] = 0;
                buf13[0 + rel] = 0;
                buf14[0 + rel] = 0;
                buf15[0 + rel] = 0;
              }
              _cg_rel1 += 1;
            }
          }
          _cg_rel2 += 8;
        }
      }
    }
    {
      {
        buf11[0] -= x(i + hipThreadIdx_x + 1, j, k + -1);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 7; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf11[1 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + -1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf9[0] -= x(i + hipThreadIdx_x, j + -1, k + -1);
        buf10[0] -= x(i + hipThreadIdx_x + 1, j + -1, k);
      }
      {
        buf9[1] -= x(i + hipThreadIdx_x, j, k + -1);
        buf10[1] -= x(i + hipThreadIdx_x + 1, j, k);
        buf11[8] -= x(i + hipThreadIdx_x + 1, j, k);
        buf11[0] += x(i + hipThreadIdx_x, j, k + -1);
        buf12[0] -= x(i + hipThreadIdx_x + 1, j, k);
        buf12[0] += x(i + hipThreadIdx_x + -1, j, k);
        buf13[0] += x(i + hipThreadIdx_x, j, k + -1);
        buf14[0] -= x(i + hipThreadIdx_x + 1, j, k);
        buf14[0] += x(i + hipThreadIdx_x + -1, j, k);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf9[0 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf9[2 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf10[0 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf10[2 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf12[0 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf12[0 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
            buf12[1 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf12[1 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
            buf13[0 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf13[1 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf11[9 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf11[1 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf14[1 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf14[1 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf9[6] += x(i + hipThreadIdx_x, j + 7, k + -1);
        buf10[6] += x(i + hipThreadIdx_x + 1, j + 7, k);
        buf12[6] += x(i + hipThreadIdx_x + 1, j + 7, k);
        buf12[6] -= x(i + hipThreadIdx_x + -1, j + 7, k);
        buf12[7] -= x(i + hipThreadIdx_x + 1, j + 7, k);
        buf12[7] += x(i + hipThreadIdx_x + -1, j + 7, k);
        buf13[6] -= x(i + hipThreadIdx_x, j + 7, k + -1);
        buf13[7] += x(i + hipThreadIdx_x, j + 7, k + -1);
        buf11[15] -= x(i + hipThreadIdx_x + 1, j + 7, k);
        buf11[7] += x(i + hipThreadIdx_x, j + 7, k + -1);
        buf14[7] -= x(i + hipThreadIdx_x + 1, j + 7, k);
        buf14[7] += x(i + hipThreadIdx_x + -1, j + 7, k);
      }
      {
        buf9[7] += x(i + hipThreadIdx_x, j + 8, k + -1);
        buf10[7] += x(i + hipThreadIdx_x + 1, j + 8, k);
        buf12[7] += x(i + hipThreadIdx_x + 1, j + 8, k);
        buf12[7] -= x(i + hipThreadIdx_x + -1, j + 8, k);
        buf13[7] -= x(i + hipThreadIdx_x, j + 8, k + -1);
      }
      {
        buf9[8] -= x(i + hipThreadIdx_x, j + -1, k);
        buf9[0] += x(i + hipThreadIdx_x, j + -1, k);
        buf10[8] -= x(i + hipThreadIdx_x + 1, j + -1, k + 1);
        buf10[0] += x(i + hipThreadIdx_x, j + -1, k);
        buf15[0] += x(i + hipThreadIdx_x, j + -1, k);
      }
      {
        buf9[9] -= x(i + hipThreadIdx_x, j, k);
        buf9[1] += x(i + hipThreadIdx_x, j, k);
        buf10[9] -= x(i + hipThreadIdx_x + 1, j, k + 1);
        buf10[1] += x(i + hipThreadIdx_x, j, k);
        buf15[1] += x(i + hipThreadIdx_x, j, k);
        buf11[0] += x(i + hipThreadIdx_x + 1, j, k + 1);
        buf11[16] -= x(i + hipThreadIdx_x + 1, j, k + 1);
        buf11[8] += x(i + hipThreadIdx_x, j, k);
        buf14[0] += x(i + hipThreadIdx_x + 1, j, k + 1);
        buf14[0] -= x(i + hipThreadIdx_x + -1, j, k + 1);
        buf14[8] -= x(i + hipThreadIdx_x + 1, j, k + 1);
        buf14[8] += x(i + hipThreadIdx_x + -1, j, k + 1);
        buf12[8] -= x(i + hipThreadIdx_x + 1, j, k + 1);
        buf12[8] += x(i + hipThreadIdx_x + -1, j, k + 1);
        buf13[8] += x(i + hipThreadIdx_x, j, k);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf9[8 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf9[0 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf9[10 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf9[2 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf10[8 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf10[0 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf10[10 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf10[2 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf12[8 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf12[8 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 1);
            buf12[9 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf12[9 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 1);
            buf13[8 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf13[9 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf15[0 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf15[2 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf11[1 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf11[17 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf11[9 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf14[1 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf14[1 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 1);
            buf14[9 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf14[9 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf9[14] += x(i + hipThreadIdx_x, j + 7, k);
        buf9[6] -= x(i + hipThreadIdx_x, j + 7, k);
        buf10[14] += x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf10[6] -= x(i + hipThreadIdx_x, j + 7, k);
        buf12[14] += x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf12[14] -= x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf12[15] -= x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf12[15] += x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf13[14] -= x(i + hipThreadIdx_x, j + 7, k);
        buf13[15] += x(i + hipThreadIdx_x, j + 7, k);
        buf15[6] -= x(i + hipThreadIdx_x, j + 7, k);
        buf11[7] += x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf11[23] -= x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf11[15] += x(i + hipThreadIdx_x, j + 7, k);
        buf14[7] += x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf14[7] -= x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf14[15] -= x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf14[15] += x(i + hipThreadIdx_x + -1, j + 7, k + 1);
      }
      {
        buf9[15] += x(i + hipThreadIdx_x, j + 8, k);
        buf9[7] -= x(i + hipThreadIdx_x, j + 8, k);
        buf10[15] += x(i + hipThreadIdx_x + 1, j + 8, k + 1);
        buf10[7] -= x(i + hipThreadIdx_x, j + 8, k);
        buf12[15] += x(i + hipThreadIdx_x + 1, j + 8, k + 1);
        buf12[15] -= x(i + hipThreadIdx_x + -1, j + 8, k + 1);
        buf13[15] -= x(i + hipThreadIdx_x, j + 8, k);
        buf15[7] -= x(i + hipThreadIdx_x, j + 8, k);
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 5; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            buf9[16 + rel] -= x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
            buf9[8 + rel] += x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
            buf10[16 + rel] -= x(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 2);
            buf10[8 + rel] += x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
            buf15[8 + rel] += x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
            buf15[0 + rel] -= x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
          }
          {
            buf9[17 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf9[9 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf10[17 + rel] -= x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf10[9 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf15[9 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf15[1 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf11[8 + rel] += x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf11[0 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf11[24 + rel] -= x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf11[16 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf14[8 + rel] += x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf14[8 + rel] -= x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
            buf14[16 + rel] -= x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf14[16 + rel] += x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
            buf13[0 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf13[16 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf12[16 + rel] -= x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf12[16 + rel] += x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
          }
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf9[16 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf9[8 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf9[18 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf9[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf10[16 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf10[8 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf10[18 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf10[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf12[16 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf12[16 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf12[17 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf12[17 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf13[16 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf13[1 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf13[17 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf13[0 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf15[8 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf15[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf15[0 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf15[2 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf11[9 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf11[1 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf11[25 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf11[17 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf14[9 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf14[9 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf14[17 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf14[17 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
              }
              _cg_rel1 += 1;
            }
          }
          {
            buf9[22 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf9[14 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf10[22 + rel] += x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf10[14 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf12[22 + rel] += x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf12[22 + rel] -= x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf12[23 + rel] -= x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf12[23 + rel] += x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf13[22 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf13[7 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf13[23 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf13[6 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf15[14 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf15[6 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf11[15 + rel] += x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf11[7 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf11[31 + rel] -= x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf11[23 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf14[15 + rel] += x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf14[15 + rel] -= x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf14[23 + rel] -= x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf14[23 + rel] += x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
          }
          {
            buf9[23 + rel] += x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf9[15 + rel] -= x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf10[23 + rel] += x(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 2);
            buf10[15 + rel] -= x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf12[23 + rel] += x(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 2);
            buf12[23 + rel] -= x(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 2);
            buf13[23 + rel] -= x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf13[7 + rel] += x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf15[15 + rel] -= x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf15[7 + rel] += x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
          }
          _cg_rel2 += 8;
        }
      }
      {
        buf9[56] -= x(i + hipThreadIdx_x, j + -1, k + 6);
        buf9[48] += x(i + hipThreadIdx_x, j + -1, k + 6);
        buf10[56] -= x(i + hipThreadIdx_x + 1, j + -1, k + 7);
        buf10[48] += x(i + hipThreadIdx_x, j + -1, k + 6);
        buf15[48] += x(i + hipThreadIdx_x, j + -1, k + 6);
        buf15[40] -= x(i + hipThreadIdx_x, j + -1, k + 6);
      }
      {
        buf9[57] -= x(i + hipThreadIdx_x, j, k + 6);
        buf9[49] += x(i + hipThreadIdx_x, j, k + 6);
        buf10[57] -= x(i + hipThreadIdx_x + 1, j, k + 7);
        buf10[49] += x(i + hipThreadIdx_x, j, k + 6);
        buf15[49] += x(i + hipThreadIdx_x, j, k + 6);
        buf15[41] -= x(i + hipThreadIdx_x, j, k + 6);
        buf11[48] += x(i + hipThreadIdx_x + 1, j, k + 7);
        buf11[40] -= x(i + hipThreadIdx_x, j, k + 6);
        buf11[56] += x(i + hipThreadIdx_x, j, k + 6);
        buf14[48] += x(i + hipThreadIdx_x + 1, j, k + 7);
        buf14[48] -= x(i + hipThreadIdx_x + -1, j, k + 7);
        buf14[56] -= x(i + hipThreadIdx_x + 1, j, k + 7);
        buf14[56] += x(i + hipThreadIdx_x + -1, j, k + 7);
        buf13[40] -= x(i + hipThreadIdx_x, j, k + 6);
        buf13[56] += x(i + hipThreadIdx_x, j, k + 6);
        buf12[56] -= x(i + hipThreadIdx_x + 1, j, k + 7);
        buf12[56] += x(i + hipThreadIdx_x + -1, j, k + 7);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf9[56 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf9[48 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf9[58 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf9[50 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf10[56 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf10[48 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf10[58 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf10[50 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf12[56 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf12[56 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf12[57 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf12[57 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf13[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf13[41 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf13[57 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf13[40 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf15[48 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf15[50 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf15[40 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf15[42 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf11[49 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf11[41 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf11[57 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf14[49 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf14[49 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf14[57 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf14[57 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf9[62] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf9[54] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf10[62] += x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf10[54] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf12[62] += x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf12[62] -= x(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf12[63] -= x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf12[63] += x(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf13[62] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf13[47] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf13[63] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf13[46] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf15[54] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf15[46] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf11[55] += x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf11[47] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf11[63] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf14[55] += x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf14[55] -= x(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf14[63] -= x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf14[63] += x(i + hipThreadIdx_x + -1, j + 7, k + 7);
      }
      {
        buf9[63] += x(i + hipThreadIdx_x, j + 8, k + 6);
        buf9[55] -= x(i + hipThreadIdx_x, j + 8, k + 6);
        buf10[63] += x(i + hipThreadIdx_x + 1, j + 8, k + 7);
        buf10[55] -= x(i + hipThreadIdx_x, j + 8, k + 6);
        buf12[63] += x(i + hipThreadIdx_x + 1, j + 8, k + 7);
        buf12[63] -= x(i + hipThreadIdx_x + -1, j + 8, k + 7);
        buf13[63] -= x(i + hipThreadIdx_x, j + 8, k + 6);
        buf13[47] += x(i + hipThreadIdx_x, j + 8, k + 6);
        buf15[55] -= x(i + hipThreadIdx_x, j + 8, k + 6);
        buf15[47] += x(i + hipThreadIdx_x, j + 8, k + 6);
      }
      {
        buf9[56] += x(i + hipThreadIdx_x, j + -1, k + 7);
        buf10[56] += x(i + hipThreadIdx_x, j + -1, k + 7);
        buf15[56] += x(i + hipThreadIdx_x, j + -1, k + 7);
        buf15[48] -= x(i + hipThreadIdx_x, j + -1, k + 7);
      }
      {
        buf9[57] += x(i + hipThreadIdx_x, j, k + 7);
        buf10[57] += x(i + hipThreadIdx_x, j, k + 7);
        buf15[57] += x(i + hipThreadIdx_x, j, k + 7);
        buf15[49] -= x(i + hipThreadIdx_x, j, k + 7);
        buf11[56] += x(i + hipThreadIdx_x + 1, j, k + 8);
        buf11[48] -= x(i + hipThreadIdx_x, j, k + 7);
        buf14[56] += x(i + hipThreadIdx_x + 1, j, k + 8);
        buf14[56] -= x(i + hipThreadIdx_x + -1, j, k + 8);
        buf13[48] -= x(i + hipThreadIdx_x, j, k + 7);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf9[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf9[58 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf10[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf10[58 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf15[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf15[58 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf15[48 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf15[50 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf11[57 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 8);
            buf11[49 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf14[57 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 8);
            buf14[57 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 8);
            buf13[49 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf13[48 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf9[62] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf10[62] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf15[62] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf15[54] += x(i + hipThreadIdx_x, j + 7, k + 7);
        buf11[63] += x(i + hipThreadIdx_x + 1, j + 7, k + 8);
        buf11[55] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf14[63] += x(i + hipThreadIdx_x + 1, j + 7, k + 8);
        buf14[63] -= x(i + hipThreadIdx_x + -1, j + 7, k + 8);
        buf13[55] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf13[54] += x(i + hipThreadIdx_x, j + 7, k + 7);
      }
      {
        buf9[63] -= x(i + hipThreadIdx_x, j + 8, k + 7);
        buf10[63] -= x(i + hipThreadIdx_x, j + 8, k + 7);
        buf15[63] -= x(i + hipThreadIdx_x, j + 8, k + 7);
        buf15[55] += x(i + hipThreadIdx_x, j + 8, k + 7);
        buf13[55] += x(i + hipThreadIdx_x, j + 8, k + 7);
      }
      {
        buf15[56] -= x(i + hipThreadIdx_x, j + -1, k + 8);
      }
      {
        buf11[56] -= x(i + hipThreadIdx_x, j, k + 8);
        buf13[56] -= x(i + hipThreadIdx_x, j, k + 8);
        buf15[57] -= x(i + hipThreadIdx_x, j, k + 8);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf11[57 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
            buf13[57 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
            buf13[56 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
            buf15[56 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
            buf15[58 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf11[63] -= x(i + hipThreadIdx_x, j + 7, k + 8);
        buf13[63] -= x(i + hipThreadIdx_x, j + 7, k + 8);
        buf13[62] += x(i + hipThreadIdx_x, j + 7, k + 8);
        buf15[62] += x(i + hipThreadIdx_x, j + 7, k + 8);
      }
      {
        buf13[63] += x(i + hipThreadIdx_x, j + 8, k + 8);
        buf15[63] += x(i + hipThreadIdx_x, j + 8, k + 8);
      }
    }
  }
  bElem buf16[64];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf16[0 + rel] = 0;
              }
              _cg_rel1 += 1;
            }
          }
          _cg_rel2 += 8;
        }
      }
    }
    {
      {
        buf16[0] += (beta_i(i + hipThreadIdx_x, j + 1, k) - beta_i(i + hipThreadIdx_x, j + -1, k)) * buf4[0];
        buf16[0] += (beta_j(i + hipThreadIdx_x + 1, j, k) - beta_j(i + hipThreadIdx_x + -1, j, k)) * buf6[0];
        buf16[0] += (beta_j(i + hipThreadIdx_x, j, k + 1) - beta_j(i + hipThreadIdx_x, j, k + -1)) * buf7[0];
        buf16[0] += (beta_k(i + hipThreadIdx_x + 1, j, k) - beta_k(i + hipThreadIdx_x + -1, j, k)) * buf8[0];
        buf16[0] += (beta_k(i + hipThreadIdx_x, j + 1, k) - beta_k(i + hipThreadIdx_x, j + -1, k)) * buf9[0];
        buf16[0] += (beta_i(i + hipThreadIdx_x + 1, j + 1, k) - beta_i(i + hipThreadIdx_x + 1, j + -1, k)) * buf10[0];
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 7; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf16[1 + rel] += (beta_i(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) - beta_i(i + hipThreadIdx_x, j + _cg_idx1, k)) * buf4[1 + rel];
            buf16[1 + rel] += (beta_j(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k) - beta_j(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k)) * buf6[1 + rel];
            buf16[1 + rel] += (beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 1) - beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1)) * buf7[1 + rel];
            buf16[1 + rel] += (beta_k(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k) - beta_k(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k)) * buf8[1 + rel];
            buf16[1 + rel] += (beta_k(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) - beta_k(i + hipThreadIdx_x, j + _cg_idx1, k)) * buf9[1 + rel];
            buf16[1 + rel] += (beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k) - beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k)) * buf10[1 + rel];
            buf16[0 + rel] += (beta_j(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k) - beta_j(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k)) * buf12[0 + rel];
            buf16[0 + rel] += (beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 1) - beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1)) * buf13[0 + rel];
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf16[7] += (beta_j(i + hipThreadIdx_x + 1, j + 8, k) - beta_j(i + hipThreadIdx_x + -1, j + 8, k)) * buf12[7];
        buf16[7] += (beta_j(i + hipThreadIdx_x, j + 8, k + 1) - beta_j(i + hipThreadIdx_x, j + 8, k + -1)) * buf13[7];
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 7; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            buf16[8 + rel] += (beta_i(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1)) * buf4[8 + rel];
            buf16[8 + rel] += (beta_j(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1) - beta_j(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1)) * buf6[8 + rel];
            buf16[8 + rel] += (beta_j(i + hipThreadIdx_x, j, k + _cg_idx2 + 2) - beta_j(i + hipThreadIdx_x, j, k + _cg_idx2)) * buf7[8 + rel];
            buf16[8 + rel] += (beta_k(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1)) * buf8[8 + rel];
            buf16[8 + rel] += (beta_k(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1)) * buf9[8 + rel];
            buf16[8 + rel] += (beta_i(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 1)) * buf10[8 + rel];
            buf16[0 + rel] += (beta_k(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1)) * buf14[0 + rel];
            buf16[0 + rel] += (beta_k(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1)) * buf15[0 + rel];
          }
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 7; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf16[9 + rel] += (beta_i(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1)) * buf4[9 + rel];
                buf16[9 + rel] += (beta_j(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) - beta_j(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1)) * buf6[9 + rel];
                buf16[9 + rel] += (beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 2) - beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2)) * buf7[9 + rel];
                buf16[9 + rel] += (beta_k(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1)) * buf8[9 + rel];
                buf16[9 + rel] += (beta_k(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1)) * buf9[9 + rel];
                buf16[9 + rel] += (beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2 + 1)) * buf10[9 + rel];
                buf16[0 + rel] += (beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + -1)) * buf5[0 + rel];
                buf16[0 + rel] += (beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2 + -1)) * buf11[0 + rel];
                buf16[8 + rel] += (beta_j(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) - beta_j(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1)) * buf12[8 + rel];
                buf16[8 + rel] += (beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 2) - beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2)) * buf13[8 + rel];
                buf16[1 + rel] += (beta_k(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1)) * buf14[1 + rel];
                buf16[1 + rel] += (beta_k(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1)) * buf15[1 + rel];
              }
              _cg_rel1 += 1;
            }
          }
          {
            buf16[7 + rel] += (beta_i(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + -1)) * buf5[7 + rel];
            buf16[7 + rel] += (beta_i(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + -1)) * buf11[7 + rel];
            buf16[15 + rel] += (beta_j(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 1) - beta_j(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 1)) * buf12[15 + rel];
            buf16[15 + rel] += (beta_j(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2) - beta_j(i + hipThreadIdx_x, j + 8, k + _cg_idx2)) * buf13[15 + rel];
          }
          _cg_rel2 += 8;
        }
      }
      {
        buf16[56] += (beta_k(i + hipThreadIdx_x + 1, j, k + 8) - beta_k(i + hipThreadIdx_x + -1, j, k + 8)) * buf14[56];
        buf16[56] += (beta_k(i + hipThreadIdx_x, j + 1, k + 8) - beta_k(i + hipThreadIdx_x, j + -1, k + 8)) * buf15[56];
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 7; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf16[56 + rel] += (beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + 8) - beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + 6)) * buf5[56 + rel];
            buf16[56 + rel] += (beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + 8) - beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + 6)) * buf11[56 + rel];
            buf16[57 + rel] += (beta_k(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 8) - beta_k(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 8)) * buf14[57 + rel];
            buf16[57 + rel] += (beta_k(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 8) - beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + 8)) * buf15[57 + rel];
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf16[63] += (beta_i(i + hipThreadIdx_x, j + 7, k + 8) - beta_i(i + hipThreadIdx_x, j + 7, k + 6)) * buf5[63];
        buf16[63] += (beta_i(i + hipThreadIdx_x + 1, j + 7, k + 8) - beta_i(i + hipThreadIdx_x + 1, j + 7, k + 6)) * buf11[63];
      }
    }
  }
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf15[0 + rel] = c[0] * alpha(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) - c[1] * c[2] * (0.0833 * (beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * (15.0 * x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + _cg_idx2) - 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) - x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + _cg_idx2) + x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2)) + beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2) * (15.0 * x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2) - 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) - x(i + hipThreadIdx_x + 2, j + _cg_idx1, k + _cg_idx2) + x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + _cg_idx2)) + beta_j(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * buf0[0 + rel] + beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2) * buf1[0 + rel] + beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * buf2[0 + rel] + beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1) * buf3[0 + rel]) + 0.020825 * buf16[0 + rel]);
              }
              _cg_rel1 += 1;
            }
          }
          _cg_rel2 += 8;
        }
      }
    }
    {
      long rel = 0;
      for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
      {
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          for (long _cg_idx0 = hipThreadIdx_x; _cg_idx0 < 64; _cg_idx0 += 64, ++rel)
          {
            out(i + _cg_idx0, j + _cg_idx1, k + _cg_idx2) = buf15[rel];
          }
        }
      }
    }
  }
}
# 80 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/helmholtz4/intermediate_gen/helmholtz4.cu" 2

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
# 1 "VSBrick-helmholtz42.py-HIP-8x8x8-8x8" 1
{
  auto *binfo = out.bInfo;
  long neighbor0 = binfo->adj[b][0];
  long neighbor1 = binfo->adj[b][1];
  long neighbor2 = binfo->adj[b][2];
  long neighbor3 = binfo->adj[b][3];
  long neighbor4 = binfo->adj[b][4];
  long neighbor5 = binfo->adj[b][5];
  long neighbor6 = binfo->adj[b][6];
  long neighbor7 = binfo->adj[b][7];
  long neighbor8 = binfo->adj[b][8];
  long neighbor9 = binfo->adj[b][9];
  long neighbor10 = binfo->adj[b][10];
  long neighbor11 = binfo->adj[b][11];
  long neighbor12 = binfo->adj[b][12];
  long neighbor13 = b;
  long neighbor14 = binfo->adj[b][14];
  long neighbor15 = binfo->adj[b][15];
  long neighbor16 = binfo->adj[b][16];
  long neighbor17 = binfo->adj[b][17];
  long neighbor18 = binfo->adj[b][18];
  long neighbor19 = binfo->adj[b][19];
  long neighbor20 = binfo->adj[b][20];
  long neighbor21 = binfo->adj[b][21];
  long neighbor22 = binfo->adj[b][22];
  long neighbor23 = binfo->adj[b][23];
  long neighbor24 = binfo->adj[b][24];
  long neighbor25 = binfo->adj[b][25];
  long neighbor26 = binfo->adj[b][26];
  bElem buf0[8];
  bElem buf1[8];
  bElem buf2[8];
  bElem buf3[8];
  bElem buf4[8];
  bElem buf5[8];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 0]
            buf0[0 + rel] = 0;
            buf1[0 + rel] = 0;
            buf2[0 + rel] = 0;
            buf3[0 + rel] = 0;
            buf4[0 + rel] = 0;
            buf5[0 + rel] = 0;
          }
          _cg_rel2 += 1;
        }
      }
    }
    {
      bElem _cg_x_100_vecbuf;
      bElem _cg_x000_vecbuf;
      {
        // New offset [1, 0, -2]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor4 * x.step + 384 + hipThreadIdx_x];
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf4[0] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 0, -1]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf4[0] += 15.0 * _cg_x_100_reg;
        buf4[1] -= _cg_x_100_reg;
        buf5[0] += _cg_x_100_reg;
      }
      {
        // New offset [1, -1, 0]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[0] += 15.0 * _cg_x_100_reg;
        buf2[0] -= _cg_x_100_reg;
        buf3[0] += _cg_x_100_reg;
      }
      {
        // New offset [0, 0, 0]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[0] += 15.0 * _cg_x_100_reg;
        buf0[0] -= _cg_x_100_reg;
        buf1[0] += _cg_x_100_reg;
      }
      {
        // New offset [1, 0, 0]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = _cg_x000_vecbuf;
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf0[0] -= 15.0 * _cg_x_100_reg;
        buf1[0] -= 15.0 * _cg_x_100_reg;
        buf2[0] -= 15.0 * _cg_x_100_reg;
        buf3[0] -= 15.0 * _cg_x_100_reg;
        buf4[0] -= 15.0 * _cg_x_100_reg;
        buf4[1] += 15.0 * _cg_x_100_reg;
        buf4[2] -= _cg_x_100_reg;
        buf5[0] -= 15.0 * _cg_x_100_reg;
        buf5[1] += _cg_x_100_reg;
      }
      {
        // New offset [2, 0, 0]
        bElem _cg_x_100_reg;
        {
          _cg_x000_vecbuf = x.dat[neighbor14 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[0] += _cg_x_100_reg;
        buf1[0] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [3, 0, 0]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf1[0] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 1, 0]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[0] += _cg_x_100_reg;
        buf3[0] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [1, 2, 0]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf3[0] -= _cg_x_100_reg;
      }
      {
        // New offset [1, -1, 1]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[1] += 15.0 * _cg_x_100_reg;
        buf2[1] -= _cg_x_100_reg;
        buf3[1] += _cg_x_100_reg;
      }
      {
        // New offset [0, 0, 1]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + 64 + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor13 * x.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[1] += 15.0 * _cg_x_100_reg;
        buf0[1] -= _cg_x_100_reg;
        buf1[1] += _cg_x_100_reg;
      }
      {
        // New offset [1, 0, 1]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = _cg_x000_vecbuf;
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf0[1] -= 15.0 * _cg_x_100_reg;
        buf1[1] -= 15.0 * _cg_x_100_reg;
        buf2[1] -= 15.0 * _cg_x_100_reg;
        buf3[1] -= 15.0 * _cg_x_100_reg;
        buf4[1] -= 15.0 * _cg_x_100_reg;
        buf4[2] += 15.0 * _cg_x_100_reg;
        buf4[3] -= _cg_x_100_reg;
        buf4[0] += _cg_x_100_reg;
        buf5[1] -= 15.0 * _cg_x_100_reg;
        buf5[2] += _cg_x_100_reg;
        buf5[0] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [2, 0, 1]
        bElem _cg_x_100_reg;
        {
          _cg_x000_vecbuf = x.dat[neighbor14 * x.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[1] += _cg_x_100_reg;
        buf1[1] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [3, 0, 1]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf1[1] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 1, 1]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[1] += _cg_x_100_reg;
        buf3[1] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [1, 2, 1]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf3[1] -= _cg_x_100_reg;
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 4; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [1, -1, 2]
            bElem _cg_x_100_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor13 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor10 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_100_vecbuf
              dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              _cg_x_100_reg = _cg_x_100_vecbuf;
            }
            buf2[2 + rel] += 15.0 * _cg_x_100_reg;
            buf2[2 + rel] -= _cg_x_100_reg;
            buf3[2 + rel] += _cg_x_100_reg;
          }
          {
            // New offset [0, 0, 2]
            bElem _cg_x_100_reg;
            {
              _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_x000_vecbuf = x.dat[neighbor13 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_x_100_reg = _cg_vectmp0;
            }
            buf0[2 + rel] += 15.0 * _cg_x_100_reg;
            buf0[2 + rel] -= _cg_x_100_reg;
            buf1[2 + rel] += _cg_x_100_reg;
          }
          {
            // New offset [1, 0, 2]
            bElem _cg_x_100_reg;
            {
              _cg_x_100_vecbuf = _cg_x000_vecbuf;
              _cg_x_100_reg = _cg_x_100_vecbuf;
            }
            buf0[2 + rel] -= 15.0 * _cg_x_100_reg;
            buf1[2 + rel] -= 15.0 * _cg_x_100_reg;
            buf2[2 + rel] -= 15.0 * _cg_x_100_reg;
            buf3[2 + rel] -= 15.0 * _cg_x_100_reg;
            buf4[2 + rel] -= 15.0 * _cg_x_100_reg;
            buf4[3 + rel] += 15.0 * _cg_x_100_reg;
            buf4[4 + rel] -= _cg_x_100_reg;
            buf4[1 + rel] += _cg_x_100_reg;
            buf5[2 + rel] -= 15.0 * _cg_x_100_reg;
            buf5[3 + rel] += _cg_x_100_reg;
            buf5[1 + rel] += 15.0 * _cg_x_100_reg;
            buf5[0 + rel] -= _cg_x_100_reg;
          }
          {
            // New offset [2, 0, 2]
            bElem _cg_x_100_reg;
            {
              _cg_x000_vecbuf = x.dat[neighbor14 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_x_100_reg = _cg_vectmp0;
            }
            buf0[2 + rel] += _cg_x_100_reg;
            buf1[2 + rel] += 15.0 * _cg_x_100_reg;
          }
          {
            // New offset [3, 0, 2]
            bElem _cg_x_100_reg;
            {
              bElem _cg_vectmp0;
              // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 2 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 6, 8, hipThreadIdx_x & 7);
              _cg_x_100_reg = _cg_vectmp0;
            }
            buf1[2 + rel] -= _cg_x_100_reg;
          }
          {
            // New offset [1, 1, 2]
            bElem _cg_x_100_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor16 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor13 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_100_vecbuf
              dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              _cg_x_100_reg = _cg_x_100_vecbuf;
            }
            buf2[2 + rel] += _cg_x_100_reg;
            buf3[2 + rel] += 15.0 * _cg_x_100_reg;
          }
          {
            // New offset [1, 2, 2]
            bElem _cg_x_100_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor16 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor13 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_x_100_vecbuf
              dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
              _cg_x_100_reg = _cg_x_100_vecbuf;
            }
            buf3[2 + rel] -= _cg_x_100_reg;
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [1, -1, 6]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[6] += 15.0 * _cg_x_100_reg;
        buf2[6] -= _cg_x_100_reg;
        buf3[6] += _cg_x_100_reg;
      }
      {
        // New offset [0, 0, 6]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + 384 + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor13 * x.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[6] += 15.0 * _cg_x_100_reg;
        buf0[6] -= _cg_x_100_reg;
        buf1[6] += _cg_x_100_reg;
      }
      {
        // New offset [1, 0, 6]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = _cg_x000_vecbuf;
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf0[6] -= 15.0 * _cg_x_100_reg;
        buf1[6] -= 15.0 * _cg_x_100_reg;
        buf2[6] -= 15.0 * _cg_x_100_reg;
        buf3[6] -= 15.0 * _cg_x_100_reg;
        buf4[6] -= 15.0 * _cg_x_100_reg;
        buf4[7] += 15.0 * _cg_x_100_reg;
        buf4[5] += _cg_x_100_reg;
        buf5[6] -= 15.0 * _cg_x_100_reg;
        buf5[7] += _cg_x_100_reg;
        buf5[5] += 15.0 * _cg_x_100_reg;
        buf5[4] -= _cg_x_100_reg;
      }
      {
        // New offset [2, 0, 6]
        bElem _cg_x_100_reg;
        {
          _cg_x000_vecbuf = x.dat[neighbor14 * x.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[6] += _cg_x_100_reg;
        buf1[6] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [3, 0, 6]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf1[6] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 1, 6]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[6] += _cg_x_100_reg;
        buf3[6] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [1, 2, 6]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf3[6] -= _cg_x_100_reg;
      }
      {
        // New offset [1, -1, 7]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[7] += 15.0 * _cg_x_100_reg;
        buf2[7] -= _cg_x_100_reg;
        buf3[7] += _cg_x_100_reg;
      }
      {
        // New offset [0, 0, 7]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + 448 + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[7] += 15.0 * _cg_x_100_reg;
        buf0[7] -= _cg_x_100_reg;
        buf1[7] += _cg_x_100_reg;
      }
      {
        // New offset [1, 0, 7]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = _cg_x000_vecbuf;
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf0[7] -= 15.0 * _cg_x_100_reg;
        buf1[7] -= 15.0 * _cg_x_100_reg;
        buf2[7] -= 15.0 * _cg_x_100_reg;
        buf3[7] -= 15.0 * _cg_x_100_reg;
        buf4[7] -= 15.0 * _cg_x_100_reg;
        buf4[6] += _cg_x_100_reg;
        buf5[7] -= 15.0 * _cg_x_100_reg;
        buf5[6] += 15.0 * _cg_x_100_reg;
        buf5[5] -= _cg_x_100_reg;
      }
      {
        // New offset [2, 0, 7]
        bElem _cg_x_100_reg;
        {
          _cg_x000_vecbuf = x.dat[neighbor14 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_x_100_reg;
        buf1[7] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [3, 0, 7]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf1[7] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 1, 7]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[7] += _cg_x_100_reg;
        buf3[7] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [1, 2, 7]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf3[7] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 0, 8]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf4[7] += _cg_x_100_reg;
        buf5[7] += 15.0 * _cg_x_100_reg;
        buf5[6] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 0, 9]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor22 * x.step + 64 + hipThreadIdx_x];
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf5[7] -= _cg_x_100_reg;
      }
    }
  }
  bElem buf6[8];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 0]
            buf6[0 + rel] = 0;
          }
          _cg_rel2 += 1;
        }
      }
    }
    {
      bElem _cg_beta_i000_vecbuf;
      bElem _cg_beta_i_100_vecbuf;
      bElem _cg_beta_j000_vecbuf;
      bElem _cg_beta_k000_vecbuf;
      {
        // New offset [0, 0, 0]
        bElem _cg_beta_i000_reg;
        bElem _cg_beta_j000_reg;
        bElem _cg_beta_k000_reg;
        {
          _cg_beta_i000_vecbuf = beta_i.dat[neighbor13 * beta_i.step + hipThreadIdx_x];
          _cg_beta_j000_vecbuf = beta_j.dat[neighbor13 * beta_j.step + hipThreadIdx_x];
          _cg_beta_k000_vecbuf = beta_k.dat[neighbor13 * beta_k.step + hipThreadIdx_x];
          _cg_beta_i000_reg = _cg_beta_i000_vecbuf;
          _cg_beta_j000_reg = _cg_beta_j000_vecbuf;
          _cg_beta_k000_reg = _cg_beta_k000_vecbuf;
        }
        buf6[0] += _cg_beta_i000_reg * buf0[0];
        buf6[0] += _cg_beta_j000_reg * buf2[0];
        buf6[0] += _cg_beta_k000_reg * buf4[0];
      }
      {
        // New offset [1, 0, 0]
        bElem _cg_beta_i000_reg;
        {
          _cg_beta_i_100_vecbuf = _cg_beta_i000_vecbuf;
          _cg_beta_i000_vecbuf = beta_i.dat[neighbor14 * beta_i.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_beta_i_100_vecbuf ,_cg_beta_i000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_beta_i_100_vecbuf, _cg_beta_i000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i000_reg = _cg_vectmp0;
        }
        buf6[0] += _cg_beta_i000_reg * buf1[0];
      }
      {
        // New offset [0, 1, 0]
        bElem _cg_beta_j000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_j.dat[neighbor16 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_j.dat[neighbor13 * beta_j.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j000_vecbuf
          dev_shl(_cg_beta_j000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_beta_j000_reg = _cg_beta_j000_vecbuf;
        }
        buf6[0] += _cg_beta_j000_reg * buf3[0];
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 7; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 1]
            bElem _cg_beta_i000_reg;
            bElem _cg_beta_j000_reg;
            bElem _cg_beta_k000_reg;
            {
              _cg_beta_i000_vecbuf = beta_i.dat[neighbor13 * beta_i.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_j000_vecbuf = beta_j.dat[neighbor13 * beta_j.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_k000_vecbuf = beta_k.dat[neighbor13 * beta_k.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_i000_reg = _cg_beta_i000_vecbuf;
              _cg_beta_j000_reg = _cg_beta_j000_vecbuf;
              _cg_beta_k000_reg = _cg_beta_k000_vecbuf;
            }
            buf6[1 + rel] += _cg_beta_i000_reg * buf0[1 + rel];
            buf6[1 + rel] += _cg_beta_j000_reg * buf2[1 + rel];
            buf6[1 + rel] += _cg_beta_k000_reg * buf4[1 + rel];
            buf6[0 + rel] += _cg_beta_k000_reg * buf5[0 + rel];
          }
          {
            // New offset [1, 0, 1]
            bElem _cg_beta_i000_reg;
            {
              _cg_beta_i_100_vecbuf = _cg_beta_i000_vecbuf;
              _cg_beta_i000_vecbuf = beta_i.dat[neighbor14 * beta_i.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_beta_i_100_vecbuf ,_cg_beta_i000_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_beta_i_100_vecbuf, _cg_beta_i000_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_i000_reg = _cg_vectmp0;
            }
            buf6[1 + rel] += _cg_beta_i000_reg * buf1[1 + rel];
          }
          {
            // New offset [0, 1, 1]
            bElem _cg_beta_j000_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = beta_j.dat[neighbor16 * beta_j.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = beta_j.dat[neighbor13 * beta_j.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j000_vecbuf
              dev_shl(_cg_beta_j000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              _cg_beta_j000_reg = _cg_beta_j000_vecbuf;
            }
            buf6[1 + rel] += _cg_beta_j000_reg * buf3[1 + rel];
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [0, 0, 8]
        bElem _cg_beta_k000_reg;
        {
          _cg_beta_k000_vecbuf = beta_k.dat[neighbor22 * beta_k.step + hipThreadIdx_x];
          _cg_beta_k000_reg = _cg_beta_k000_vecbuf;
        }
        buf6[7] += _cg_beta_k000_reg * buf5[7];
      }
    }
  }
  bElem buf7[8];
  bElem buf8[8];
  bElem buf9[8];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 0]
            buf5[0 + rel] = 0;
            buf4[0 + rel] = 0;
            buf3[0 + rel] = 0;
            buf2[0 + rel] = 0;
            buf1[0 + rel] = 0;
            buf0[0 + rel] = 0;
            buf7[0 + rel] = 0;
            buf8[0 + rel] = 0;
            buf9[0 + rel] = 0;
          }
          _cg_rel2 += 1;
        }
      }
    }
    {
      bElem _cg_x_110_vecbuf;
      bElem _cg_x010_vecbuf;
      {
        // New offset [1, -2, -1]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor1 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf2[0] -= _cg_x_110_reg;
        buf0[0] -= _cg_x_110_reg;
      }
      {
        // New offset [0, -1, -1]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = x.dat[neighbor3 * x.step + 448 + hipThreadIdx_x];
          _cg_x010_vecbuf = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf4[0] -= _cg_x_110_reg;
        buf1[0] -= _cg_x_110_reg;
      }
      {
        // New offset [1, -1, -1]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf4[0] += _cg_x_110_reg;
        buf2[0] += _cg_x_110_reg;
        buf8[0] += _cg_x_110_reg;
      }
      {
        // New offset [2, -1, -1]
        bElem _cg_x_110_reg;
        {
          _cg_x010_vecbuf = x.dat[neighbor5 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf1[0] += _cg_x_110_reg;
        buf8[0] -= _cg_x_110_reg;
      }
      {
        // New offset [1, 0, -1]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor7 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf0[0] += _cg_x_110_reg;
      }
      {
        // New offset [0, -2, 0]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor12 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor9 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = x.dat[neighbor10 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp4;
        }
        buf5[0] -= _cg_x_110_reg;
        buf3[0] -= _cg_x_110_reg;
      }
      {
        // New offset [1, -2, 0]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf5[0] += _cg_x_110_reg;
        buf0[0] += _cg_x_110_reg;
        buf0[1] -= _cg_x_110_reg;
        buf7[0] += _cg_x_110_reg;
        buf2[1] -= _cg_x_110_reg;
      }
      {
        // New offset [2, -2, 0]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor14 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor11 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp2;
        }
        buf3[0] += _cg_x_110_reg;
        buf7[0] -= _cg_x_110_reg;
      }
      {
        // New offset [0, -1, 0]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = x.dat[neighbor12 * x.step + hipThreadIdx_x];
          _cg_x010_vecbuf = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf4[1] -= _cg_x_110_reg;
        buf1[1] -= _cg_x_110_reg;
        buf1[0] += _cg_x_110_reg;
        buf3[0] += _cg_x_110_reg;
        buf9[0] += _cg_x_110_reg;
      }
      {
        // New offset [1, -1, 0]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf4[1] += _cg_x_110_reg;
        buf2[1] += _cg_x_110_reg;
        buf8[1] += _cg_x_110_reg;
      }
      {
        // New offset [2, -1, 0]
        bElem _cg_x_110_reg;
        {
          _cg_x010_vecbuf = x.dat[neighbor14 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf3[0] -= _cg_x_110_reg;
        buf1[0] -= _cg_x_110_reg;
        buf1[1] += _cg_x_110_reg;
        buf9[0] -= _cg_x_110_reg;
        buf8[1] -= _cg_x_110_reg;
      }
      {
        // New offset [0, 0, 0]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor15 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor12 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = x.dat[neighbor16 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp4;
        }
        buf5[0] += _cg_x_110_reg;
        buf9[0] -= _cg_x_110_reg;
      }
      {
        // New offset [1, 0, 0]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf5[0] -= _cg_x_110_reg;
        buf0[0] -= _cg_x_110_reg;
        buf0[1] += _cg_x_110_reg;
        buf7[0] -= _cg_x_110_reg;
      }
      {
        // New offset [2, 0, 0]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor17 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor14 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp2;
        }
        buf7[0] += _cg_x_110_reg;
        buf9[0] += _cg_x_110_reg;
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 6; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, -2, 1]
            bElem _cg_x_110_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor12 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor9 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_110_vecbuf
              dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = x.dat[neighbor10 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_x010_vecbuf
              dev_shl(_cg_x010_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_x_110_reg = _cg_vectmp4;
            }
            buf5[1 + rel] -= _cg_x_110_reg;
            buf3[1 + rel] -= _cg_x_110_reg;
          }
          {
            // New offset [1, -2, 1]
            bElem _cg_x_110_reg;
            {
              _cg_x_110_vecbuf = _cg_x010_vecbuf;
              _cg_x_110_reg = _cg_x_110_vecbuf;
            }
            buf5[1 + rel] += _cg_x_110_reg;
            buf0[1 + rel] += _cg_x_110_reg;
            buf0[2 + rel] -= _cg_x_110_reg;
            buf7[1 + rel] += _cg_x_110_reg;
            buf2[0 + rel] += _cg_x_110_reg;
            buf2[2 + rel] -= _cg_x_110_reg;
          }
          {
            // New offset [2, -2, 1]
            bElem _cg_x_110_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor14 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor11 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x010_vecbuf
              dev_shl(_cg_x010_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_x_110_reg = _cg_vectmp2;
            }
            buf3[1 + rel] += _cg_x_110_reg;
            buf7[1 + rel] -= _cg_x_110_reg;
          }
          {
            // New offset [0, -1, 1]
            bElem _cg_x_110_reg;
            {
              _cg_x_110_vecbuf = x.dat[neighbor12 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_x010_vecbuf = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_x_110_reg = _cg_vectmp0;
            }
            buf4[0 + rel] += _cg_x_110_reg;
            buf4[2 + rel] -= _cg_x_110_reg;
            buf1[2 + rel] -= _cg_x_110_reg;
            buf1[1 + rel] += _cg_x_110_reg;
            buf3[1 + rel] += _cg_x_110_reg;
            buf9[1 + rel] += _cg_x_110_reg;
          }
          {
            // New offset [1, -1, 1]
            bElem _cg_x_110_reg;
            {
              _cg_x_110_vecbuf = _cg_x010_vecbuf;
              _cg_x_110_reg = _cg_x_110_vecbuf;
            }
            buf4[0 + rel] -= _cg_x_110_reg;
            buf4[2 + rel] += _cg_x_110_reg;
            buf2[0 + rel] -= _cg_x_110_reg;
            buf2[2 + rel] += _cg_x_110_reg;
            buf8[0 + rel] -= _cg_x_110_reg;
            buf8[2 + rel] += _cg_x_110_reg;
          }
          {
            // New offset [2, -1, 1]
            bElem _cg_x_110_reg;
            {
              _cg_x010_vecbuf = x.dat[neighbor14 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_x_110_reg = _cg_vectmp0;
            }
            buf3[1 + rel] -= _cg_x_110_reg;
            buf1[1 + rel] -= _cg_x_110_reg;
            buf1[2 + rel] += _cg_x_110_reg;
            buf9[1 + rel] -= _cg_x_110_reg;
            buf8[2 + rel] -= _cg_x_110_reg;
            buf8[0 + rel] += _cg_x_110_reg;
          }
          {
            // New offset [0, 0, 1]
            bElem _cg_x_110_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor15 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor12 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_110_vecbuf
              dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = x.dat[neighbor16 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_x010_vecbuf
              dev_shl(_cg_x010_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_x_110_reg = _cg_vectmp4;
            }
            buf5[1 + rel] += _cg_x_110_reg;
            buf9[1 + rel] -= _cg_x_110_reg;
          }
          {
            // New offset [1, 0, 1]
            bElem _cg_x_110_reg;
            {
              _cg_x_110_vecbuf = _cg_x010_vecbuf;
              _cg_x_110_reg = _cg_x_110_vecbuf;
            }
            buf5[1 + rel] -= _cg_x_110_reg;
            buf0[1 + rel] -= _cg_x_110_reg;
            buf0[2 + rel] += _cg_x_110_reg;
            buf7[1 + rel] -= _cg_x_110_reg;
          }
          {
            // New offset [2, 0, 1]
            bElem _cg_x_110_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor17 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor14 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x010_vecbuf
              dev_shl(_cg_x010_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_x_110_reg = _cg_vectmp2;
            }
            buf7[1 + rel] += _cg_x_110_reg;
            buf9[1 + rel] += _cg_x_110_reg;
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [0, -2, 7]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor12 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor9 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = x.dat[neighbor10 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp4;
        }
        buf5[7] -= _cg_x_110_reg;
        buf3[7] -= _cg_x_110_reg;
      }
      {
        // New offset [1, -2, 7]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf5[7] += _cg_x_110_reg;
        buf0[7] += _cg_x_110_reg;
        buf7[7] += _cg_x_110_reg;
        buf2[6] += _cg_x_110_reg;
      }
      {
        // New offset [2, -2, 7]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor14 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor11 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp2;
        }
        buf3[7] += _cg_x_110_reg;
        buf7[7] -= _cg_x_110_reg;
      }
      {
        // New offset [0, -1, 7]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = x.dat[neighbor12 * x.step + 448 + hipThreadIdx_x];
          _cg_x010_vecbuf = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf4[6] += _cg_x_110_reg;
        buf3[7] += _cg_x_110_reg;
        buf1[7] += _cg_x_110_reg;
        buf9[7] += _cg_x_110_reg;
      }
      {
        // New offset [1, -1, 7]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf4[6] -= _cg_x_110_reg;
        buf2[6] -= _cg_x_110_reg;
        buf8[6] -= _cg_x_110_reg;
      }
      {
        // New offset [2, -1, 7]
        bElem _cg_x_110_reg;
        {
          _cg_x010_vecbuf = x.dat[neighbor14 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf3[7] -= _cg_x_110_reg;
        buf1[7] -= _cg_x_110_reg;
        buf9[7] -= _cg_x_110_reg;
        buf8[6] += _cg_x_110_reg;
      }
      {
        // New offset [0, 0, 7]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor15 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor12 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = x.dat[neighbor16 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp4;
        }
        buf5[7] += _cg_x_110_reg;
        buf9[7] -= _cg_x_110_reg;
      }
      {
        // New offset [1, 0, 7]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf5[7] -= _cg_x_110_reg;
        buf0[7] -= _cg_x_110_reg;
        buf7[7] -= _cg_x_110_reg;
      }
      {
        // New offset [2, 0, 7]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor17 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor14 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp2;
        }
        buf7[7] += _cg_x_110_reg;
        buf9[7] += _cg_x_110_reg;
      }
      {
        // New offset [1, -2, 8]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor19 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf2[7] += _cg_x_110_reg;
      }
      {
        // New offset [0, -1, 8]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = x.dat[neighbor21 * x.step + hipThreadIdx_x];
          _cg_x010_vecbuf = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf4[7] += _cg_x_110_reg;
      }
      {
        // New offset [1, -1, 8]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf4[7] -= _cg_x_110_reg;
        buf2[7] -= _cg_x_110_reg;
        buf8[7] -= _cg_x_110_reg;
      }
      {
        // New offset [2, -1, 8]
        bElem _cg_x_110_reg;
        {
          _cg_x010_vecbuf = x.dat[neighbor23 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf8[7] += _cg_x_110_reg;
      }
    }
  }
  bElem buf10[8];
  bElem buf11[8];
  bElem buf12[8];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 0]
            buf10[0 + rel] = 0;
            buf11[0 + rel] = 0;
            buf12[0 + rel] = 0;
          }
          _cg_rel2 += 1;
        }
      }
    }
    {
      bElem _cg_x011_vecbuf;
      bElem _cg_x_111_vecbuf;
      {
        // New offset [0, -1, -2]
        bElem _cg_x011_reg;
        {
          _cg_x011_vecbuf = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[0] += _cg_x011_reg;
      }
      {
        // New offset [0, 0, -2]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor7 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[0] -= _cg_x011_reg;
      }
      {
        // New offset [0, -2, -1]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf12[0] += _cg_x011_reg;
      }
      {
        // New offset [-1, -1, -1]
        bElem _cg_x011_reg;
        {
          _cg_x_111_vecbuf = x.dat[neighbor12 * x.step + hipThreadIdx_x];
          _cg_x011_vecbuf = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x011_reg = _cg_vectmp0;
        }
        buf11[0] += _cg_x011_reg;
      }
      {
        // New offset [0, -1, -1]
        bElem _cg_x011_reg;
        {
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[1] += _cg_x011_reg;
      }
      {
        // New offset [1, -1, -1]
        bElem _cg_x011_reg;
        {
          _cg_x_111_vecbuf = _cg_x011_vecbuf;
          _cg_x011_vecbuf = x.dat[neighbor14 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x011_reg = _cg_vectmp0;
        }
        buf11[0] -= _cg_x011_reg;
      }
      {
        // New offset [0, 0, -1]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[1] -= _cg_x011_reg;
        buf12[0] -= _cg_x011_reg;
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 6; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, -2, 0]
            bElem _cg_x011_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor10 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x011_vecbuf
              dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              _cg_x011_reg = _cg_x011_vecbuf;
            }
            buf12[0 + rel] -= _cg_x011_reg;
            buf12[1 + rel] += _cg_x011_reg;
          }
          {
            // New offset [-1, -1, 0]
            bElem _cg_x011_reg;
            {
              _cg_x_111_vecbuf = x.dat[neighbor12 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_x011_vecbuf = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_x011_reg = _cg_vectmp0;
            }
            buf11[0 + rel] -= _cg_x011_reg;
            buf11[1 + rel] += _cg_x011_reg;
          }
          {
            // New offset [0, -1, 0]
            bElem _cg_x011_reg;
            {
              _cg_x011_reg = _cg_x011_vecbuf;
            }
            buf10[0 + rel] -= _cg_x011_reg;
            buf10[2 + rel] += _cg_x011_reg;
          }
          {
            // New offset [1, -1, 0]
            bElem _cg_x011_reg;
            {
              _cg_x_111_vecbuf = _cg_x011_vecbuf;
              _cg_x011_vecbuf = x.dat[neighbor14 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_x011_reg = _cg_vectmp0;
            }
            buf11[0 + rel] += _cg_x011_reg;
            buf11[1 + rel] -= _cg_x011_reg;
          }
          {
            // New offset [0, 0, 0]
            bElem _cg_x011_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor16 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x011_vecbuf
              dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              _cg_x011_reg = _cg_x011_vecbuf;
            }
            buf10[0 + rel] += _cg_x011_reg;
            buf10[2 + rel] -= _cg_x011_reg;
            buf12[0 + rel] += _cg_x011_reg;
            buf12[1 + rel] -= _cg_x011_reg;
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [0, -2, 6]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf12[6] -= _cg_x011_reg;
        buf12[7] += _cg_x011_reg;
      }
      {
        // New offset [-1, -1, 6]
        bElem _cg_x011_reg;
        {
          _cg_x_111_vecbuf = x.dat[neighbor12 * x.step + 448 + hipThreadIdx_x];
          _cg_x011_vecbuf = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x011_reg = _cg_vectmp0;
        }
        buf11[6] -= _cg_x011_reg;
        buf11[7] += _cg_x011_reg;
      }
      {
        // New offset [0, -1, 6]
        bElem _cg_x011_reg;
        {
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[6] -= _cg_x011_reg;
      }
      {
        // New offset [1, -1, 6]
        bElem _cg_x011_reg;
        {
          _cg_x_111_vecbuf = _cg_x011_vecbuf;
          _cg_x011_vecbuf = x.dat[neighbor14 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x011_reg = _cg_vectmp0;
        }
        buf11[6] += _cg_x011_reg;
        buf11[7] -= _cg_x011_reg;
      }
      {
        // New offset [0, 0, 6]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[6] += _cg_x011_reg;
        buf12[6] += _cg_x011_reg;
        buf12[7] -= _cg_x011_reg;
      }
      {
        // New offset [0, -2, 7]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor19 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf12[7] -= _cg_x011_reg;
      }
      {
        // New offset [-1, -1, 7]
        bElem _cg_x011_reg;
        {
          _cg_x_111_vecbuf = x.dat[neighbor21 * x.step + hipThreadIdx_x];
          _cg_x011_vecbuf = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x011_reg = _cg_vectmp0;
        }
        buf11[7] -= _cg_x011_reg;
      }
      {
        // New offset [0, -1, 7]
        bElem _cg_x011_reg;
        {
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[7] -= _cg_x011_reg;
      }
      {
        // New offset [1, -1, 7]
        bElem _cg_x011_reg;
        {
          _cg_x_111_vecbuf = _cg_x011_vecbuf;
          _cg_x011_vecbuf = x.dat[neighbor23 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x011_reg = _cg_vectmp0;
        }
        buf11[7] += _cg_x011_reg;
      }
      {
        // New offset [0, 0, 7]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor25 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[7] += _cg_x011_reg;
        buf12[7] += _cg_x011_reg;
      }
    }
  }
  bElem buf13[8];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 0]
            buf13[0 + rel] = 0;
          }
          _cg_rel2 += 1;
        }
      }
    }
    {
      bElem _cg_beta_i010_vecbuf;
      bElem _cg_beta_i0_10_vecbuf;
      bElem _cg_beta_i01_2_vecbuf;
      bElem _cg_beta_i_110_vecbuf;
      bElem _cg_beta_i_1_10_vecbuf;
      bElem _cg_beta_i_11_2_vecbuf;
      bElem _cg_beta_j000_vecbuf;
      bElem _cg_beta_j100_vecbuf;
      bElem _cg_beta_j_100_vecbuf;
      bElem _cg_beta_j00_2_vecbuf;
      bElem _cg_beta_k000_vecbuf;
      bElem _cg_beta_k100_vecbuf;
      bElem _cg_beta_k_100_vecbuf;
      bElem _cg_beta_k0_20_vecbuf;
      {
        // New offset [0, 0, 0]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i0_10_reg;
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j_100_reg;
        bElem _cg_beta_k100_reg;
        bElem _cg_beta_k_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_i.dat[neighbor13 * beta_i.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_i.dat[neighbor10 * beta_i.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_i0_10_vecbuf
          dev_shl(_cg_beta_i0_10_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_i.dat[neighbor16 * beta_i.step + hipThreadIdx_x];
          // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_i010_vecbuf
          dev_shl(_cg_beta_i010_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          _cg_beta_j_100_vecbuf = beta_j.dat[neighbor12 * beta_j.step + hipThreadIdx_x];
          _cg_beta_j000_vecbuf = beta_j.dat[neighbor13 * beta_j.step + hipThreadIdx_x];
          _cg_beta_j100_vecbuf = beta_j.dat[neighbor14 * beta_j.step + hipThreadIdx_x];
          _cg_beta_k_100_vecbuf = beta_k.dat[neighbor12 * beta_k.step + hipThreadIdx_x];
          _cg_beta_k000_vecbuf = beta_k.dat[neighbor13 * beta_k.step + hipThreadIdx_x];
          _cg_beta_k100_vecbuf = beta_k.dat[neighbor14 * beta_k.step + hipThreadIdx_x];
          _cg_beta_i010_reg = _cg_beta_i010_vecbuf;
          _cg_beta_i0_10_reg = _cg_beta_i0_10_vecbuf;
          bElem _cg_vectmp3;
          // merge0 _cg_beta_j000_vecbuf ,_cg_beta_j100_vecbuf, 1 -> _cg_vectmp3
          dev_shl(_cg_vectmp3, _cg_beta_j000_vecbuf, _cg_beta_j100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_j100_reg = _cg_vectmp3;
          bElem _cg_vectmp4;
          // merge0 _cg_beta_j_100_vecbuf ,_cg_beta_j000_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_beta_j_100_vecbuf, _cg_beta_j000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_j_100_reg = _cg_vectmp4;
          bElem _cg_vectmp5;
          // merge0 _cg_beta_k000_vecbuf ,_cg_beta_k100_vecbuf, 1 -> _cg_vectmp5
          dev_shl(_cg_vectmp5, _cg_beta_k000_vecbuf, _cg_beta_k100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_k100_reg = _cg_vectmp5;
          bElem _cg_vectmp6;
          // merge0 _cg_beta_k_100_vecbuf ,_cg_beta_k000_vecbuf, 7 -> _cg_vectmp6
          dev_shl(_cg_vectmp6, _cg_beta_k_100_vecbuf, _cg_beta_k000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_k_100_reg = _cg_vectmp6;
        }
        buf13[0] += (_cg_beta_i010_reg - _cg_beta_i0_10_reg) * buf5[0];
        buf13[0] += (_cg_beta_j100_reg - _cg_beta_j_100_reg) * buf3[0];
        buf13[0] += (_cg_beta_k100_reg - _cg_beta_k_100_reg) * buf1[0];
      }
      {
        // New offset [1, 0, 0]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i0_10_reg;
        {
          _cg_beta_i_1_10_vecbuf = _cg_beta_i0_10_vecbuf;
          _cg_beta_i_110_vecbuf = _cg_beta_i010_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_i.dat[neighbor14 * beta_i.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_i.dat[neighbor11 * beta_i.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_i0_10_vecbuf
          dev_shl(_cg_beta_i0_10_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_i.dat[neighbor17 * beta_i.step + hipThreadIdx_x];
          // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_i010_vecbuf
          dev_shl(_cg_beta_i010_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp3;
          // merge0 _cg_beta_i_110_vecbuf ,_cg_beta_i010_vecbuf, 1 -> _cg_vectmp3
          dev_shl(_cg_vectmp3, _cg_beta_i_110_vecbuf, _cg_beta_i010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i010_reg = _cg_vectmp3;
          bElem _cg_vectmp4;
          // merge0 _cg_beta_i_1_10_vecbuf ,_cg_beta_i0_10_vecbuf, 1 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_beta_i_1_10_vecbuf, _cg_beta_i0_10_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i0_10_reg = _cg_vectmp4;
        }
        buf13[0] += (_cg_beta_i010_reg - _cg_beta_i0_10_reg) * buf7[0];
      }
      {
        // New offset [-1, 1, 0]
        bElem _cg_beta_k100_reg;
        bElem _cg_beta_k1_20_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_k.dat[neighbor13 * beta_k.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_k.dat[neighbor10 * beta_k.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_k0_20_vecbuf
          dev_shl(_cg_beta_k0_20_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_k.dat[neighbor16 * beta_k.step + hipThreadIdx_x];
          // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_k000_vecbuf
          dev_shl(_cg_beta_k000_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
          _cg_beta_k1_20_reg = _cg_beta_k0_20_vecbuf;
        }
        buf13[0] += (_cg_beta_k100_reg - _cg_beta_k1_20_reg) * buf0[0];
      }
      {
        // New offset [0, 1, 0]
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_j.dat[neighbor15 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_j.dat[neighbor12 * beta_j.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j_100_vecbuf
          dev_shl(_cg_beta_j_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_j.dat[neighbor16 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = beta_j.dat[neighbor13 * beta_j.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_beta_j000_vecbuf
          dev_shl(_cg_beta_j000_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          _cg_vectmp4 = beta_j.dat[neighbor17 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp5;
          _cg_vectmp5 = beta_j.dat[neighbor14 * beta_j.step + hipThreadIdx_x];
          // merge1 _cg_vectmp5 ,_cg_vectmp4, 1 -> _cg_beta_j100_vecbuf
          dev_shl(_cg_beta_j100_vecbuf, _cg_vectmp5, _cg_vectmp4, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp6;
          // merge0 _cg_beta_j000_vecbuf ,_cg_beta_j100_vecbuf, 1 -> _cg_vectmp6
          dev_shl(_cg_vectmp6, _cg_beta_j000_vecbuf, _cg_beta_j100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_j100_reg = _cg_vectmp6;
          bElem _cg_vectmp7;
          // merge0 _cg_beta_j_100_vecbuf ,_cg_beta_j000_vecbuf, 7 -> _cg_vectmp7
          dev_shl(_cg_vectmp7, _cg_beta_j_100_vecbuf, _cg_beta_j000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_j_100_reg = _cg_vectmp7;
        }
        buf13[0] += (_cg_beta_j100_reg - _cg_beta_j_100_reg) * buf9[0];
      }
      {
        // New offset [0, -1, 1]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i01_2_reg;
        {
          _cg_beta_i01_2_vecbuf = beta_i.dat[neighbor4 * beta_i.step + 448 + hipThreadIdx_x];
          _cg_beta_i010_vecbuf = beta_i.dat[neighbor13 * beta_i.step + 64 + hipThreadIdx_x];
          _cg_beta_i010_reg = _cg_beta_i010_vecbuf;
          _cg_beta_i01_2_reg = _cg_beta_i01_2_vecbuf;
        }
        buf13[0] += (_cg_beta_i010_reg - _cg_beta_i01_2_reg) * buf4[0];
      }
      {
        // New offset [1, -1, 1]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i01_2_reg;
        {
          _cg_beta_i_11_2_vecbuf = _cg_beta_i01_2_vecbuf;
          _cg_beta_i_110_vecbuf = _cg_beta_i010_vecbuf;
          _cg_beta_i01_2_vecbuf = beta_i.dat[neighbor5 * beta_i.step + 448 + hipThreadIdx_x];
          _cg_beta_i010_vecbuf = beta_i.dat[neighbor14 * beta_i.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_beta_i_110_vecbuf ,_cg_beta_i010_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_beta_i_110_vecbuf, _cg_beta_i010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i010_reg = _cg_vectmp0;
          bElem _cg_vectmp1;
          // merge0 _cg_beta_i_11_2_vecbuf ,_cg_beta_i01_2_vecbuf, 1 -> _cg_vectmp1
          dev_shl(_cg_vectmp1, _cg_beta_i_11_2_vecbuf, _cg_beta_i01_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i01_2_reg = _cg_vectmp1;
        }
        buf13[0] += (_cg_beta_i010_reg - _cg_beta_i01_2_reg) * buf8[0];
      }
      {
        // New offset [-1, 0, 1]
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j10_2_reg;
        {
          _cg_beta_j00_2_vecbuf = beta_j.dat[neighbor4 * beta_j.step + 448 + hipThreadIdx_x];
          _cg_beta_j000_vecbuf = beta_j.dat[neighbor13 * beta_j.step + 64 + hipThreadIdx_x];
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_beta_j10_2_reg = _cg_beta_j00_2_vecbuf;
        }
        buf13[0] += (_cg_beta_j100_reg - _cg_beta_j10_2_reg) * buf2[0];
      }
      {
        // New offset [0, 0, 1]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i0_10_reg;
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j_100_reg;
        bElem _cg_beta_k100_reg;
        bElem _cg_beta_k_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_i.dat[neighbor13 * beta_i.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_i.dat[neighbor10 * beta_i.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_i0_10_vecbuf
          dev_shl(_cg_beta_i0_10_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_i.dat[neighbor16 * beta_i.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_i010_vecbuf
          dev_shl(_cg_beta_i010_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          _cg_beta_j_100_vecbuf = beta_j.dat[neighbor12 * beta_j.step + 64 + hipThreadIdx_x];
          _cg_beta_j100_vecbuf = beta_j.dat[neighbor14 * beta_j.step + 64 + hipThreadIdx_x];
          _cg_beta_k_100_vecbuf = beta_k.dat[neighbor12 * beta_k.step + 64 + hipThreadIdx_x];
          _cg_beta_k000_vecbuf = beta_k.dat[neighbor13 * beta_k.step + 64 + hipThreadIdx_x];
          _cg_beta_k100_vecbuf = beta_k.dat[neighbor14 * beta_k.step + 64 + hipThreadIdx_x];
          _cg_beta_i010_reg = _cg_beta_i010_vecbuf;
          _cg_beta_i0_10_reg = _cg_beta_i0_10_vecbuf;
          bElem _cg_vectmp3;
          // merge0 _cg_beta_j000_vecbuf ,_cg_beta_j100_vecbuf, 1 -> _cg_vectmp3
          dev_shl(_cg_vectmp3, _cg_beta_j000_vecbuf, _cg_beta_j100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_j100_reg = _cg_vectmp3;
          bElem _cg_vectmp4;
          // merge0 _cg_beta_j_100_vecbuf ,_cg_beta_j000_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_beta_j_100_vecbuf, _cg_beta_j000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_j_100_reg = _cg_vectmp4;
          bElem _cg_vectmp5;
          // merge0 _cg_beta_k000_vecbuf ,_cg_beta_k100_vecbuf, 1 -> _cg_vectmp5
          dev_shl(_cg_vectmp5, _cg_beta_k000_vecbuf, _cg_beta_k100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_k100_reg = _cg_vectmp5;
          bElem _cg_vectmp6;
          // merge0 _cg_beta_k_100_vecbuf ,_cg_beta_k000_vecbuf, 7 -> _cg_vectmp6
          dev_shl(_cg_vectmp6, _cg_beta_k_100_vecbuf, _cg_beta_k000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_k_100_reg = _cg_vectmp6;
        }
        buf13[1] += (_cg_beta_i010_reg - _cg_beta_i0_10_reg) * buf5[1];
        buf13[1] += (_cg_beta_j100_reg - _cg_beta_j_100_reg) * buf3[1];
        buf13[1] += (_cg_beta_k100_reg - _cg_beta_k_100_reg) * buf1[1];
        buf13[0] += (_cg_beta_k100_reg - _cg_beta_k_100_reg) * buf11[0];
      }
      {
        // New offset [1, 0, 1]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i0_10_reg;
        {
          _cg_beta_i_1_10_vecbuf = _cg_beta_i0_10_vecbuf;
          _cg_beta_i_110_vecbuf = _cg_beta_i010_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_i.dat[neighbor14 * beta_i.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_i.dat[neighbor11 * beta_i.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_i0_10_vecbuf
          dev_shl(_cg_beta_i0_10_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_i.dat[neighbor17 * beta_i.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_i010_vecbuf
          dev_shl(_cg_beta_i010_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp3;
          // merge0 _cg_beta_i_110_vecbuf ,_cg_beta_i010_vecbuf, 1 -> _cg_vectmp3
          dev_shl(_cg_vectmp3, _cg_beta_i_110_vecbuf, _cg_beta_i010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i010_reg = _cg_vectmp3;
          bElem _cg_vectmp4;
          // merge0 _cg_beta_i_1_10_vecbuf ,_cg_beta_i0_10_vecbuf, 1 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_beta_i_1_10_vecbuf, _cg_beta_i0_10_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i0_10_reg = _cg_vectmp4;
        }
        buf13[1] += (_cg_beta_i010_reg - _cg_beta_i0_10_reg) * buf7[1];
      }
      {
        // New offset [-1, 1, 1]
        bElem _cg_beta_k100_reg;
        bElem _cg_beta_k1_20_reg;
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j10_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_k.dat[neighbor13 * beta_k.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_k.dat[neighbor10 * beta_k.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_k0_20_vecbuf
          dev_shl(_cg_beta_k0_20_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_k.dat[neighbor16 * beta_k.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_k000_vecbuf
          dev_shl(_cg_beta_k000_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp3;
          _cg_vectmp3 = beta_j.dat[neighbor7 * beta_j.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp4;
          _cg_vectmp4 = beta_j.dat[neighbor4 * beta_j.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp4 ,_cg_vectmp3, 1 -> _cg_beta_j00_2_vecbuf
          dev_shl(_cg_beta_j00_2_vecbuf, _cg_vectmp4, _cg_vectmp3, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp5;
          _cg_vectmp5 = beta_j.dat[neighbor16 * beta_j.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp6;
          _cg_vectmp6 = beta_j.dat[neighbor13 * beta_j.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp6 ,_cg_vectmp5, 1 -> _cg_beta_j000_vecbuf
          dev_shl(_cg_beta_j000_vecbuf, _cg_vectmp6, _cg_vectmp5, 56, 64, hipThreadIdx_x);
          _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
          _cg_beta_k1_20_reg = _cg_beta_k0_20_vecbuf;
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_beta_j10_2_reg = _cg_beta_j00_2_vecbuf;
        }
        buf13[1] += (_cg_beta_k100_reg - _cg_beta_k1_20_reg) * buf0[1];
        buf13[0] += (_cg_beta_j100_reg - _cg_beta_j10_2_reg) * buf10[0];
        buf13[0] += (_cg_beta_k100_reg - _cg_beta_k1_20_reg) * buf12[0];
      }
      {
        // New offset [0, 1, 1]
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_j.dat[neighbor15 * beta_j.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_j.dat[neighbor12 * beta_j.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j_100_vecbuf
          dev_shl(_cg_beta_j_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_j.dat[neighbor17 * beta_j.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = beta_j.dat[neighbor14 * beta_j.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_beta_j100_vecbuf
          dev_shl(_cg_beta_j100_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_beta_j000_vecbuf ,_cg_beta_j100_vecbuf, 1 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_beta_j000_vecbuf, _cg_beta_j100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_j100_reg = _cg_vectmp4;
          bElem _cg_vectmp5;
          // merge0 _cg_beta_j_100_vecbuf ,_cg_beta_j000_vecbuf, 7 -> _cg_vectmp5
          dev_shl(_cg_vectmp5, _cg_beta_j_100_vecbuf, _cg_beta_j000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_j_100_reg = _cg_vectmp5;
        }
        buf13[1] += (_cg_beta_j100_reg - _cg_beta_j_100_reg) * buf9[1];
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 6; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, -1, 2]
            bElem _cg_beta_i010_reg;
            bElem _cg_beta_i01_2_reg;
            {
              _cg_beta_i01_2_vecbuf = beta_i.dat[neighbor13 * beta_i.step + (hipThreadIdx_x + rel * 64)];
              _cg_beta_i010_vecbuf = beta_i.dat[neighbor13 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_i010_reg = _cg_beta_i010_vecbuf;
              _cg_beta_i01_2_reg = _cg_beta_i01_2_vecbuf;
            }
            buf13[1 + rel] += (_cg_beta_i010_reg - _cg_beta_i01_2_reg) * buf4[1 + rel];
          }
          {
            // New offset [1, -1, 2]
            bElem _cg_beta_i010_reg;
            bElem _cg_beta_i01_2_reg;
            {
              _cg_beta_i_11_2_vecbuf = _cg_beta_i01_2_vecbuf;
              _cg_beta_i_110_vecbuf = _cg_beta_i010_vecbuf;
              _cg_beta_i01_2_vecbuf = beta_i.dat[neighbor14 * beta_i.step + (hipThreadIdx_x + rel * 64)];
              _cg_beta_i010_vecbuf = beta_i.dat[neighbor14 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_beta_i_110_vecbuf ,_cg_beta_i010_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_beta_i_110_vecbuf, _cg_beta_i010_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_i010_reg = _cg_vectmp0;
              bElem _cg_vectmp1;
              // merge0 _cg_beta_i_11_2_vecbuf ,_cg_beta_i01_2_vecbuf, 1 -> _cg_vectmp1
              dev_shl(_cg_vectmp1, _cg_beta_i_11_2_vecbuf, _cg_beta_i01_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_i01_2_reg = _cg_vectmp1;
            }
            buf13[1 + rel] += (_cg_beta_i010_reg - _cg_beta_i01_2_reg) * buf8[1 + rel];
          }
          {
            // New offset [-1, 0, 2]
            bElem _cg_beta_j100_reg;
            bElem _cg_beta_j10_2_reg;
            {
              _cg_beta_j00_2_vecbuf = beta_j.dat[neighbor13 * beta_j.step + (hipThreadIdx_x + rel * 64)];
              _cg_beta_j000_vecbuf = beta_j.dat[neighbor13 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
              _cg_beta_j10_2_reg = _cg_beta_j00_2_vecbuf;
            }
            buf13[1 + rel] += (_cg_beta_j100_reg - _cg_beta_j10_2_reg) * buf2[1 + rel];
          }
          {
            // New offset [0, 0, 2]
            bElem _cg_beta_i010_reg;
            bElem _cg_beta_i0_10_reg;
            bElem _cg_beta_j100_reg;
            bElem _cg_beta_j_100_reg;
            bElem _cg_beta_k100_reg;
            bElem _cg_beta_k_100_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = beta_i.dat[neighbor13 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = beta_i.dat[neighbor10 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_i0_10_vecbuf
              dev_shl(_cg_beta_i0_10_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = beta_i.dat[neighbor16 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_i010_vecbuf
              dev_shl(_cg_beta_i010_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
              _cg_beta_j_100_vecbuf = beta_j.dat[neighbor12 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_j100_vecbuf = beta_j.dat[neighbor14 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_k_100_vecbuf = beta_k.dat[neighbor12 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_k000_vecbuf = beta_k.dat[neighbor13 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_k100_vecbuf = beta_k.dat[neighbor14 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_i010_reg = _cg_beta_i010_vecbuf;
              _cg_beta_i0_10_reg = _cg_beta_i0_10_vecbuf;
              bElem _cg_vectmp3;
              // merge0 _cg_beta_j000_vecbuf ,_cg_beta_j100_vecbuf, 1 -> _cg_vectmp3
              dev_shl(_cg_vectmp3, _cg_beta_j000_vecbuf, _cg_beta_j100_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_j100_reg = _cg_vectmp3;
              bElem _cg_vectmp4;
              // merge0 _cg_beta_j_100_vecbuf ,_cg_beta_j000_vecbuf, 7 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_beta_j_100_vecbuf, _cg_beta_j000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_beta_j_100_reg = _cg_vectmp4;
              bElem _cg_vectmp5;
              // merge0 _cg_beta_k000_vecbuf ,_cg_beta_k100_vecbuf, 1 -> _cg_vectmp5
              dev_shl(_cg_vectmp5, _cg_beta_k000_vecbuf, _cg_beta_k100_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_k100_reg = _cg_vectmp5;
              bElem _cg_vectmp6;
              // merge0 _cg_beta_k_100_vecbuf ,_cg_beta_k000_vecbuf, 7 -> _cg_vectmp6
              dev_shl(_cg_vectmp6, _cg_beta_k_100_vecbuf, _cg_beta_k000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_beta_k_100_reg = _cg_vectmp6;
            }
            buf13[2 + rel] += (_cg_beta_i010_reg - _cg_beta_i0_10_reg) * buf5[2 + rel];
            buf13[2 + rel] += (_cg_beta_j100_reg - _cg_beta_j_100_reg) * buf3[2 + rel];
            buf13[2 + rel] += (_cg_beta_k100_reg - _cg_beta_k_100_reg) * buf1[2 + rel];
            buf13[1 + rel] += (_cg_beta_k100_reg - _cg_beta_k_100_reg) * buf11[1 + rel];
          }
          {
            // New offset [1, 0, 2]
            bElem _cg_beta_i010_reg;
            bElem _cg_beta_i0_10_reg;
            {
              _cg_beta_i_1_10_vecbuf = _cg_beta_i0_10_vecbuf;
              _cg_beta_i_110_vecbuf = _cg_beta_i010_vecbuf;
              bElem _cg_vectmp0;
              _cg_vectmp0 = beta_i.dat[neighbor14 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = beta_i.dat[neighbor11 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_i0_10_vecbuf
              dev_shl(_cg_beta_i0_10_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = beta_i.dat[neighbor17 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_i010_vecbuf
              dev_shl(_cg_beta_i010_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp3;
              // merge0 _cg_beta_i_110_vecbuf ,_cg_beta_i010_vecbuf, 1 -> _cg_vectmp3
              dev_shl(_cg_vectmp3, _cg_beta_i_110_vecbuf, _cg_beta_i010_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_i010_reg = _cg_vectmp3;
              bElem _cg_vectmp4;
              // merge0 _cg_beta_i_1_10_vecbuf ,_cg_beta_i0_10_vecbuf, 1 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_beta_i_1_10_vecbuf, _cg_beta_i0_10_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_i0_10_reg = _cg_vectmp4;
            }
            buf13[2 + rel] += (_cg_beta_i010_reg - _cg_beta_i0_10_reg) * buf7[2 + rel];
          }
          {
            // New offset [-1, 1, 2]
            bElem _cg_beta_k100_reg;
            bElem _cg_beta_k1_20_reg;
            bElem _cg_beta_j100_reg;
            bElem _cg_beta_j10_2_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = beta_k.dat[neighbor13 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = beta_k.dat[neighbor10 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_k0_20_vecbuf
              dev_shl(_cg_beta_k0_20_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = beta_k.dat[neighbor16 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_k000_vecbuf
              dev_shl(_cg_beta_k000_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp3;
              _cg_vectmp3 = beta_j.dat[neighbor16 * beta_j.step + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp4;
              _cg_vectmp4 = beta_j.dat[neighbor13 * beta_j.step + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp4 ,_cg_vectmp3, 1 -> _cg_beta_j00_2_vecbuf
              dev_shl(_cg_beta_j00_2_vecbuf, _cg_vectmp4, _cg_vectmp3, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp5;
              _cg_vectmp5 = beta_j.dat[neighbor16 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp6;
              _cg_vectmp6 = beta_j.dat[neighbor13 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp6 ,_cg_vectmp5, 1 -> _cg_beta_j000_vecbuf
              dev_shl(_cg_beta_j000_vecbuf, _cg_vectmp6, _cg_vectmp5, 56, 64, hipThreadIdx_x);
              _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
              _cg_beta_k1_20_reg = _cg_beta_k0_20_vecbuf;
              _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
              _cg_beta_j10_2_reg = _cg_beta_j00_2_vecbuf;
            }
            buf13[2 + rel] += (_cg_beta_k100_reg - _cg_beta_k1_20_reg) * buf0[2 + rel];
            buf13[1 + rel] += (_cg_beta_j100_reg - _cg_beta_j10_2_reg) * buf10[1 + rel];
            buf13[1 + rel] += (_cg_beta_k100_reg - _cg_beta_k1_20_reg) * buf12[1 + rel];
          }
          {
            // New offset [0, 1, 2]
            bElem _cg_beta_j100_reg;
            bElem _cg_beta_j_100_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = beta_j.dat[neighbor15 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = beta_j.dat[neighbor12 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j_100_vecbuf
              dev_shl(_cg_beta_j_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = beta_j.dat[neighbor17 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = beta_j.dat[neighbor14 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_beta_j100_vecbuf
              dev_shl(_cg_beta_j100_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_beta_j000_vecbuf ,_cg_beta_j100_vecbuf, 1 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_beta_j000_vecbuf, _cg_beta_j100_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_j100_reg = _cg_vectmp4;
              bElem _cg_vectmp5;
              // merge0 _cg_beta_j_100_vecbuf ,_cg_beta_j000_vecbuf, 7 -> _cg_vectmp5
              dev_shl(_cg_vectmp5, _cg_beta_j_100_vecbuf, _cg_beta_j000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_beta_j_100_reg = _cg_vectmp5;
            }
            buf13[2 + rel] += (_cg_beta_j100_reg - _cg_beta_j_100_reg) * buf9[2 + rel];
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [0, -1, 8]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i01_2_reg;
        {
          _cg_beta_i01_2_vecbuf = beta_i.dat[neighbor13 * beta_i.step + 384 + hipThreadIdx_x];
          _cg_beta_i010_vecbuf = beta_i.dat[neighbor22 * beta_i.step + hipThreadIdx_x];
          _cg_beta_i010_reg = _cg_beta_i010_vecbuf;
          _cg_beta_i01_2_reg = _cg_beta_i01_2_vecbuf;
        }
        buf13[7] += (_cg_beta_i010_reg - _cg_beta_i01_2_reg) * buf4[7];
      }
      {
        // New offset [1, -1, 8]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i01_2_reg;
        {
          _cg_beta_i_11_2_vecbuf = _cg_beta_i01_2_vecbuf;
          _cg_beta_i_110_vecbuf = _cg_beta_i010_vecbuf;
          _cg_beta_i01_2_vecbuf = beta_i.dat[neighbor14 * beta_i.step + 384 + hipThreadIdx_x];
          _cg_beta_i010_vecbuf = beta_i.dat[neighbor23 * beta_i.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_beta_i_110_vecbuf ,_cg_beta_i010_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_beta_i_110_vecbuf, _cg_beta_i010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i010_reg = _cg_vectmp0;
          bElem _cg_vectmp1;
          // merge0 _cg_beta_i_11_2_vecbuf ,_cg_beta_i01_2_vecbuf, 1 -> _cg_vectmp1
          dev_shl(_cg_vectmp1, _cg_beta_i_11_2_vecbuf, _cg_beta_i01_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i01_2_reg = _cg_vectmp1;
        }
        buf13[7] += (_cg_beta_i010_reg - _cg_beta_i01_2_reg) * buf8[7];
      }
      {
        // New offset [-1, 0, 8]
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j10_2_reg;
        {
          _cg_beta_j00_2_vecbuf = beta_j.dat[neighbor13 * beta_j.step + 384 + hipThreadIdx_x];
          _cg_beta_j000_vecbuf = beta_j.dat[neighbor22 * beta_j.step + hipThreadIdx_x];
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_beta_j10_2_reg = _cg_beta_j00_2_vecbuf;
        }
        buf13[7] += (_cg_beta_j100_reg - _cg_beta_j10_2_reg) * buf2[7];
      }
      {
        // New offset [0, 0, 8]
        bElem _cg_beta_k100_reg;
        bElem _cg_beta_k_100_reg;
        {
          _cg_beta_k_100_vecbuf = beta_k.dat[neighbor21 * beta_k.step + hipThreadIdx_x];
          _cg_beta_k000_vecbuf = beta_k.dat[neighbor22 * beta_k.step + hipThreadIdx_x];
          _cg_beta_k100_vecbuf = beta_k.dat[neighbor23 * beta_k.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_beta_k000_vecbuf ,_cg_beta_k100_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_beta_k000_vecbuf, _cg_beta_k100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_k100_reg = _cg_vectmp0;
          bElem _cg_vectmp1;
          // merge0 _cg_beta_k_100_vecbuf ,_cg_beta_k000_vecbuf, 7 -> _cg_vectmp1
          dev_shl(_cg_vectmp1, _cg_beta_k_100_vecbuf, _cg_beta_k000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_k_100_reg = _cg_vectmp1;
        }
        buf13[7] += (_cg_beta_k100_reg - _cg_beta_k_100_reg) * buf11[7];
      }
      {
        // New offset [-1, 1, 8]
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j10_2_reg;
        bElem _cg_beta_k100_reg;
        bElem _cg_beta_k1_20_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_j.dat[neighbor16 * beta_j.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_j.dat[neighbor13 * beta_j.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j00_2_vecbuf
          dev_shl(_cg_beta_j00_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_j.dat[neighbor25 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = beta_j.dat[neighbor22 * beta_j.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_beta_j000_vecbuf
          dev_shl(_cg_beta_j000_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          _cg_vectmp4 = beta_k.dat[neighbor22 * beta_k.step + hipThreadIdx_x];
          bElem _cg_vectmp5;
          _cg_vectmp5 = beta_k.dat[neighbor19 * beta_k.step + hipThreadIdx_x];
          // merge1 _cg_vectmp5 ,_cg_vectmp4, 7 -> _cg_beta_k0_20_vecbuf
          dev_shl(_cg_beta_k0_20_vecbuf, _cg_vectmp5, _cg_vectmp4, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp6;
          _cg_vectmp6 = beta_k.dat[neighbor25 * beta_k.step + hipThreadIdx_x];
          // merge1 _cg_vectmp4 ,_cg_vectmp6, 1 -> _cg_beta_k000_vecbuf
          dev_shl(_cg_beta_k000_vecbuf, _cg_vectmp4, _cg_vectmp6, 56, 64, hipThreadIdx_x);
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_beta_j10_2_reg = _cg_beta_j00_2_vecbuf;
          _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
          _cg_beta_k1_20_reg = _cg_beta_k0_20_vecbuf;
        }
        buf13[7] += (_cg_beta_j100_reg - _cg_beta_j10_2_reg) * buf10[7];
        buf13[7] += (_cg_beta_k100_reg - _cg_beta_k1_20_reg) * buf12[7];
      }
    }
  }
  {
    {
      bElem _cg_alpha000_vecbuf;
      bElem _cg_x000_vecbuf;
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 0]
            bElem _cg_alpha000_reg;
            bElem _cg_x000_reg;
            {
              _cg_alpha000_vecbuf = alpha.dat[neighbor13 * alpha.step + (hipThreadIdx_x + rel * 64)];
              _cg_x000_vecbuf = x.dat[neighbor13 * x.step + (hipThreadIdx_x + rel * 64)];
              _cg_alpha000_reg = _cg_alpha000_vecbuf;
              _cg_x000_reg = _cg_x000_vecbuf;
            }
            buf12[0 + rel] = c[0] * _cg_alpha000_reg * _cg_x000_reg - c[1] * c[2] * (0.0833 * buf6[0 + rel] + 0.020825 * buf13[0 + rel]);
          }
          _cg_rel2 += 1;
        }
      }
    }
    bElem *out_ref = &out.dat[neighbor13 * out.step];
    for (long sti = 0; sti < 8; ++sti)
    {
      out_ref[sti * 64 + hipThreadIdx_x] = buf12[sti];
    }
  }
}
# 156 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/helmholtz4/intermediate_gen/helmholtz4.cu" 2

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
# 1 "VSTile-helmholtz43.py-HIP-8x8x64" 1
{
  bElem buf0[64];
  bElem buf1[64];
  bElem buf2[64];
  bElem buf3[64];
  bElem buf4[64];
  bElem buf5[64];
  bElem buf6[64];
  bElem buf7[64];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf0[0 + rel] = 0;
                buf1[0 + rel] = 0;
                buf2[0 + rel] = 0;
                buf3[0 + rel] = 0;
                buf4[0 + rel] = 0;
                buf5[0 + rel] = 0;
                buf6[0 + rel] = 0;
                buf7[0 + rel] = 0;
              }
              _cg_rel1 += 1;
            }
          }
          _cg_rel2 += 8;
        }
      }
    }
    {
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 7; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf2[0 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + -2);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf2[7] -= x(i + hipThreadIdx_x, j + 7, k + -2);
      }
      {
        buf5[0] -= x(i + hipThreadIdx_x + -1, j, k + -1);
      }
      {
        buf5[1] -= x(i + hipThreadIdx_x + -1, j + 1, k + -1);
        buf7[0] -= x(i + hipThreadIdx_x, j + -1, k + -1);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf2[0 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf2[8 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf3[0 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf5[0 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf5[2 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + -1);
            buf7[0 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf7[1 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf2[6] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + -1);
        buf2[14] -= x(i + hipThreadIdx_x, j + 6, k + -1);
        buf3[6] += x(i + hipThreadIdx_x, j + 6, k + -1);
        buf5[6] += x(i + hipThreadIdx_x, j + 6, k + -1);
        buf7[6] += x(i + hipThreadIdx_x, j + 6, k + -1);
        buf7[7] -= x(i + hipThreadIdx_x, j + 6, k + -1);
      }
      {
        buf2[7] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + -1);
        buf2[15] -= x(i + hipThreadIdx_x, j + 7, k + -1);
        buf3[7] += x(i + hipThreadIdx_x, j + 7, k + -1);
        buf5[7] += x(i + hipThreadIdx_x, j + 7, k + -1);
        buf7[7] += x(i + hipThreadIdx_x, j + 7, k + -1);
      }
      {
        buf4[0] -= x(i + hipThreadIdx_x + -1, j + -1, k);
        buf6[0] -= x(i + hipThreadIdx_x + -1, j + -1, k);
      }
      {
        buf4[1] -= x(i + hipThreadIdx_x + -1, j, k);
        buf6[1] -= x(i + hipThreadIdx_x + -1, j, k);
        buf6[0] += x(i + hipThreadIdx_x + -1, j, k);
        buf5[8] -= x(i + hipThreadIdx_x + -1, j, k);
      }
      {
        buf0[0] += 15.0 * x(i + hipThreadIdx_x, j + -1, k);
        buf0[0] -= x(i + hipThreadIdx_x, j + -1, k);
        buf1[0] += x(i + hipThreadIdx_x, j + -1, k);
        buf4[0] += x(i + hipThreadIdx_x + -1, j + 1, k);
        buf4[0] += x(i + hipThreadIdx_x, j + -1, k);
        buf4[2] -= x(i + hipThreadIdx_x + -1, j + 1, k);
        buf6[0] += x(i + hipThreadIdx_x + 1, j + -1, k);
        buf6[2] -= x(i + hipThreadIdx_x + -1, j + 1, k);
        buf6[1] += x(i + hipThreadIdx_x + -1, j + 1, k);
        buf5[9] -= x(i + hipThreadIdx_x + -1, j + 1, k);
        buf7[8] -= x(i + hipThreadIdx_x, j + -1, k);
      }
      {
        buf0[1] += 15.0 * x(i + hipThreadIdx_x, j, k);
        buf0[1] -= x(i + hipThreadIdx_x, j, k);
        buf0[0] -= 15.0 * x(i + hipThreadIdx_x, j, k);
        buf1[1] += x(i + hipThreadIdx_x, j, k);
        buf1[0] -= 15.0 * x(i + hipThreadIdx_x, j, k);
        buf4[1] += x(i + hipThreadIdx_x + -1, j + 2, k);
        buf4[1] += x(i + hipThreadIdx_x, j, k);
        buf4[3] -= x(i + hipThreadIdx_x + -1, j + 2, k);
        buf6[1] += x(i + hipThreadIdx_x + 1, j, k);
        buf6[0] -= x(i + hipThreadIdx_x + 1, j, k);
        buf6[3] -= x(i + hipThreadIdx_x + -1, j + 2, k);
        buf6[2] += x(i + hipThreadIdx_x + -1, j + 2, k);
        buf2[0] -= 15.0 * x(i + hipThreadIdx_x, j, k);
        buf2[8] += 15.0 * x(i + hipThreadIdx_x, j, k);
        buf2[16] -= x(i + hipThreadIdx_x, j, k);
        buf3[0] -= 15.0 * x(i + hipThreadIdx_x, j, k);
        buf3[8] += x(i + hipThreadIdx_x, j, k);
        buf5[8] += x(i + hipThreadIdx_x, j, k);
        buf5[10] -= x(i + hipThreadIdx_x + -1, j + 2, k);
        buf7[8] += x(i + hipThreadIdx_x, j, k);
        buf7[9] -= x(i + hipThreadIdx_x, j, k);
      }
      {
        buf0[2] += 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf0[2] -= x(i + hipThreadIdx_x, j + 1, k);
        buf0[1] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf0[0] += x(i + hipThreadIdx_x, j + 1, k);
        buf1[2] += x(i + hipThreadIdx_x, j + 1, k);
        buf1[1] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf1[0] += 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf4[2] += x(i + hipThreadIdx_x + -1, j + 3, k);
        buf4[2] += x(i + hipThreadIdx_x, j + 1, k);
        buf4[0] -= x(i + hipThreadIdx_x, j + 1, k);
        buf4[4] -= x(i + hipThreadIdx_x + -1, j + 3, k);
        buf6[2] += x(i + hipThreadIdx_x + 1, j + 1, k);
        buf6[1] -= x(i + hipThreadIdx_x + 1, j + 1, k);
        buf6[4] -= x(i + hipThreadIdx_x + -1, j + 3, k);
        buf6[3] += x(i + hipThreadIdx_x + -1, j + 3, k);
        buf2[1] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf2[9] += 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf2[17] -= x(i + hipThreadIdx_x, j + 1, k);
        buf3[1] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf3[9] += x(i + hipThreadIdx_x, j + 1, k);
        buf5[9] += x(i + hipThreadIdx_x, j + 1, k);
        buf5[11] -= x(i + hipThreadIdx_x + -1, j + 3, k);
        buf7[9] += x(i + hipThreadIdx_x, j + 1, k);
        buf7[10] -= x(i + hipThreadIdx_x, j + 1, k);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 3; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[3 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf0[3 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf0[2 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf0[1 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf1[3 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf1[2 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf1[1 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf1[0 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf4[3 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k);
            buf4[3 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf4[1 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf4[5 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k);
            buf6[3 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k);
            buf6[2 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k);
            buf6[5 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k);
            buf6[4 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k);
            buf2[2 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf2[10 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf2[18 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf3[2 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf3[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf5[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf5[12 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k);
            buf7[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf7[11 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[6] += 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf0[6] -= x(i + hipThreadIdx_x, j + 5, k);
        buf0[5] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf0[4] += x(i + hipThreadIdx_x, j + 5, k);
        buf1[6] += x(i + hipThreadIdx_x, j + 5, k);
        buf1[5] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf1[4] += 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf1[3] -= x(i + hipThreadIdx_x, j + 5, k);
        buf4[6] += x(i + hipThreadIdx_x + -1, j + 7, k);
        buf4[6] += x(i + hipThreadIdx_x, j + 5, k);
        buf4[4] -= x(i + hipThreadIdx_x, j + 5, k);
        buf6[6] += x(i + hipThreadIdx_x + 1, j + 5, k);
        buf6[5] -= x(i + hipThreadIdx_x + 1, j + 5, k);
        buf6[7] += x(i + hipThreadIdx_x + -1, j + 7, k);
        buf2[5] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf2[13] += 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf2[21] -= x(i + hipThreadIdx_x, j + 5, k);
        buf3[5] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf3[13] += x(i + hipThreadIdx_x, j + 5, k);
        buf5[13] += x(i + hipThreadIdx_x, j + 5, k);
        buf5[15] -= x(i + hipThreadIdx_x + -1, j + 7, k);
        buf7[13] += x(i + hipThreadIdx_x, j + 5, k);
        buf7[14] -= x(i + hipThreadIdx_x, j + 5, k);
      }
      {
        buf0[7] += 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf0[7] -= x(i + hipThreadIdx_x, j + 6, k);
        buf0[6] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf0[5] += x(i + hipThreadIdx_x, j + 6, k);
        buf1[7] += x(i + hipThreadIdx_x, j + 6, k);
        buf1[6] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf1[5] += 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf1[4] -= x(i + hipThreadIdx_x, j + 6, k);
        buf4[7] += x(i + hipThreadIdx_x + -1, j + 8, k);
        buf4[7] += x(i + hipThreadIdx_x, j + 6, k);
        buf4[5] -= x(i + hipThreadIdx_x, j + 6, k);
        buf6[7] += x(i + hipThreadIdx_x + 1, j + 6, k);
        buf6[6] -= x(i + hipThreadIdx_x + 1, j + 6, k);
        buf2[6] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf2[14] += 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf2[22] -= x(i + hipThreadIdx_x, j + 6, k);
        buf3[6] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf3[14] += x(i + hipThreadIdx_x, j + 6, k);
        buf5[14] += x(i + hipThreadIdx_x, j + 6, k);
        buf7[14] += x(i + hipThreadIdx_x, j + 6, k);
        buf7[15] -= x(i + hipThreadIdx_x, j + 6, k);
      }
      {
        buf0[7] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k);
        buf0[6] += x(i + hipThreadIdx_x, j + 7, k);
        buf1[7] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k);
        buf1[6] += 15.0 * x(i + hipThreadIdx_x, j + 7, k);
        buf1[5] -= x(i + hipThreadIdx_x, j + 7, k);
        buf2[7] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k);
        buf2[15] += 15.0 * x(i + hipThreadIdx_x, j + 7, k);
        buf2[23] -= x(i + hipThreadIdx_x, j + 7, k);
        buf3[7] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k);
        buf3[15] += x(i + hipThreadIdx_x, j + 7, k);
        buf6[7] -= x(i + hipThreadIdx_x + 1, j + 7, k);
        buf4[6] -= x(i + hipThreadIdx_x, j + 7, k);
        buf5[15] += x(i + hipThreadIdx_x, j + 7, k);
        buf7[15] += x(i + hipThreadIdx_x, j + 7, k);
      }
      {
        buf0[7] += x(i + hipThreadIdx_x, j + 8, k);
        buf1[7] += 15.0 * x(i + hipThreadIdx_x, j + 8, k);
        buf1[6] -= x(i + hipThreadIdx_x, j + 8, k);
        buf4[7] -= x(i + hipThreadIdx_x, j + 8, k);
      }
      {
        buf1[7] -= x(i + hipThreadIdx_x, j + 9, k);
      }
      {
        buf4[8] -= x(i + hipThreadIdx_x + -1, j + -1, k + 1);
        buf6[8] -= x(i + hipThreadIdx_x + -1, j + -1, k + 1);
      }
      {
        buf4[9] -= x(i + hipThreadIdx_x + -1, j, k + 1);
        buf6[9] -= x(i + hipThreadIdx_x + -1, j, k + 1);
        buf6[8] += x(i + hipThreadIdx_x + -1, j, k + 1);
        buf5[0] += x(i + hipThreadIdx_x + -1, j, k + 1);
        buf5[16] -= x(i + hipThreadIdx_x + -1, j, k + 1);
      }
      {
        buf0[8] += 15.0 * x(i + hipThreadIdx_x, j + -1, k + 1);
        buf0[8] -= x(i + hipThreadIdx_x, j + -1, k + 1);
        buf1[8] += x(i + hipThreadIdx_x, j + -1, k + 1);
        buf4[8] += x(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf4[8] += x(i + hipThreadIdx_x, j + -1, k + 1);
        buf4[10] -= x(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf6[8] += x(i + hipThreadIdx_x + 1, j + -1, k + 1);
        buf6[10] -= x(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf6[9] += x(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf5[1] += x(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf5[17] -= x(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf7[0] += x(i + hipThreadIdx_x, j + -1, k + 1);
        buf7[16] -= x(i + hipThreadIdx_x, j + -1, k + 1);
      }
      {
        buf0[9] += 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf0[9] -= x(i + hipThreadIdx_x, j, k + 1);
        buf0[8] -= 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf1[9] += x(i + hipThreadIdx_x, j, k + 1);
        buf1[8] -= 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf4[9] += x(i + hipThreadIdx_x + -1, j + 2, k + 1);
        buf4[9] += x(i + hipThreadIdx_x, j, k + 1);
        buf4[11] -= x(i + hipThreadIdx_x + -1, j + 2, k + 1);
        buf6[9] += x(i + hipThreadIdx_x + 1, j, k + 1);
        buf6[8] -= x(i + hipThreadIdx_x + 1, j, k + 1);
        buf6[11] -= x(i + hipThreadIdx_x + -1, j + 2, k + 1);
        buf6[10] += x(i + hipThreadIdx_x + -1, j + 2, k + 1);
        buf2[8] -= 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf2[16] += 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf2[24] -= x(i + hipThreadIdx_x, j, k + 1);
        buf2[0] += x(i + hipThreadIdx_x, j, k + 1);
        buf3[8] -= 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf3[16] += x(i + hipThreadIdx_x, j, k + 1);
        buf3[0] += 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf5[16] += x(i + hipThreadIdx_x, j, k + 1);
        buf5[0] -= x(i + hipThreadIdx_x, j, k + 1);
        buf5[2] += x(i + hipThreadIdx_x + -1, j + 2, k + 1);
        buf5[18] -= x(i + hipThreadIdx_x + -1, j + 2, k + 1);
        buf7[16] += x(i + hipThreadIdx_x, j, k + 1);
        buf7[0] -= x(i + hipThreadIdx_x, j, k + 1);
        buf7[1] += x(i + hipThreadIdx_x, j, k + 1);
        buf7[17] -= x(i + hipThreadIdx_x, j, k + 1);
      }
      {
        buf0[10] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[10] -= x(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[9] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[8] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf1[10] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf1[9] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf1[8] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf4[10] += x(i + hipThreadIdx_x + -1, j + 3, k + 1);
        buf4[10] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf4[8] -= x(i + hipThreadIdx_x, j + 1, k + 1);
        buf4[12] -= x(i + hipThreadIdx_x + -1, j + 3, k + 1);
        buf6[10] += x(i + hipThreadIdx_x + 1, j + 1, k + 1);
        buf6[9] -= x(i + hipThreadIdx_x + 1, j + 1, k + 1);
        buf6[12] -= x(i + hipThreadIdx_x + -1, j + 3, k + 1);
        buf6[11] += x(i + hipThreadIdx_x + -1, j + 3, k + 1);
        buf2[9] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf2[17] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf2[25] -= x(i + hipThreadIdx_x, j + 1, k + 1);
        buf2[1] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf3[9] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf3[17] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf3[1] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf5[17] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf5[1] -= x(i + hipThreadIdx_x, j + 1, k + 1);
        buf5[3] += x(i + hipThreadIdx_x + -1, j + 3, k + 1);
        buf5[19] -= x(i + hipThreadIdx_x + -1, j + 3, k + 1);
        buf7[17] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf7[1] -= x(i + hipThreadIdx_x, j + 1, k + 1);
        buf7[2] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf7[18] -= x(i + hipThreadIdx_x, j + 1, k + 1);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 3; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[11 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf0[11 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf0[10 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf0[9 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf1[11 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf1[10 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf1[9 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf1[8 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf4[11 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 1);
            buf4[11 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf4[9 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf4[13 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 1);
            buf6[11 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1);
            buf6[10 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1);
            buf6[13 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 1);
            buf6[12 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 1);
            buf2[10 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf2[18 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf2[26 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf2[2 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf3[10 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf3[18 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf3[2 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf5[18 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf5[2 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf5[4 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 1);
            buf5[20 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 1);
            buf7[18 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf7[2 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf7[3 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf7[19 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[14] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[14] -= x(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[13] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[12] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf1[14] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf1[13] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf1[12] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf1[11] -= x(i + hipThreadIdx_x, j + 5, k + 1);
        buf4[14] += x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf4[14] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf4[12] -= x(i + hipThreadIdx_x, j + 5, k + 1);
        buf6[14] += x(i + hipThreadIdx_x + 1, j + 5, k + 1);
        buf6[13] -= x(i + hipThreadIdx_x + 1, j + 5, k + 1);
        buf6[15] += x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf2[13] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf2[21] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf2[29] -= x(i + hipThreadIdx_x, j + 5, k + 1);
        buf2[5] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf3[13] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf3[21] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf3[5] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf5[21] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf5[5] -= x(i + hipThreadIdx_x, j + 5, k + 1);
        buf5[7] += x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf5[23] -= x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf7[21] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf7[5] -= x(i + hipThreadIdx_x, j + 5, k + 1);
        buf7[6] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf7[22] -= x(i + hipThreadIdx_x, j + 5, k + 1);
      }
      {
        buf0[15] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[15] -= x(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[14] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[13] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf1[15] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf1[14] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf1[13] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf1[12] -= x(i + hipThreadIdx_x, j + 6, k + 1);
        buf4[15] += x(i + hipThreadIdx_x + -1, j + 8, k + 1);
        buf4[15] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf4[13] -= x(i + hipThreadIdx_x, j + 6, k + 1);
        buf6[15] += x(i + hipThreadIdx_x + 1, j + 6, k + 1);
        buf6[14] -= x(i + hipThreadIdx_x + 1, j + 6, k + 1);
        buf2[14] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf2[22] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf2[30] -= x(i + hipThreadIdx_x, j + 6, k + 1);
        buf2[6] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf3[14] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf3[22] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf3[6] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf5[22] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf5[6] -= x(i + hipThreadIdx_x, j + 6, k + 1);
        buf7[22] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf7[6] -= x(i + hipThreadIdx_x, j + 6, k + 1);
        buf7[7] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf7[23] -= x(i + hipThreadIdx_x, j + 6, k + 1);
      }
      {
        buf0[15] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[14] += x(i + hipThreadIdx_x, j + 7, k + 1);
        buf1[15] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf1[14] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf1[13] -= x(i + hipThreadIdx_x, j + 7, k + 1);
        buf2[15] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf2[23] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf2[31] -= x(i + hipThreadIdx_x, j + 7, k + 1);
        buf2[7] += x(i + hipThreadIdx_x, j + 7, k + 1);
        buf3[15] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf3[23] += x(i + hipThreadIdx_x, j + 7, k + 1);
        buf3[7] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf6[15] -= x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf4[14] -= x(i + hipThreadIdx_x, j + 7, k + 1);
        buf5[23] += x(i + hipThreadIdx_x, j + 7, k + 1);
        buf5[7] -= x(i + hipThreadIdx_x, j + 7, k + 1);
        buf7[23] += x(i + hipThreadIdx_x, j + 7, k + 1);
        buf7[7] -= x(i + hipThreadIdx_x, j + 7, k + 1);
      }
      {
        buf0[15] += x(i + hipThreadIdx_x, j + 8, k + 1);
        buf1[15] += 15.0 * x(i + hipThreadIdx_x, j + 8, k + 1);
        buf1[14] -= x(i + hipThreadIdx_x, j + 8, k + 1);
        buf4[15] -= x(i + hipThreadIdx_x, j + 8, k + 1);
      }
      {
        buf1[15] -= x(i + hipThreadIdx_x, j + 9, k + 1);
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 4; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            buf4[16 + rel] -= x(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 2);
            buf6[16 + rel] -= x(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 2);
          }
          {
            buf4[17 + rel] -= x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
            buf6[17 + rel] -= x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
            buf6[16 + rel] += x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
            buf5[8 + rel] += x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
            buf5[24 + rel] -= x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
          }
          {
            buf0[16 + rel] += 15.0 * x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2);
            buf0[16 + rel] -= x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2);
            buf1[16 + rel] += x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2);
            buf4[16 + rel] += x(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2);
            buf4[16 + rel] += x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2);
            buf4[18 + rel] -= x(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2);
            buf6[16 + rel] += x(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 2);
            buf6[18 + rel] -= x(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2);
            buf6[17 + rel] += x(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2);
            buf5[9 + rel] += x(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2);
            buf5[25 + rel] -= x(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2);
            buf7[8 + rel] += x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2);
            buf7[24 + rel] -= x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2);
          }
          {
            buf0[17 + rel] += 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf0[17 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf0[16 + rel] -= 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf1[17 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf1[16 + rel] -= 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf4[17 + rel] += x(i + hipThreadIdx_x + -1, j + 2, k + _cg_idx2 + 2);
            buf4[17 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf4[19 + rel] -= x(i + hipThreadIdx_x + -1, j + 2, k + _cg_idx2 + 2);
            buf6[17 + rel] += x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf6[16 + rel] -= x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf6[19 + rel] -= x(i + hipThreadIdx_x + -1, j + 2, k + _cg_idx2 + 2);
            buf6[18 + rel] += x(i + hipThreadIdx_x + -1, j + 2, k + _cg_idx2 + 2);
            buf2[16 + rel] -= 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf2[24 + rel] += 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf2[32 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf2[8 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf3[16 + rel] -= 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf3[24 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf3[8 + rel] += 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf3[0 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf5[24 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf5[8 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf5[10 + rel] += x(i + hipThreadIdx_x + -1, j + 2, k + _cg_idx2 + 2);
            buf5[26 + rel] -= x(i + hipThreadIdx_x + -1, j + 2, k + _cg_idx2 + 2);
            buf7[24 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf7[8 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf7[9 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf7[25 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
          }
          {
            buf0[18 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf0[18 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf0[17 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf0[16 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf1[18 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf1[17 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf1[16 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf4[18 + rel] += x(i + hipThreadIdx_x + -1, j + 3, k + _cg_idx2 + 2);
            buf4[18 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf4[16 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf4[20 + rel] -= x(i + hipThreadIdx_x + -1, j + 3, k + _cg_idx2 + 2);
            buf6[18 + rel] += x(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2);
            buf6[17 + rel] -= x(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2);
            buf6[20 + rel] -= x(i + hipThreadIdx_x + -1, j + 3, k + _cg_idx2 + 2);
            buf6[19 + rel] += x(i + hipThreadIdx_x + -1, j + 3, k + _cg_idx2 + 2);
            buf2[17 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf2[25 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf2[33 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf2[9 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf3[17 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf3[25 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf3[9 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf3[1 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf5[25 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf5[9 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf5[11 + rel] += x(i + hipThreadIdx_x + -1, j + 3, k + _cg_idx2 + 2);
            buf5[27 + rel] -= x(i + hipThreadIdx_x + -1, j + 3, k + _cg_idx2 + 2);
            buf7[25 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf7[9 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf7[10 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf7[26 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
          }
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 3; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf0[19 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[19 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[18 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[17 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf1[19 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf1[18 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf1[17 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf1[16 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf4[19 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + _cg_idx2 + 2);
                buf4[19 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf4[17 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf4[21 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + _cg_idx2 + 2);
                buf6[19 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf6[18 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf6[21 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + _cg_idx2 + 2);
                buf6[20 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + _cg_idx2 + 2);
                buf2[18 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf2[26 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf2[34 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf2[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf3[18 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf3[26 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf3[10 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf3[2 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf5[26 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf5[10 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf5[12 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + _cg_idx2 + 2);
                buf5[28 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + _cg_idx2 + 2);
                buf7[26 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf7[10 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf7[11 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf7[27 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
              }
              _cg_rel1 += 1;
            }
          }
          {
            buf0[22 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf0[22 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf0[21 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf0[20 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf1[22 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf1[21 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf1[20 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf1[19 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf4[22 + rel] += x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf4[22 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf4[20 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf6[22 + rel] += x(i + hipThreadIdx_x + 1, j + 5, k + _cg_idx2 + 2);
            buf6[21 + rel] -= x(i + hipThreadIdx_x + 1, j + 5, k + _cg_idx2 + 2);
            buf6[23 + rel] += x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf2[21 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf2[29 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf2[37 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf2[13 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf3[21 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf3[29 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf3[13 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf3[5 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf5[29 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf5[13 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf5[15 + rel] += x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf5[31 + rel] -= x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf7[29 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf7[13 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf7[14 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf7[30 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
          }
          {
            buf0[23 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf0[23 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf0[22 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf0[21 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf1[23 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf1[22 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf1[21 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf1[20 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf4[23 + rel] += x(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 2);
            buf4[23 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf4[21 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf6[23 + rel] += x(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2);
            buf6[22 + rel] -= x(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2);
            buf2[22 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf2[30 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf2[38 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf2[14 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf3[22 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf3[30 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf3[14 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf3[6 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf5[30 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf5[14 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf7[30 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf7[14 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf7[15 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf7[31 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
          }
          {
            buf0[23 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf0[22 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf1[23 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf1[22 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf1[21 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf2[23 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf2[31 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf2[39 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf2[15 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf3[23 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf3[31 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf3[15 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf3[7 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf6[23 + rel] -= x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf4[22 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf5[31 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf5[15 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf7[31 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf7[15 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
          }
          {
            buf0[23 + rel] += x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2);
            buf1[23 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2);
            buf1[22 + rel] -= x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2);
            buf4[23 + rel] -= x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2);
          }
          {
            buf1[23 + rel] -= x(i + hipThreadIdx_x, j + 9, k + _cg_idx2 + 2);
          }
          _cg_rel2 += 8;
        }
      }
      {
        buf4[48] -= x(i + hipThreadIdx_x + -1, j + -1, k + 6);
        buf6[48] -= x(i + hipThreadIdx_x + -1, j + -1, k + 6);
      }
      {
        buf4[49] -= x(i + hipThreadIdx_x + -1, j, k + 6);
        buf6[49] -= x(i + hipThreadIdx_x + -1, j, k + 6);
        buf6[48] += x(i + hipThreadIdx_x + -1, j, k + 6);
        buf5[40] += x(i + hipThreadIdx_x + -1, j, k + 6);
        buf5[56] -= x(i + hipThreadIdx_x + -1, j, k + 6);
      }
      {
        buf0[48] += 15.0 * x(i + hipThreadIdx_x, j + -1, k + 6);
        buf0[48] -= x(i + hipThreadIdx_x, j + -1, k + 6);
        buf1[48] += x(i + hipThreadIdx_x, j + -1, k + 6);
        buf4[48] += x(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf4[48] += x(i + hipThreadIdx_x, j + -1, k + 6);
        buf4[50] -= x(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf6[48] += x(i + hipThreadIdx_x + 1, j + -1, k + 6);
        buf6[50] -= x(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf6[49] += x(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf5[41] += x(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf5[57] -= x(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf7[40] += x(i + hipThreadIdx_x, j + -1, k + 6);
        buf7[56] -= x(i + hipThreadIdx_x, j + -1, k + 6);
      }
      {
        buf0[49] += 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf0[49] -= x(i + hipThreadIdx_x, j, k + 6);
        buf0[48] -= 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf1[49] += x(i + hipThreadIdx_x, j, k + 6);
        buf1[48] -= 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf4[49] += x(i + hipThreadIdx_x + -1, j + 2, k + 6);
        buf4[49] += x(i + hipThreadIdx_x, j, k + 6);
        buf4[51] -= x(i + hipThreadIdx_x + -1, j + 2, k + 6);
        buf6[49] += x(i + hipThreadIdx_x + 1, j, k + 6);
        buf6[48] -= x(i + hipThreadIdx_x + 1, j, k + 6);
        buf6[51] -= x(i + hipThreadIdx_x + -1, j + 2, k + 6);
        buf6[50] += x(i + hipThreadIdx_x + -1, j + 2, k + 6);
        buf2[48] -= 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf2[56] += 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf2[40] += x(i + hipThreadIdx_x, j, k + 6);
        buf3[48] -= 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf3[56] += x(i + hipThreadIdx_x, j, k + 6);
        buf3[40] += 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf3[32] -= x(i + hipThreadIdx_x, j, k + 6);
        buf5[56] += x(i + hipThreadIdx_x, j, k + 6);
        buf5[40] -= x(i + hipThreadIdx_x, j, k + 6);
        buf5[42] += x(i + hipThreadIdx_x + -1, j + 2, k + 6);
        buf5[58] -= x(i + hipThreadIdx_x + -1, j + 2, k + 6);
        buf7[56] += x(i + hipThreadIdx_x, j, k + 6);
        buf7[40] -= x(i + hipThreadIdx_x, j, k + 6);
        buf7[41] += x(i + hipThreadIdx_x, j, k + 6);
        buf7[57] -= x(i + hipThreadIdx_x, j, k + 6);
      }
      {
        buf0[50] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[50] -= x(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[49] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[48] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf1[50] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf1[49] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf1[48] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf4[50] += x(i + hipThreadIdx_x + -1, j + 3, k + 6);
        buf4[50] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf4[48] -= x(i + hipThreadIdx_x, j + 1, k + 6);
        buf4[52] -= x(i + hipThreadIdx_x + -1, j + 3, k + 6);
        buf6[50] += x(i + hipThreadIdx_x + 1, j + 1, k + 6);
        buf6[49] -= x(i + hipThreadIdx_x + 1, j + 1, k + 6);
        buf6[52] -= x(i + hipThreadIdx_x + -1, j + 3, k + 6);
        buf6[51] += x(i + hipThreadIdx_x + -1, j + 3, k + 6);
        buf2[49] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf2[57] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf2[41] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf3[49] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf3[57] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf3[41] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf3[33] -= x(i + hipThreadIdx_x, j + 1, k + 6);
        buf5[57] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf5[41] -= x(i + hipThreadIdx_x, j + 1, k + 6);
        buf5[43] += x(i + hipThreadIdx_x + -1, j + 3, k + 6);
        buf5[59] -= x(i + hipThreadIdx_x + -1, j + 3, k + 6);
        buf7[57] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf7[41] -= x(i + hipThreadIdx_x, j + 1, k + 6);
        buf7[42] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf7[58] -= x(i + hipThreadIdx_x, j + 1, k + 6);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 3; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[51 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf0[51 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf0[50 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf0[49 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf1[51 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf1[50 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf1[49 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf1[48 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf4[51 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 6);
            buf4[51 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf4[49 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf4[53 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 6);
            buf6[51 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6);
            buf6[50 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6);
            buf6[53 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 6);
            buf6[52 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 6);
            buf2[50 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf2[58 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf2[42 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf3[50 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf3[58 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf3[42 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf3[34 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf5[58 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf5[42 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf5[44 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 6);
            buf5[60 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 6);
            buf7[58 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf7[42 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf7[43 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf7[59 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[54] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[54] -= x(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[53] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[52] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf1[54] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf1[53] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf1[52] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf1[51] -= x(i + hipThreadIdx_x, j + 5, k + 6);
        buf4[54] += x(i + hipThreadIdx_x + -1, j + 7, k + 6);
        buf4[54] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf4[52] -= x(i + hipThreadIdx_x, j + 5, k + 6);
        buf6[54] += x(i + hipThreadIdx_x + 1, j + 5, k + 6);
        buf6[53] -= x(i + hipThreadIdx_x + 1, j + 5, k + 6);
        buf6[55] += x(i + hipThreadIdx_x + -1, j + 7, k + 6);
        buf2[53] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf2[61] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf2[45] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf3[53] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf3[61] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf3[45] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf3[37] -= x(i + hipThreadIdx_x, j + 5, k + 6);
        buf5[61] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf5[45] -= x(i + hipThreadIdx_x, j + 5, k + 6);
        buf5[47] += x(i + hipThreadIdx_x + -1, j + 7, k + 6);
        buf5[63] -= x(i + hipThreadIdx_x + -1, j + 7, k + 6);
        buf7[61] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf7[45] -= x(i + hipThreadIdx_x, j + 5, k + 6);
        buf7[46] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf7[62] -= x(i + hipThreadIdx_x, j + 5, k + 6);
      }
      {
        buf0[55] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[55] -= x(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[54] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[53] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf1[55] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf1[54] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf1[53] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf1[52] -= x(i + hipThreadIdx_x, j + 6, k + 6);
        buf4[55] += x(i + hipThreadIdx_x + -1, j + 8, k + 6);
        buf4[55] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf4[53] -= x(i + hipThreadIdx_x, j + 6, k + 6);
        buf6[55] += x(i + hipThreadIdx_x + 1, j + 6, k + 6);
        buf6[54] -= x(i + hipThreadIdx_x + 1, j + 6, k + 6);
        buf2[54] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf2[62] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf2[46] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf3[54] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf3[62] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf3[46] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf3[38] -= x(i + hipThreadIdx_x, j + 6, k + 6);
        buf5[62] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf5[46] -= x(i + hipThreadIdx_x, j + 6, k + 6);
        buf7[62] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf7[46] -= x(i + hipThreadIdx_x, j + 6, k + 6);
        buf7[47] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf7[63] -= x(i + hipThreadIdx_x, j + 6, k + 6);
      }
      {
        buf0[55] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[54] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf1[55] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf1[54] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf1[53] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf2[55] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf2[63] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf2[47] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf3[55] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf3[63] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf3[47] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf3[39] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf6[55] -= x(i + hipThreadIdx_x + 1, j + 7, k + 6);
        buf4[54] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf5[63] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf5[47] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf7[63] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf7[47] -= x(i + hipThreadIdx_x, j + 7, k + 6);
      }
      {
        buf0[55] += x(i + hipThreadIdx_x, j + 8, k + 6);
        buf1[55] += 15.0 * x(i + hipThreadIdx_x, j + 8, k + 6);
        buf1[54] -= x(i + hipThreadIdx_x, j + 8, k + 6);
        buf4[55] -= x(i + hipThreadIdx_x, j + 8, k + 6);
      }
      {
        buf1[55] -= x(i + hipThreadIdx_x, j + 9, k + 6);
      }
      {
        buf4[56] -= x(i + hipThreadIdx_x + -1, j + -1, k + 7);
        buf6[56] -= x(i + hipThreadIdx_x + -1, j + -1, k + 7);
      }
      {
        buf4[57] -= x(i + hipThreadIdx_x + -1, j, k + 7);
        buf6[57] -= x(i + hipThreadIdx_x + -1, j, k + 7);
        buf6[56] += x(i + hipThreadIdx_x + -1, j, k + 7);
        buf5[48] += x(i + hipThreadIdx_x + -1, j, k + 7);
      }
      {
        buf0[56] += 15.0 * x(i + hipThreadIdx_x, j + -1, k + 7);
        buf0[56] -= x(i + hipThreadIdx_x, j + -1, k + 7);
        buf1[56] += x(i + hipThreadIdx_x, j + -1, k + 7);
        buf4[56] += x(i + hipThreadIdx_x + -1, j + 1, k + 7);
        buf4[56] += x(i + hipThreadIdx_x, j + -1, k + 7);
        buf4[58] -= x(i + hipThreadIdx_x + -1, j + 1, k + 7);
        buf6[56] += x(i + hipThreadIdx_x + 1, j + -1, k + 7);
        buf6[58] -= x(i + hipThreadIdx_x + -1, j + 1, k + 7);
        buf6[57] += x(i + hipThreadIdx_x + -1, j + 1, k + 7);
        buf5[49] += x(i + hipThreadIdx_x + -1, j + 1, k + 7);
        buf7[48] += x(i + hipThreadIdx_x, j + -1, k + 7);
      }
      {
        buf0[57] += 15.0 * x(i + hipThreadIdx_x, j, k + 7);
        buf0[57] -= x(i + hipThreadIdx_x, j, k + 7);
        buf0[56] -= 15.0 * x(i + hipThreadIdx_x, j, k + 7);
        buf1[57] += x(i + hipThreadIdx_x, j, k + 7);
        buf1[56] -= 15.0 * x(i + hipThreadIdx_x, j, k + 7);
        buf4[57] += x(i + hipThreadIdx_x + -1, j + 2, k + 7);
        buf4[57] += x(i + hipThreadIdx_x, j, k + 7);
        buf4[59] -= x(i + hipThreadIdx_x + -1, j + 2, k + 7);
        buf6[57] += x(i + hipThreadIdx_x + 1, j, k + 7);
        buf6[56] -= x(i + hipThreadIdx_x + 1, j, k + 7);
        buf6[59] -= x(i + hipThreadIdx_x + -1, j + 2, k + 7);
        buf6[58] += x(i + hipThreadIdx_x + -1, j + 2, k + 7);
        buf2[56] -= 15.0 * x(i + hipThreadIdx_x, j, k + 7);
        buf2[48] += x(i + hipThreadIdx_x, j, k + 7);
        buf3[56] -= 15.0 * x(i + hipThreadIdx_x, j, k + 7);
        buf3[48] += 15.0 * x(i + hipThreadIdx_x, j, k + 7);
        buf3[40] -= x(i + hipThreadIdx_x, j, k + 7);
        buf5[48] -= x(i + hipThreadIdx_x, j, k + 7);
        buf5[50] += x(i + hipThreadIdx_x + -1, j + 2, k + 7);
        buf7[48] -= x(i + hipThreadIdx_x, j, k + 7);
        buf7[49] += x(i + hipThreadIdx_x, j, k + 7);
      }
      {
        buf0[58] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[58] -= x(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[57] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[56] += x(i + hipThreadIdx_x, j + 1, k + 7);
        buf1[58] += x(i + hipThreadIdx_x, j + 1, k + 7);
        buf1[57] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf1[56] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf4[58] += x(i + hipThreadIdx_x + -1, j + 3, k + 7);
        buf4[58] += x(i + hipThreadIdx_x, j + 1, k + 7);
        buf4[56] -= x(i + hipThreadIdx_x, j + 1, k + 7);
        buf4[60] -= x(i + hipThreadIdx_x + -1, j + 3, k + 7);
        buf6[58] += x(i + hipThreadIdx_x + 1, j + 1, k + 7);
        buf6[57] -= x(i + hipThreadIdx_x + 1, j + 1, k + 7);
        buf6[60] -= x(i + hipThreadIdx_x + -1, j + 3, k + 7);
        buf6[59] += x(i + hipThreadIdx_x + -1, j + 3, k + 7);
        buf2[57] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf2[49] += x(i + hipThreadIdx_x, j + 1, k + 7);
        buf3[57] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf3[49] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf3[41] -= x(i + hipThreadIdx_x, j + 1, k + 7);
        buf5[49] -= x(i + hipThreadIdx_x, j + 1, k + 7);
        buf5[51] += x(i + hipThreadIdx_x + -1, j + 3, k + 7);
        buf7[49] -= x(i + hipThreadIdx_x, j + 1, k + 7);
        buf7[50] += x(i + hipThreadIdx_x, j + 1, k + 7);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 3; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[59 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf0[59 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf0[58 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf0[57 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf1[59 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf1[58 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf1[57 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf1[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf4[59 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 7);
            buf4[59 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf4[57 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf4[61 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 7);
            buf6[59 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7);
            buf6[58 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7);
            buf6[61 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 7);
            buf6[60 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 7);
            buf2[58 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf2[50 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf3[58 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf3[50 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf3[42 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf5[50 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf5[52 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 7);
            buf7[50 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf7[51 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[62] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[62] -= x(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[61] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[60] += x(i + hipThreadIdx_x, j + 5, k + 7);
        buf1[62] += x(i + hipThreadIdx_x, j + 5, k + 7);
        buf1[61] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf1[60] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf1[59] -= x(i + hipThreadIdx_x, j + 5, k + 7);
        buf4[62] += x(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf4[62] += x(i + hipThreadIdx_x, j + 5, k + 7);
        buf4[60] -= x(i + hipThreadIdx_x, j + 5, k + 7);
        buf6[62] += x(i + hipThreadIdx_x + 1, j + 5, k + 7);
        buf6[61] -= x(i + hipThreadIdx_x + 1, j + 5, k + 7);
        buf6[63] += x(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf2[61] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf2[53] += x(i + hipThreadIdx_x, j + 5, k + 7);
        buf3[61] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf3[53] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf3[45] -= x(i + hipThreadIdx_x, j + 5, k + 7);
        buf5[53] -= x(i + hipThreadIdx_x, j + 5, k + 7);
        buf5[55] += x(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf7[53] -= x(i + hipThreadIdx_x, j + 5, k + 7);
        buf7[54] += x(i + hipThreadIdx_x, j + 5, k + 7);
      }
      {
        buf0[63] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[63] -= x(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[62] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[61] += x(i + hipThreadIdx_x, j + 6, k + 7);
        buf1[63] += x(i + hipThreadIdx_x, j + 6, k + 7);
        buf1[62] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf1[61] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf1[60] -= x(i + hipThreadIdx_x, j + 6, k + 7);
        buf4[63] += x(i + hipThreadIdx_x + -1, j + 8, k + 7);
        buf4[63] += x(i + hipThreadIdx_x, j + 6, k + 7);
        buf4[61] -= x(i + hipThreadIdx_x, j + 6, k + 7);
        buf6[63] += x(i + hipThreadIdx_x + 1, j + 6, k + 7);
        buf6[62] -= x(i + hipThreadIdx_x + 1, j + 6, k + 7);
        buf2[62] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf2[54] += x(i + hipThreadIdx_x, j + 6, k + 7);
        buf3[62] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf3[54] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf3[46] -= x(i + hipThreadIdx_x, j + 6, k + 7);
        buf5[54] -= x(i + hipThreadIdx_x, j + 6, k + 7);
        buf7[54] -= x(i + hipThreadIdx_x, j + 6, k + 7);
        buf7[55] += x(i + hipThreadIdx_x, j + 6, k + 7);
      }
      {
        buf0[63] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[62] += x(i + hipThreadIdx_x, j + 7, k + 7);
        buf1[63] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 7);
        buf1[62] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 7);
        buf1[61] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf2[63] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 7);
        buf2[55] += x(i + hipThreadIdx_x, j + 7, k + 7);
        buf3[63] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 7);
        buf3[55] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 7);
        buf3[47] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf6[63] -= x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf4[62] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf5[55] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf7[55] -= x(i + hipThreadIdx_x, j + 7, k + 7);
      }
      {
        buf0[63] += x(i + hipThreadIdx_x, j + 8, k + 7);
        buf1[63] += 15.0 * x(i + hipThreadIdx_x, j + 8, k + 7);
        buf1[62] -= x(i + hipThreadIdx_x, j + 8, k + 7);
        buf4[63] -= x(i + hipThreadIdx_x, j + 8, k + 7);
      }
      {
        buf1[63] -= x(i + hipThreadIdx_x, j + 9, k + 7);
      }
      {
        buf5[56] += x(i + hipThreadIdx_x + -1, j, k + 8);
      }
      {
        buf5[57] += x(i + hipThreadIdx_x + -1, j + 1, k + 8);
        buf7[56] += x(i + hipThreadIdx_x, j + -1, k + 8);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf2[56 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf3[56 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf3[48 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf5[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf5[58 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 8);
            buf7[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf7[57 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf2[62] += x(i + hipThreadIdx_x, j + 6, k + 8);
        buf3[62] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 8);
        buf3[54] -= x(i + hipThreadIdx_x, j + 6, k + 8);
        buf5[62] -= x(i + hipThreadIdx_x, j + 6, k + 8);
        buf7[62] -= x(i + hipThreadIdx_x, j + 6, k + 8);
        buf7[63] += x(i + hipThreadIdx_x, j + 6, k + 8);
      }
      {
        buf2[63] += x(i + hipThreadIdx_x, j + 7, k + 8);
        buf3[63] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 8);
        buf3[55] -= x(i + hipThreadIdx_x, j + 7, k + 8);
        buf5[63] -= x(i + hipThreadIdx_x, j + 7, k + 8);
        buf7[63] -= x(i + hipThreadIdx_x, j + 7, k + 8);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 7; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf3[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + 9);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf3[63] -= x(i + hipThreadIdx_x, j + 7, k + 9);
      }
    }
  }
  bElem buf8[64];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf8[0 + rel] = 0;
              }
              _cg_rel1 += 1;
            }
          }
          _cg_rel2 += 8;
        }
      }
    }
    {
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf8[0 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + -1);
            buf8[0 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + -1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 7; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf8[8 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2);
                buf8[8 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + _cg_idx2);
                buf8[0 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2);
                buf8[0 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + _cg_idx2);
              }
              _cg_rel1 += 1;
            }
          }
          _cg_rel2 += 8;
        }
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf8[56 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + 7);
            buf8[56 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + 7);
          }
          _cg_rel1 += 1;
        }
      }
    }
  }
  bElem buf9[64];
  bElem buf10[64];
  bElem buf11[64];
  bElem buf12[64];
  bElem buf13[64];
  bElem buf14[64];
  bElem buf15[64];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf9[0 + rel] = 0;
                buf10[0 + rel] = 0;
                buf11[0 + rel] = 0;
                buf12[0 + rel] = 0;
                buf13[0 + rel] = 0;
                buf14[0 + rel] = 0;
                buf15[0 + rel] = 0;
              }
              _cg_rel1 += 1;
            }
          }
          _cg_rel2 += 8;
        }
      }
    }
    {
      {
        buf11[0] -= x(i + hipThreadIdx_x + 1, j, k + -1);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 7; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf11[1 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + -1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf9[0] -= x(i + hipThreadIdx_x, j + -1, k + -1);
        buf10[0] -= x(i + hipThreadIdx_x + 1, j + -1, k);
      }
      {
        buf9[1] -= x(i + hipThreadIdx_x, j, k + -1);
        buf10[1] -= x(i + hipThreadIdx_x + 1, j, k);
        buf11[8] -= x(i + hipThreadIdx_x + 1, j, k);
        buf11[0] += x(i + hipThreadIdx_x, j, k + -1);
        buf12[0] -= x(i + hipThreadIdx_x + 1, j, k);
        buf12[0] += x(i + hipThreadIdx_x + -1, j, k);
        buf13[0] += x(i + hipThreadIdx_x, j, k + -1);
        buf14[0] -= x(i + hipThreadIdx_x + 1, j, k);
        buf14[0] += x(i + hipThreadIdx_x + -1, j, k);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf9[0 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf9[2 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf10[0 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf10[2 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf12[0 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf12[0 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
            buf12[1 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf12[1 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
            buf13[0 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf13[1 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf11[9 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf11[1 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf14[1 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf14[1 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf9[6] += x(i + hipThreadIdx_x, j + 7, k + -1);
        buf10[6] += x(i + hipThreadIdx_x + 1, j + 7, k);
        buf12[6] += x(i + hipThreadIdx_x + 1, j + 7, k);
        buf12[6] -= x(i + hipThreadIdx_x + -1, j + 7, k);
        buf12[7] -= x(i + hipThreadIdx_x + 1, j + 7, k);
        buf12[7] += x(i + hipThreadIdx_x + -1, j + 7, k);
        buf13[6] -= x(i + hipThreadIdx_x, j + 7, k + -1);
        buf13[7] += x(i + hipThreadIdx_x, j + 7, k + -1);
        buf11[15] -= x(i + hipThreadIdx_x + 1, j + 7, k);
        buf11[7] += x(i + hipThreadIdx_x, j + 7, k + -1);
        buf14[7] -= x(i + hipThreadIdx_x + 1, j + 7, k);
        buf14[7] += x(i + hipThreadIdx_x + -1, j + 7, k);
      }
      {
        buf9[7] += x(i + hipThreadIdx_x, j + 8, k + -1);
        buf10[7] += x(i + hipThreadIdx_x + 1, j + 8, k);
        buf12[7] += x(i + hipThreadIdx_x + 1, j + 8, k);
        buf12[7] -= x(i + hipThreadIdx_x + -1, j + 8, k);
        buf13[7] -= x(i + hipThreadIdx_x, j + 8, k + -1);
      }
      {
        buf9[8] -= x(i + hipThreadIdx_x, j + -1, k);
        buf9[0] += x(i + hipThreadIdx_x, j + -1, k);
        buf10[8] -= x(i + hipThreadIdx_x + 1, j + -1, k + 1);
        buf10[0] += x(i + hipThreadIdx_x, j + -1, k);
        buf15[0] += x(i + hipThreadIdx_x, j + -1, k);
      }
      {
        buf9[9] -= x(i + hipThreadIdx_x, j, k);
        buf9[1] += x(i + hipThreadIdx_x, j, k);
        buf10[9] -= x(i + hipThreadIdx_x + 1, j, k + 1);
        buf10[1] += x(i + hipThreadIdx_x, j, k);
        buf15[1] += x(i + hipThreadIdx_x, j, k);
        buf11[0] += x(i + hipThreadIdx_x + 1, j, k + 1);
        buf11[16] -= x(i + hipThreadIdx_x + 1, j, k + 1);
        buf11[8] += x(i + hipThreadIdx_x, j, k);
        buf14[0] += x(i + hipThreadIdx_x + 1, j, k + 1);
        buf14[0] -= x(i + hipThreadIdx_x + -1, j, k + 1);
        buf14[8] -= x(i + hipThreadIdx_x + 1, j, k + 1);
        buf14[8] += x(i + hipThreadIdx_x + -1, j, k + 1);
        buf12[8] -= x(i + hipThreadIdx_x + 1, j, k + 1);
        buf12[8] += x(i + hipThreadIdx_x + -1, j, k + 1);
        buf13[8] += x(i + hipThreadIdx_x, j, k);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf9[8 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf9[0 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf9[10 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf9[2 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf10[8 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf10[0 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf10[10 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf10[2 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf12[8 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf12[8 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 1);
            buf12[9 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf12[9 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 1);
            buf13[8 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf13[9 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf15[0 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf15[2 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf11[1 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf11[17 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf11[9 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf14[1 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf14[1 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 1);
            buf14[9 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf14[9 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf9[14] += x(i + hipThreadIdx_x, j + 7, k);
        buf9[6] -= x(i + hipThreadIdx_x, j + 7, k);
        buf10[14] += x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf10[6] -= x(i + hipThreadIdx_x, j + 7, k);
        buf12[14] += x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf12[14] -= x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf12[15] -= x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf12[15] += x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf13[14] -= x(i + hipThreadIdx_x, j + 7, k);
        buf13[15] += x(i + hipThreadIdx_x, j + 7, k);
        buf15[6] -= x(i + hipThreadIdx_x, j + 7, k);
        buf11[7] += x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf11[23] -= x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf11[15] += x(i + hipThreadIdx_x, j + 7, k);
        buf14[7] += x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf14[7] -= x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf14[15] -= x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf14[15] += x(i + hipThreadIdx_x + -1, j + 7, k + 1);
      }
      {
        buf9[15] += x(i + hipThreadIdx_x, j + 8, k);
        buf9[7] -= x(i + hipThreadIdx_x, j + 8, k);
        buf10[15] += x(i + hipThreadIdx_x + 1, j + 8, k + 1);
        buf10[7] -= x(i + hipThreadIdx_x, j + 8, k);
        buf12[15] += x(i + hipThreadIdx_x + 1, j + 8, k + 1);
        buf12[15] -= x(i + hipThreadIdx_x + -1, j + 8, k + 1);
        buf13[15] -= x(i + hipThreadIdx_x, j + 8, k);
        buf15[7] -= x(i + hipThreadIdx_x, j + 8, k);
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 5; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            buf9[16 + rel] -= x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
            buf9[8 + rel] += x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
            buf10[16 + rel] -= x(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 2);
            buf10[8 + rel] += x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
            buf15[8 + rel] += x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
            buf15[0 + rel] -= x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
          }
          {
            buf9[17 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf9[9 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf10[17 + rel] -= x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf10[9 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf15[9 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf15[1 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf11[8 + rel] += x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf11[0 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf11[24 + rel] -= x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf11[16 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf14[8 + rel] += x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf14[8 + rel] -= x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
            buf14[16 + rel] -= x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf14[16 + rel] += x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
            buf13[0 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf13[16 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf12[16 + rel] -= x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf12[16 + rel] += x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
          }
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf9[16 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf9[8 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf9[18 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf9[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf10[16 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf10[8 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf10[18 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf10[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf12[16 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf12[16 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf12[17 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf12[17 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf13[16 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf13[1 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf13[17 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf13[0 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf15[8 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf15[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf15[0 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf15[2 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf11[9 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf11[1 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf11[25 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf11[17 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf14[9 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf14[9 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf14[17 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf14[17 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
              }
              _cg_rel1 += 1;
            }
          }
          {
            buf9[22 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf9[14 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf10[22 + rel] += x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf10[14 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf12[22 + rel] += x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf12[22 + rel] -= x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf12[23 + rel] -= x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf12[23 + rel] += x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf13[22 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf13[7 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf13[23 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf13[6 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf15[14 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf15[6 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf11[15 + rel] += x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf11[7 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf11[31 + rel] -= x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf11[23 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf14[15 + rel] += x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf14[15 + rel] -= x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf14[23 + rel] -= x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf14[23 + rel] += x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
          }
          {
            buf9[23 + rel] += x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf9[15 + rel] -= x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf10[23 + rel] += x(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 2);
            buf10[15 + rel] -= x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf12[23 + rel] += x(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 2);
            buf12[23 + rel] -= x(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 2);
            buf13[23 + rel] -= x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf13[7 + rel] += x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf15[15 + rel] -= x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf15[7 + rel] += x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
          }
          _cg_rel2 += 8;
        }
      }
      {
        buf9[56] -= x(i + hipThreadIdx_x, j + -1, k + 6);
        buf9[48] += x(i + hipThreadIdx_x, j + -1, k + 6);
        buf10[56] -= x(i + hipThreadIdx_x + 1, j + -1, k + 7);
        buf10[48] += x(i + hipThreadIdx_x, j + -1, k + 6);
        buf15[48] += x(i + hipThreadIdx_x, j + -1, k + 6);
        buf15[40] -= x(i + hipThreadIdx_x, j + -1, k + 6);
      }
      {
        buf9[57] -= x(i + hipThreadIdx_x, j, k + 6);
        buf9[49] += x(i + hipThreadIdx_x, j, k + 6);
        buf10[57] -= x(i + hipThreadIdx_x + 1, j, k + 7);
        buf10[49] += x(i + hipThreadIdx_x, j, k + 6);
        buf15[49] += x(i + hipThreadIdx_x, j, k + 6);
        buf15[41] -= x(i + hipThreadIdx_x, j, k + 6);
        buf11[48] += x(i + hipThreadIdx_x + 1, j, k + 7);
        buf11[40] -= x(i + hipThreadIdx_x, j, k + 6);
        buf11[56] += x(i + hipThreadIdx_x, j, k + 6);
        buf14[48] += x(i + hipThreadIdx_x + 1, j, k + 7);
        buf14[48] -= x(i + hipThreadIdx_x + -1, j, k + 7);
        buf14[56] -= x(i + hipThreadIdx_x + 1, j, k + 7);
        buf14[56] += x(i + hipThreadIdx_x + -1, j, k + 7);
        buf13[40] -= x(i + hipThreadIdx_x, j, k + 6);
        buf13[56] += x(i + hipThreadIdx_x, j, k + 6);
        buf12[56] -= x(i + hipThreadIdx_x + 1, j, k + 7);
        buf12[56] += x(i + hipThreadIdx_x + -1, j, k + 7);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf9[56 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf9[48 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf9[58 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf9[50 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf10[56 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf10[48 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf10[58 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf10[50 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf12[56 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf12[56 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf12[57 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf12[57 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf13[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf13[41 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf13[57 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf13[40 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf15[48 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf15[50 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf15[40 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf15[42 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf11[49 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf11[41 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf11[57 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf14[49 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf14[49 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf14[57 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf14[57 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf9[62] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf9[54] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf10[62] += x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf10[54] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf12[62] += x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf12[62] -= x(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf12[63] -= x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf12[63] += x(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf13[62] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf13[47] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf13[63] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf13[46] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf15[54] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf15[46] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf11[55] += x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf11[47] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf11[63] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf14[55] += x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf14[55] -= x(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf14[63] -= x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf14[63] += x(i + hipThreadIdx_x + -1, j + 7, k + 7);
      }
      {
        buf9[63] += x(i + hipThreadIdx_x, j + 8, k + 6);
        buf9[55] -= x(i + hipThreadIdx_x, j + 8, k + 6);
        buf10[63] += x(i + hipThreadIdx_x + 1, j + 8, k + 7);
        buf10[55] -= x(i + hipThreadIdx_x, j + 8, k + 6);
        buf12[63] += x(i + hipThreadIdx_x + 1, j + 8, k + 7);
        buf12[63] -= x(i + hipThreadIdx_x + -1, j + 8, k + 7);
        buf13[63] -= x(i + hipThreadIdx_x, j + 8, k + 6);
        buf13[47] += x(i + hipThreadIdx_x, j + 8, k + 6);
        buf15[55] -= x(i + hipThreadIdx_x, j + 8, k + 6);
        buf15[47] += x(i + hipThreadIdx_x, j + 8, k + 6);
      }
      {
        buf9[56] += x(i + hipThreadIdx_x, j + -1, k + 7);
        buf10[56] += x(i + hipThreadIdx_x, j + -1, k + 7);
        buf15[56] += x(i + hipThreadIdx_x, j + -1, k + 7);
        buf15[48] -= x(i + hipThreadIdx_x, j + -1, k + 7);
      }
      {
        buf9[57] += x(i + hipThreadIdx_x, j, k + 7);
        buf10[57] += x(i + hipThreadIdx_x, j, k + 7);
        buf15[57] += x(i + hipThreadIdx_x, j, k + 7);
        buf15[49] -= x(i + hipThreadIdx_x, j, k + 7);
        buf11[56] += x(i + hipThreadIdx_x + 1, j, k + 8);
        buf11[48] -= x(i + hipThreadIdx_x, j, k + 7);
        buf14[56] += x(i + hipThreadIdx_x + 1, j, k + 8);
        buf14[56] -= x(i + hipThreadIdx_x + -1, j, k + 8);
        buf13[48] -= x(i + hipThreadIdx_x, j, k + 7);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf9[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf9[58 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf10[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf10[58 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf15[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf15[58 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf15[48 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf15[50 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf11[57 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 8);
            buf11[49 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf14[57 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 8);
            buf14[57 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 8);
            buf13[49 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf13[48 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf9[62] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf10[62] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf15[62] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf15[54] += x(i + hipThreadIdx_x, j + 7, k + 7);
        buf11[63] += x(i + hipThreadIdx_x + 1, j + 7, k + 8);
        buf11[55] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf14[63] += x(i + hipThreadIdx_x + 1, j + 7, k + 8);
        buf14[63] -= x(i + hipThreadIdx_x + -1, j + 7, k + 8);
        buf13[55] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf13[54] += x(i + hipThreadIdx_x, j + 7, k + 7);
      }
      {
        buf9[63] -= x(i + hipThreadIdx_x, j + 8, k + 7);
        buf10[63] -= x(i + hipThreadIdx_x, j + 8, k + 7);
        buf15[63] -= x(i + hipThreadIdx_x, j + 8, k + 7);
        buf15[55] += x(i + hipThreadIdx_x, j + 8, k + 7);
        buf13[55] += x(i + hipThreadIdx_x, j + 8, k + 7);
      }
      {
        buf15[56] -= x(i + hipThreadIdx_x, j + -1, k + 8);
      }
      {
        buf11[56] -= x(i + hipThreadIdx_x, j, k + 8);
        buf13[56] -= x(i + hipThreadIdx_x, j, k + 8);
        buf15[57] -= x(i + hipThreadIdx_x, j, k + 8);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf11[57 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
            buf13[57 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
            buf13[56 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
            buf15[56 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
            buf15[58 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf11[63] -= x(i + hipThreadIdx_x, j + 7, k + 8);
        buf13[63] -= x(i + hipThreadIdx_x, j + 7, k + 8);
        buf13[62] += x(i + hipThreadIdx_x, j + 7, k + 8);
        buf15[62] += x(i + hipThreadIdx_x, j + 7, k + 8);
      }
      {
        buf13[63] += x(i + hipThreadIdx_x, j + 8, k + 8);
        buf15[63] += x(i + hipThreadIdx_x, j + 8, k + 8);
      }
    }
  }
  bElem buf16[64];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf16[0 + rel] = 0;
              }
              _cg_rel1 += 1;
            }
          }
          _cg_rel2 += 8;
        }
      }
    }
    {
      {
        buf16[0] += (beta_i(i + hipThreadIdx_x, j + 1, k) - beta_i(i + hipThreadIdx_x, j + -1, k)) * buf4[0];
        buf16[0] += (beta_j(i + hipThreadIdx_x + 1, j, k) - beta_j(i + hipThreadIdx_x + -1, j, k)) * buf6[0];
        buf16[0] += (beta_j(i + hipThreadIdx_x, j, k + 1) - beta_j(i + hipThreadIdx_x, j, k + -1)) * buf7[0];
        buf16[0] += (beta_k(i + hipThreadIdx_x + 1, j, k) - beta_k(i + hipThreadIdx_x + -1, j, k)) * buf8[0];
        buf16[0] += (beta_k(i + hipThreadIdx_x, j + 1, k) - beta_k(i + hipThreadIdx_x, j + -1, k)) * buf9[0];
        buf16[0] += (beta_i(i + hipThreadIdx_x + 1, j + 1, k) - beta_i(i + hipThreadIdx_x + 1, j + -1, k)) * buf10[0];
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 7; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf16[1 + rel] += (beta_i(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) - beta_i(i + hipThreadIdx_x, j + _cg_idx1, k)) * buf4[1 + rel];
            buf16[1 + rel] += (beta_j(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k) - beta_j(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k)) * buf6[1 + rel];
            buf16[1 + rel] += (beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 1) - beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1)) * buf7[1 + rel];
            buf16[1 + rel] += (beta_k(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k) - beta_k(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k)) * buf8[1 + rel];
            buf16[1 + rel] += (beta_k(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) - beta_k(i + hipThreadIdx_x, j + _cg_idx1, k)) * buf9[1 + rel];
            buf16[1 + rel] += (beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k) - beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k)) * buf10[1 + rel];
            buf16[0 + rel] += (beta_j(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k) - beta_j(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k)) * buf12[0 + rel];
            buf16[0 + rel] += (beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 1) - beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1)) * buf13[0 + rel];
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf16[7] += (beta_j(i + hipThreadIdx_x + 1, j + 8, k) - beta_j(i + hipThreadIdx_x + -1, j + 8, k)) * buf12[7];
        buf16[7] += (beta_j(i + hipThreadIdx_x, j + 8, k + 1) - beta_j(i + hipThreadIdx_x, j + 8, k + -1)) * buf13[7];
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 7; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            buf16[8 + rel] += (beta_i(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1)) * buf4[8 + rel];
            buf16[8 + rel] += (beta_j(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1) - beta_j(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1)) * buf6[8 + rel];
            buf16[8 + rel] += (beta_j(i + hipThreadIdx_x, j, k + _cg_idx2 + 2) - beta_j(i + hipThreadIdx_x, j, k + _cg_idx2)) * buf7[8 + rel];
            buf16[8 + rel] += (beta_k(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1)) * buf8[8 + rel];
            buf16[8 + rel] += (beta_k(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1)) * buf9[8 + rel];
            buf16[8 + rel] += (beta_i(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 1)) * buf10[8 + rel];
            buf16[0 + rel] += (beta_k(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1)) * buf14[0 + rel];
            buf16[0 + rel] += (beta_k(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1)) * buf15[0 + rel];
          }
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 7; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf16[9 + rel] += (beta_i(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1)) * buf4[9 + rel];
                buf16[9 + rel] += (beta_j(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) - beta_j(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1)) * buf6[9 + rel];
                buf16[9 + rel] += (beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 2) - beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2)) * buf7[9 + rel];
                buf16[9 + rel] += (beta_k(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1)) * buf8[9 + rel];
                buf16[9 + rel] += (beta_k(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1)) * buf9[9 + rel];
                buf16[9 + rel] += (beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2 + 1)) * buf10[9 + rel];
                buf16[0 + rel] += (beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + -1)) * buf5[0 + rel];
                buf16[0 + rel] += (beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2 + -1)) * buf11[0 + rel];
                buf16[8 + rel] += (beta_j(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) - beta_j(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1)) * buf12[8 + rel];
                buf16[8 + rel] += (beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 2) - beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2)) * buf13[8 + rel];
                buf16[1 + rel] += (beta_k(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1)) * buf14[1 + rel];
                buf16[1 + rel] += (beta_k(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1)) * buf15[1 + rel];
              }
              _cg_rel1 += 1;
            }
          }
          {
            buf16[7 + rel] += (beta_i(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + -1)) * buf5[7 + rel];
            buf16[7 + rel] += (beta_i(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + -1)) * buf11[7 + rel];
            buf16[15 + rel] += (beta_j(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 1) - beta_j(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 1)) * buf12[15 + rel];
            buf16[15 + rel] += (beta_j(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2) - beta_j(i + hipThreadIdx_x, j + 8, k + _cg_idx2)) * buf13[15 + rel];
          }
          _cg_rel2 += 8;
        }
      }
      {
        buf16[56] += (beta_k(i + hipThreadIdx_x + 1, j, k + 8) - beta_k(i + hipThreadIdx_x + -1, j, k + 8)) * buf14[56];
        buf16[56] += (beta_k(i + hipThreadIdx_x, j + 1, k + 8) - beta_k(i + hipThreadIdx_x, j + -1, k + 8)) * buf15[56];
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 7; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf16[56 + rel] += (beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + 8) - beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + 6)) * buf5[56 + rel];
            buf16[56 + rel] += (beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + 8) - beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + 6)) * buf11[56 + rel];
            buf16[57 + rel] += (beta_k(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 8) - beta_k(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 8)) * buf14[57 + rel];
            buf16[57 + rel] += (beta_k(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 8) - beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + 8)) * buf15[57 + rel];
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf16[63] += (beta_i(i + hipThreadIdx_x, j + 7, k + 8) - beta_i(i + hipThreadIdx_x, j + 7, k + 6)) * buf5[63];
        buf16[63] += (beta_i(i + hipThreadIdx_x + 1, j + 7, k + 8) - beta_i(i + hipThreadIdx_x + 1, j + 7, k + 6)) * buf11[63];
      }
    }
  }
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf15[0 + rel] = c[0] * alpha(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) - c[1] * c[2] * (0.0833 * (beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * (15.0 * x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + _cg_idx2) - 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) - x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + _cg_idx2) + x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2)) + beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2) * (15.0 * x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2) - 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) - x(i + hipThreadIdx_x + 2, j + _cg_idx1, k + _cg_idx2) + x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + _cg_idx2)) + beta_j(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * buf0[0 + rel] + beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2) * buf1[0 + rel] + beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * buf2[0 + rel] + beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1) * buf3[0 + rel]) + 0.020825 * buf16[0 + rel]);
              }
              _cg_rel1 += 1;
            }
          }
          _cg_rel2 += 8;
        }
      }
    }
    {
      long rel = 0;
      for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
      {
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          for (long _cg_idx0 = hipThreadIdx_x; _cg_idx0 < 64; _cg_idx0 += 64, ++rel)
          {
            out(i + _cg_idx0, j + _cg_idx1, k + _cg_idx2) = buf15[rel];
          }
        }
      }
    }
  }
}
# 230 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/helmholtz4/intermediate_gen/helmholtz4.cu" 2

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
# 1 "VSBrick-helmholtz43.py-HIP-8x8x8-8x8" 1
{
  auto *binfo = out.bInfo;
  long neighbor0 = binfo->adj[b][0];
  long neighbor1 = binfo->adj[b][1];
  long neighbor2 = binfo->adj[b][2];
  long neighbor3 = binfo->adj[b][3];
  long neighbor4 = binfo->adj[b][4];
  long neighbor5 = binfo->adj[b][5];
  long neighbor6 = binfo->adj[b][6];
  long neighbor7 = binfo->adj[b][7];
  long neighbor8 = binfo->adj[b][8];
  long neighbor9 = binfo->adj[b][9];
  long neighbor10 = binfo->adj[b][10];
  long neighbor11 = binfo->adj[b][11];
  long neighbor12 = binfo->adj[b][12];
  long neighbor13 = b;
  long neighbor14 = binfo->adj[b][14];
  long neighbor15 = binfo->adj[b][15];
  long neighbor16 = binfo->adj[b][16];
  long neighbor17 = binfo->adj[b][17];
  long neighbor18 = binfo->adj[b][18];
  long neighbor19 = binfo->adj[b][19];
  long neighbor20 = binfo->adj[b][20];
  long neighbor21 = binfo->adj[b][21];
  long neighbor22 = binfo->adj[b][22];
  long neighbor23 = binfo->adj[b][23];
  long neighbor24 = binfo->adj[b][24];
  long neighbor25 = binfo->adj[b][25];
  long neighbor26 = binfo->adj[b][26];
  bElem buf0[8];
  bElem buf1[8];
  bElem buf2[8];
  bElem buf3[8];
  bElem buf4[8];
  bElem buf5[8];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 0]
            buf0[0 + rel] = 0;
            buf1[0 + rel] = 0;
            buf2[0 + rel] = 0;
            buf3[0 + rel] = 0;
            buf4[0 + rel] = 0;
            buf5[0 + rel] = 0;
          }
          _cg_rel2 += 1;
        }
      }
    }
    {
      bElem _cg_x_100_vecbuf;
      bElem _cg_x000_vecbuf;
      {
        // New offset [1, 0, -2]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor4 * x.step + 384 + hipThreadIdx_x];
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf4[0] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 0, -1]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf4[0] += 15.0 * _cg_x_100_reg;
        buf4[1] -= _cg_x_100_reg;
        buf5[0] += _cg_x_100_reg;
      }
      {
        // New offset [1, -1, 0]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[0] += 15.0 * _cg_x_100_reg;
        buf2[0] -= _cg_x_100_reg;
        buf3[0] += _cg_x_100_reg;
      }
      {
        // New offset [0, 0, 0]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[0] += 15.0 * _cg_x_100_reg;
        buf0[0] -= _cg_x_100_reg;
        buf1[0] += _cg_x_100_reg;
      }
      {
        // New offset [1, 0, 0]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = _cg_x000_vecbuf;
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf0[0] -= 15.0 * _cg_x_100_reg;
        buf1[0] -= 15.0 * _cg_x_100_reg;
        buf2[0] -= 15.0 * _cg_x_100_reg;
        buf3[0] -= 15.0 * _cg_x_100_reg;
        buf4[0] -= 15.0 * _cg_x_100_reg;
        buf4[1] += 15.0 * _cg_x_100_reg;
        buf4[2] -= _cg_x_100_reg;
        buf5[0] -= 15.0 * _cg_x_100_reg;
        buf5[1] += _cg_x_100_reg;
      }
      {
        // New offset [2, 0, 0]
        bElem _cg_x_100_reg;
        {
          _cg_x000_vecbuf = x.dat[neighbor14 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[0] += _cg_x_100_reg;
        buf1[0] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [3, 0, 0]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf1[0] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 1, 0]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[0] += _cg_x_100_reg;
        buf3[0] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [1, 2, 0]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf3[0] -= _cg_x_100_reg;
      }
      {
        // New offset [1, -1, 1]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[1] += 15.0 * _cg_x_100_reg;
        buf2[1] -= _cg_x_100_reg;
        buf3[1] += _cg_x_100_reg;
      }
      {
        // New offset [0, 0, 1]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + 64 + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor13 * x.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[1] += 15.0 * _cg_x_100_reg;
        buf0[1] -= _cg_x_100_reg;
        buf1[1] += _cg_x_100_reg;
      }
      {
        // New offset [1, 0, 1]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = _cg_x000_vecbuf;
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf0[1] -= 15.0 * _cg_x_100_reg;
        buf1[1] -= 15.0 * _cg_x_100_reg;
        buf2[1] -= 15.0 * _cg_x_100_reg;
        buf3[1] -= 15.0 * _cg_x_100_reg;
        buf4[1] -= 15.0 * _cg_x_100_reg;
        buf4[2] += 15.0 * _cg_x_100_reg;
        buf4[3] -= _cg_x_100_reg;
        buf4[0] += _cg_x_100_reg;
        buf5[1] -= 15.0 * _cg_x_100_reg;
        buf5[2] += _cg_x_100_reg;
        buf5[0] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [2, 0, 1]
        bElem _cg_x_100_reg;
        {
          _cg_x000_vecbuf = x.dat[neighbor14 * x.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[1] += _cg_x_100_reg;
        buf1[1] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [3, 0, 1]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf1[1] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 1, 1]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[1] += _cg_x_100_reg;
        buf3[1] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [1, 2, 1]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf3[1] -= _cg_x_100_reg;
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 4; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [1, -1, 2]
            bElem _cg_x_100_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor13 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor10 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_100_vecbuf
              dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              _cg_x_100_reg = _cg_x_100_vecbuf;
            }
            buf2[2 + rel] += 15.0 * _cg_x_100_reg;
            buf2[2 + rel] -= _cg_x_100_reg;
            buf3[2 + rel] += _cg_x_100_reg;
          }
          {
            // New offset [0, 0, 2]
            bElem _cg_x_100_reg;
            {
              _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_x000_vecbuf = x.dat[neighbor13 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_x_100_reg = _cg_vectmp0;
            }
            buf0[2 + rel] += 15.0 * _cg_x_100_reg;
            buf0[2 + rel] -= _cg_x_100_reg;
            buf1[2 + rel] += _cg_x_100_reg;
          }
          {
            // New offset [1, 0, 2]
            bElem _cg_x_100_reg;
            {
              _cg_x_100_vecbuf = _cg_x000_vecbuf;
              _cg_x_100_reg = _cg_x_100_vecbuf;
            }
            buf0[2 + rel] -= 15.0 * _cg_x_100_reg;
            buf1[2 + rel] -= 15.0 * _cg_x_100_reg;
            buf2[2 + rel] -= 15.0 * _cg_x_100_reg;
            buf3[2 + rel] -= 15.0 * _cg_x_100_reg;
            buf4[2 + rel] -= 15.0 * _cg_x_100_reg;
            buf4[3 + rel] += 15.0 * _cg_x_100_reg;
            buf4[4 + rel] -= _cg_x_100_reg;
            buf4[1 + rel] += _cg_x_100_reg;
            buf5[2 + rel] -= 15.0 * _cg_x_100_reg;
            buf5[3 + rel] += _cg_x_100_reg;
            buf5[1 + rel] += 15.0 * _cg_x_100_reg;
            buf5[0 + rel] -= _cg_x_100_reg;
          }
          {
            // New offset [2, 0, 2]
            bElem _cg_x_100_reg;
            {
              _cg_x000_vecbuf = x.dat[neighbor14 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_x_100_reg = _cg_vectmp0;
            }
            buf0[2 + rel] += _cg_x_100_reg;
            buf1[2 + rel] += 15.0 * _cg_x_100_reg;
          }
          {
            // New offset [3, 0, 2]
            bElem _cg_x_100_reg;
            {
              bElem _cg_vectmp0;
              // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 2 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 6, 8, hipThreadIdx_x & 7);
              _cg_x_100_reg = _cg_vectmp0;
            }
            buf1[2 + rel] -= _cg_x_100_reg;
          }
          {
            // New offset [1, 1, 2]
            bElem _cg_x_100_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor16 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor13 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_100_vecbuf
              dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              _cg_x_100_reg = _cg_x_100_vecbuf;
            }
            buf2[2 + rel] += _cg_x_100_reg;
            buf3[2 + rel] += 15.0 * _cg_x_100_reg;
          }
          {
            // New offset [1, 2, 2]
            bElem _cg_x_100_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor16 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor13 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_x_100_vecbuf
              dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
              _cg_x_100_reg = _cg_x_100_vecbuf;
            }
            buf3[2 + rel] -= _cg_x_100_reg;
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [1, -1, 6]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[6] += 15.0 * _cg_x_100_reg;
        buf2[6] -= _cg_x_100_reg;
        buf3[6] += _cg_x_100_reg;
      }
      {
        // New offset [0, 0, 6]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + 384 + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor13 * x.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[6] += 15.0 * _cg_x_100_reg;
        buf0[6] -= _cg_x_100_reg;
        buf1[6] += _cg_x_100_reg;
      }
      {
        // New offset [1, 0, 6]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = _cg_x000_vecbuf;
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf0[6] -= 15.0 * _cg_x_100_reg;
        buf1[6] -= 15.0 * _cg_x_100_reg;
        buf2[6] -= 15.0 * _cg_x_100_reg;
        buf3[6] -= 15.0 * _cg_x_100_reg;
        buf4[6] -= 15.0 * _cg_x_100_reg;
        buf4[7] += 15.0 * _cg_x_100_reg;
        buf4[5] += _cg_x_100_reg;
        buf5[6] -= 15.0 * _cg_x_100_reg;
        buf5[7] += _cg_x_100_reg;
        buf5[5] += 15.0 * _cg_x_100_reg;
        buf5[4] -= _cg_x_100_reg;
      }
      {
        // New offset [2, 0, 6]
        bElem _cg_x_100_reg;
        {
          _cg_x000_vecbuf = x.dat[neighbor14 * x.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[6] += _cg_x_100_reg;
        buf1[6] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [3, 0, 6]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf1[6] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 1, 6]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[6] += _cg_x_100_reg;
        buf3[6] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [1, 2, 6]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf3[6] -= _cg_x_100_reg;
      }
      {
        // New offset [1, -1, 7]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[7] += 15.0 * _cg_x_100_reg;
        buf2[7] -= _cg_x_100_reg;
        buf3[7] += _cg_x_100_reg;
      }
      {
        // New offset [0, 0, 7]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + 448 + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[7] += 15.0 * _cg_x_100_reg;
        buf0[7] -= _cg_x_100_reg;
        buf1[7] += _cg_x_100_reg;
      }
      {
        // New offset [1, 0, 7]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = _cg_x000_vecbuf;
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf0[7] -= 15.0 * _cg_x_100_reg;
        buf1[7] -= 15.0 * _cg_x_100_reg;
        buf2[7] -= 15.0 * _cg_x_100_reg;
        buf3[7] -= 15.0 * _cg_x_100_reg;
        buf4[7] -= 15.0 * _cg_x_100_reg;
        buf4[6] += _cg_x_100_reg;
        buf5[7] -= 15.0 * _cg_x_100_reg;
        buf5[6] += 15.0 * _cg_x_100_reg;
        buf5[5] -= _cg_x_100_reg;
      }
      {
        // New offset [2, 0, 7]
        bElem _cg_x_100_reg;
        {
          _cg_x000_vecbuf = x.dat[neighbor14 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_x_100_reg;
        buf1[7] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [3, 0, 7]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf1[7] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 1, 7]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[7] += _cg_x_100_reg;
        buf3[7] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [1, 2, 7]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf3[7] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 0, 8]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf4[7] += _cg_x_100_reg;
        buf5[7] += 15.0 * _cg_x_100_reg;
        buf5[6] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 0, 9]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor22 * x.step + 64 + hipThreadIdx_x];
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf5[7] -= _cg_x_100_reg;
      }
    }
  }
  bElem buf6[8];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 0]
            buf6[0 + rel] = 0;
          }
          _cg_rel2 += 1;
        }
      }
    }
    {
      bElem _cg_beta_i000_vecbuf;
      bElem _cg_beta_i_100_vecbuf;
      bElem _cg_beta_j000_vecbuf;
      bElem _cg_beta_k000_vecbuf;
      {
        // New offset [0, 0, 0]
        bElem _cg_beta_i000_reg;
        bElem _cg_beta_j000_reg;
        bElem _cg_beta_k000_reg;
        {
          _cg_beta_i000_vecbuf = beta_i.dat[neighbor13 * beta_i.step + hipThreadIdx_x];
          _cg_beta_j000_vecbuf = beta_j.dat[neighbor13 * beta_j.step + hipThreadIdx_x];
          _cg_beta_k000_vecbuf = beta_k.dat[neighbor13 * beta_k.step + hipThreadIdx_x];
          _cg_beta_i000_reg = _cg_beta_i000_vecbuf;
          _cg_beta_j000_reg = _cg_beta_j000_vecbuf;
          _cg_beta_k000_reg = _cg_beta_k000_vecbuf;
        }
        buf6[0] += _cg_beta_i000_reg * buf0[0];
        buf6[0] += _cg_beta_j000_reg * buf2[0];
        buf6[0] += _cg_beta_k000_reg * buf4[0];
      }
      {
        // New offset [1, 0, 0]
        bElem _cg_beta_i000_reg;
        {
          _cg_beta_i_100_vecbuf = _cg_beta_i000_vecbuf;
          _cg_beta_i000_vecbuf = beta_i.dat[neighbor14 * beta_i.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_beta_i_100_vecbuf ,_cg_beta_i000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_beta_i_100_vecbuf, _cg_beta_i000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i000_reg = _cg_vectmp0;
        }
        buf6[0] += _cg_beta_i000_reg * buf1[0];
      }
      {
        // New offset [0, 1, 0]
        bElem _cg_beta_j000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_j.dat[neighbor16 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_j.dat[neighbor13 * beta_j.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j000_vecbuf
          dev_shl(_cg_beta_j000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_beta_j000_reg = _cg_beta_j000_vecbuf;
        }
        buf6[0] += _cg_beta_j000_reg * buf3[0];
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 7; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 1]
            bElem _cg_beta_i000_reg;
            bElem _cg_beta_j000_reg;
            bElem _cg_beta_k000_reg;
            {
              _cg_beta_i000_vecbuf = beta_i.dat[neighbor13 * beta_i.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_j000_vecbuf = beta_j.dat[neighbor13 * beta_j.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_k000_vecbuf = beta_k.dat[neighbor13 * beta_k.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_i000_reg = _cg_beta_i000_vecbuf;
              _cg_beta_j000_reg = _cg_beta_j000_vecbuf;
              _cg_beta_k000_reg = _cg_beta_k000_vecbuf;
            }
            buf6[1 + rel] += _cg_beta_i000_reg * buf0[1 + rel];
            buf6[1 + rel] += _cg_beta_j000_reg * buf2[1 + rel];
            buf6[1 + rel] += _cg_beta_k000_reg * buf4[1 + rel];
            buf6[0 + rel] += _cg_beta_k000_reg * buf5[0 + rel];
          }
          {
            // New offset [1, 0, 1]
            bElem _cg_beta_i000_reg;
            {
              _cg_beta_i_100_vecbuf = _cg_beta_i000_vecbuf;
              _cg_beta_i000_vecbuf = beta_i.dat[neighbor14 * beta_i.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_beta_i_100_vecbuf ,_cg_beta_i000_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_beta_i_100_vecbuf, _cg_beta_i000_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_i000_reg = _cg_vectmp0;
            }
            buf6[1 + rel] += _cg_beta_i000_reg * buf1[1 + rel];
          }
          {
            // New offset [0, 1, 1]
            bElem _cg_beta_j000_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = beta_j.dat[neighbor16 * beta_j.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = beta_j.dat[neighbor13 * beta_j.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j000_vecbuf
              dev_shl(_cg_beta_j000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              _cg_beta_j000_reg = _cg_beta_j000_vecbuf;
            }
            buf6[1 + rel] += _cg_beta_j000_reg * buf3[1 + rel];
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [0, 0, 8]
        bElem _cg_beta_k000_reg;
        {
          _cg_beta_k000_vecbuf = beta_k.dat[neighbor22 * beta_k.step + hipThreadIdx_x];
          _cg_beta_k000_reg = _cg_beta_k000_vecbuf;
        }
        buf6[7] += _cg_beta_k000_reg * buf5[7];
      }
    }
  }
  bElem buf7[8];
  bElem buf8[8];
  bElem buf9[8];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 0]
            buf5[0 + rel] = 0;
            buf4[0 + rel] = 0;
            buf3[0 + rel] = 0;
            buf2[0 + rel] = 0;
            buf1[0 + rel] = 0;
            buf0[0 + rel] = 0;
            buf7[0 + rel] = 0;
            buf8[0 + rel] = 0;
            buf9[0 + rel] = 0;
          }
          _cg_rel2 += 1;
        }
      }
    }
    {
      bElem _cg_x_110_vecbuf;
      bElem _cg_x010_vecbuf;
      {
        // New offset [1, -2, -1]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor1 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf2[0] -= _cg_x_110_reg;
        buf0[0] -= _cg_x_110_reg;
      }
      {
        // New offset [0, -1, -1]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = x.dat[neighbor3 * x.step + 448 + hipThreadIdx_x];
          _cg_x010_vecbuf = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf4[0] -= _cg_x_110_reg;
        buf1[0] -= _cg_x_110_reg;
      }
      {
        // New offset [1, -1, -1]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf4[0] += _cg_x_110_reg;
        buf2[0] += _cg_x_110_reg;
        buf8[0] += _cg_x_110_reg;
      }
      {
        // New offset [2, -1, -1]
        bElem _cg_x_110_reg;
        {
          _cg_x010_vecbuf = x.dat[neighbor5 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf1[0] += _cg_x_110_reg;
        buf8[0] -= _cg_x_110_reg;
      }
      {
        // New offset [1, 0, -1]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor7 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf0[0] += _cg_x_110_reg;
      }
      {
        // New offset [0, -2, 0]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor12 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor9 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = x.dat[neighbor10 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp4;
        }
        buf5[0] -= _cg_x_110_reg;
        buf3[0] -= _cg_x_110_reg;
      }
      {
        // New offset [1, -2, 0]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf5[0] += _cg_x_110_reg;
        buf0[0] += _cg_x_110_reg;
        buf0[1] -= _cg_x_110_reg;
        buf7[0] += _cg_x_110_reg;
        buf2[1] -= _cg_x_110_reg;
      }
      {
        // New offset [2, -2, 0]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor14 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor11 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp2;
        }
        buf3[0] += _cg_x_110_reg;
        buf7[0] -= _cg_x_110_reg;
      }
      {
        // New offset [0, -1, 0]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = x.dat[neighbor12 * x.step + hipThreadIdx_x];
          _cg_x010_vecbuf = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf4[1] -= _cg_x_110_reg;
        buf1[1] -= _cg_x_110_reg;
        buf1[0] += _cg_x_110_reg;
        buf3[0] += _cg_x_110_reg;
        buf9[0] += _cg_x_110_reg;
      }
      {
        // New offset [1, -1, 0]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf4[1] += _cg_x_110_reg;
        buf2[1] += _cg_x_110_reg;
        buf8[1] += _cg_x_110_reg;
      }
      {
        // New offset [2, -1, 0]
        bElem _cg_x_110_reg;
        {
          _cg_x010_vecbuf = x.dat[neighbor14 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf3[0] -= _cg_x_110_reg;
        buf1[0] -= _cg_x_110_reg;
        buf1[1] += _cg_x_110_reg;
        buf9[0] -= _cg_x_110_reg;
        buf8[1] -= _cg_x_110_reg;
      }
      {
        // New offset [0, 0, 0]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor15 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor12 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = x.dat[neighbor16 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp4;
        }
        buf5[0] += _cg_x_110_reg;
        buf9[0] -= _cg_x_110_reg;
      }
      {
        // New offset [1, 0, 0]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf5[0] -= _cg_x_110_reg;
        buf0[0] -= _cg_x_110_reg;
        buf0[1] += _cg_x_110_reg;
        buf7[0] -= _cg_x_110_reg;
      }
      {
        // New offset [2, 0, 0]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor17 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor14 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp2;
        }
        buf7[0] += _cg_x_110_reg;
        buf9[0] += _cg_x_110_reg;
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 6; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, -2, 1]
            bElem _cg_x_110_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor12 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor9 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_110_vecbuf
              dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = x.dat[neighbor10 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_x010_vecbuf
              dev_shl(_cg_x010_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_x_110_reg = _cg_vectmp4;
            }
            buf5[1 + rel] -= _cg_x_110_reg;
            buf3[1 + rel] -= _cg_x_110_reg;
          }
          {
            // New offset [1, -2, 1]
            bElem _cg_x_110_reg;
            {
              _cg_x_110_vecbuf = _cg_x010_vecbuf;
              _cg_x_110_reg = _cg_x_110_vecbuf;
            }
            buf5[1 + rel] += _cg_x_110_reg;
            buf0[1 + rel] += _cg_x_110_reg;
            buf0[2 + rel] -= _cg_x_110_reg;
            buf7[1 + rel] += _cg_x_110_reg;
            buf2[0 + rel] += _cg_x_110_reg;
            buf2[2 + rel] -= _cg_x_110_reg;
          }
          {
            // New offset [2, -2, 1]
            bElem _cg_x_110_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor14 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor11 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x010_vecbuf
              dev_shl(_cg_x010_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_x_110_reg = _cg_vectmp2;
            }
            buf3[1 + rel] += _cg_x_110_reg;
            buf7[1 + rel] -= _cg_x_110_reg;
          }
          {
            // New offset [0, -1, 1]
            bElem _cg_x_110_reg;
            {
              _cg_x_110_vecbuf = x.dat[neighbor12 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_x010_vecbuf = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_x_110_reg = _cg_vectmp0;
            }
            buf4[0 + rel] += _cg_x_110_reg;
            buf4[2 + rel] -= _cg_x_110_reg;
            buf1[2 + rel] -= _cg_x_110_reg;
            buf1[1 + rel] += _cg_x_110_reg;
            buf3[1 + rel] += _cg_x_110_reg;
            buf9[1 + rel] += _cg_x_110_reg;
          }
          {
            // New offset [1, -1, 1]
            bElem _cg_x_110_reg;
            {
              _cg_x_110_vecbuf = _cg_x010_vecbuf;
              _cg_x_110_reg = _cg_x_110_vecbuf;
            }
            buf4[0 + rel] -= _cg_x_110_reg;
            buf4[2 + rel] += _cg_x_110_reg;
            buf2[0 + rel] -= _cg_x_110_reg;
            buf2[2 + rel] += _cg_x_110_reg;
            buf8[0 + rel] -= _cg_x_110_reg;
            buf8[2 + rel] += _cg_x_110_reg;
          }
          {
            // New offset [2, -1, 1]
            bElem _cg_x_110_reg;
            {
              _cg_x010_vecbuf = x.dat[neighbor14 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_x_110_reg = _cg_vectmp0;
            }
            buf3[1 + rel] -= _cg_x_110_reg;
            buf1[1 + rel] -= _cg_x_110_reg;
            buf1[2 + rel] += _cg_x_110_reg;
            buf9[1 + rel] -= _cg_x_110_reg;
            buf8[2 + rel] -= _cg_x_110_reg;
            buf8[0 + rel] += _cg_x_110_reg;
          }
          {
            // New offset [0, 0, 1]
            bElem _cg_x_110_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor15 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor12 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_110_vecbuf
              dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = x.dat[neighbor16 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_x010_vecbuf
              dev_shl(_cg_x010_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_x_110_reg = _cg_vectmp4;
            }
            buf5[1 + rel] += _cg_x_110_reg;
            buf9[1 + rel] -= _cg_x_110_reg;
          }
          {
            // New offset [1, 0, 1]
            bElem _cg_x_110_reg;
            {
              _cg_x_110_vecbuf = _cg_x010_vecbuf;
              _cg_x_110_reg = _cg_x_110_vecbuf;
            }
            buf5[1 + rel] -= _cg_x_110_reg;
            buf0[1 + rel] -= _cg_x_110_reg;
            buf0[2 + rel] += _cg_x_110_reg;
            buf7[1 + rel] -= _cg_x_110_reg;
          }
          {
            // New offset [2, 0, 1]
            bElem _cg_x_110_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor17 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor14 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x010_vecbuf
              dev_shl(_cg_x010_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_x_110_reg = _cg_vectmp2;
            }
            buf7[1 + rel] += _cg_x_110_reg;
            buf9[1 + rel] += _cg_x_110_reg;
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [0, -2, 7]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor12 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor9 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = x.dat[neighbor10 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp4;
        }
        buf5[7] -= _cg_x_110_reg;
        buf3[7] -= _cg_x_110_reg;
      }
      {
        // New offset [1, -2, 7]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf5[7] += _cg_x_110_reg;
        buf0[7] += _cg_x_110_reg;
        buf7[7] += _cg_x_110_reg;
        buf2[6] += _cg_x_110_reg;
      }
      {
        // New offset [2, -2, 7]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor14 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor11 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp2;
        }
        buf3[7] += _cg_x_110_reg;
        buf7[7] -= _cg_x_110_reg;
      }
      {
        // New offset [0, -1, 7]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = x.dat[neighbor12 * x.step + 448 + hipThreadIdx_x];
          _cg_x010_vecbuf = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf4[6] += _cg_x_110_reg;
        buf3[7] += _cg_x_110_reg;
        buf1[7] += _cg_x_110_reg;
        buf9[7] += _cg_x_110_reg;
      }
      {
        // New offset [1, -1, 7]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf4[6] -= _cg_x_110_reg;
        buf2[6] -= _cg_x_110_reg;
        buf8[6] -= _cg_x_110_reg;
      }
      {
        // New offset [2, -1, 7]
        bElem _cg_x_110_reg;
        {
          _cg_x010_vecbuf = x.dat[neighbor14 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf3[7] -= _cg_x_110_reg;
        buf1[7] -= _cg_x_110_reg;
        buf9[7] -= _cg_x_110_reg;
        buf8[6] += _cg_x_110_reg;
      }
      {
        // New offset [0, 0, 7]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor15 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor12 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = x.dat[neighbor16 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp4;
        }
        buf5[7] += _cg_x_110_reg;
        buf9[7] -= _cg_x_110_reg;
      }
      {
        // New offset [1, 0, 7]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf5[7] -= _cg_x_110_reg;
        buf0[7] -= _cg_x_110_reg;
        buf7[7] -= _cg_x_110_reg;
      }
      {
        // New offset [2, 0, 7]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor17 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor14 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp2;
        }
        buf7[7] += _cg_x_110_reg;
        buf9[7] += _cg_x_110_reg;
      }
      {
        // New offset [1, -2, 8]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor19 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf2[7] += _cg_x_110_reg;
      }
      {
        // New offset [0, -1, 8]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = x.dat[neighbor21 * x.step + hipThreadIdx_x];
          _cg_x010_vecbuf = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf4[7] += _cg_x_110_reg;
      }
      {
        // New offset [1, -1, 8]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf4[7] -= _cg_x_110_reg;
        buf2[7] -= _cg_x_110_reg;
        buf8[7] -= _cg_x_110_reg;
      }
      {
        // New offset [2, -1, 8]
        bElem _cg_x_110_reg;
        {
          _cg_x010_vecbuf = x.dat[neighbor23 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf8[7] += _cg_x_110_reg;
      }
    }
  }
  bElem buf10[8];
  bElem buf11[8];
  bElem buf12[8];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 0]
            buf10[0 + rel] = 0;
            buf11[0 + rel] = 0;
            buf12[0 + rel] = 0;
          }
          _cg_rel2 += 1;
        }
      }
    }
    {
      bElem _cg_x011_vecbuf;
      bElem _cg_x_111_vecbuf;
      {
        // New offset [0, -1, -2]
        bElem _cg_x011_reg;
        {
          _cg_x011_vecbuf = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[0] += _cg_x011_reg;
      }
      {
        // New offset [0, 0, -2]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor7 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[0] -= _cg_x011_reg;
      }
      {
        // New offset [0, -2, -1]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf12[0] += _cg_x011_reg;
      }
      {
        // New offset [-1, -1, -1]
        bElem _cg_x011_reg;
        {
          _cg_x_111_vecbuf = x.dat[neighbor12 * x.step + hipThreadIdx_x];
          _cg_x011_vecbuf = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x011_reg = _cg_vectmp0;
        }
        buf11[0] += _cg_x011_reg;
      }
      {
        // New offset [0, -1, -1]
        bElem _cg_x011_reg;
        {
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[1] += _cg_x011_reg;
      }
      {
        // New offset [1, -1, -1]
        bElem _cg_x011_reg;
        {
          _cg_x_111_vecbuf = _cg_x011_vecbuf;
          _cg_x011_vecbuf = x.dat[neighbor14 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x011_reg = _cg_vectmp0;
        }
        buf11[0] -= _cg_x011_reg;
      }
      {
        // New offset [0, 0, -1]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[1] -= _cg_x011_reg;
        buf12[0] -= _cg_x011_reg;
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 6; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, -2, 0]
            bElem _cg_x011_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor10 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x011_vecbuf
              dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              _cg_x011_reg = _cg_x011_vecbuf;
            }
            buf12[0 + rel] -= _cg_x011_reg;
            buf12[1 + rel] += _cg_x011_reg;
          }
          {
            // New offset [-1, -1, 0]
            bElem _cg_x011_reg;
            {
              _cg_x_111_vecbuf = x.dat[neighbor12 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_x011_vecbuf = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_x011_reg = _cg_vectmp0;
            }
            buf11[0 + rel] -= _cg_x011_reg;
            buf11[1 + rel] += _cg_x011_reg;
          }
          {
            // New offset [0, -1, 0]
            bElem _cg_x011_reg;
            {
              _cg_x011_reg = _cg_x011_vecbuf;
            }
            buf10[0 + rel] -= _cg_x011_reg;
            buf10[2 + rel] += _cg_x011_reg;
          }
          {
            // New offset [1, -1, 0]
            bElem _cg_x011_reg;
            {
              _cg_x_111_vecbuf = _cg_x011_vecbuf;
              _cg_x011_vecbuf = x.dat[neighbor14 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_x011_reg = _cg_vectmp0;
            }
            buf11[0 + rel] += _cg_x011_reg;
            buf11[1 + rel] -= _cg_x011_reg;
          }
          {
            // New offset [0, 0, 0]
            bElem _cg_x011_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor16 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x011_vecbuf
              dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              _cg_x011_reg = _cg_x011_vecbuf;
            }
            buf10[0 + rel] += _cg_x011_reg;
            buf10[2 + rel] -= _cg_x011_reg;
            buf12[0 + rel] += _cg_x011_reg;
            buf12[1 + rel] -= _cg_x011_reg;
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [0, -2, 6]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf12[6] -= _cg_x011_reg;
        buf12[7] += _cg_x011_reg;
      }
      {
        // New offset [-1, -1, 6]
        bElem _cg_x011_reg;
        {
          _cg_x_111_vecbuf = x.dat[neighbor12 * x.step + 448 + hipThreadIdx_x];
          _cg_x011_vecbuf = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x011_reg = _cg_vectmp0;
        }
        buf11[6] -= _cg_x011_reg;
        buf11[7] += _cg_x011_reg;
      }
      {
        // New offset [0, -1, 6]
        bElem _cg_x011_reg;
        {
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[6] -= _cg_x011_reg;
      }
      {
        // New offset [1, -1, 6]
        bElem _cg_x011_reg;
        {
          _cg_x_111_vecbuf = _cg_x011_vecbuf;
          _cg_x011_vecbuf = x.dat[neighbor14 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x011_reg = _cg_vectmp0;
        }
        buf11[6] += _cg_x011_reg;
        buf11[7] -= _cg_x011_reg;
      }
      {
        // New offset [0, 0, 6]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[6] += _cg_x011_reg;
        buf12[6] += _cg_x011_reg;
        buf12[7] -= _cg_x011_reg;
      }
      {
        // New offset [0, -2, 7]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor19 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf12[7] -= _cg_x011_reg;
      }
      {
        // New offset [-1, -1, 7]
        bElem _cg_x011_reg;
        {
          _cg_x_111_vecbuf = x.dat[neighbor21 * x.step + hipThreadIdx_x];
          _cg_x011_vecbuf = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x011_reg = _cg_vectmp0;
        }
        buf11[7] -= _cg_x011_reg;
      }
      {
        // New offset [0, -1, 7]
        bElem _cg_x011_reg;
        {
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[7] -= _cg_x011_reg;
      }
      {
        // New offset [1, -1, 7]
        bElem _cg_x011_reg;
        {
          _cg_x_111_vecbuf = _cg_x011_vecbuf;
          _cg_x011_vecbuf = x.dat[neighbor23 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x011_reg = _cg_vectmp0;
        }
        buf11[7] += _cg_x011_reg;
      }
      {
        // New offset [0, 0, 7]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor25 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[7] += _cg_x011_reg;
        buf12[7] += _cg_x011_reg;
      }
    }
  }
  bElem buf13[8];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 0]
            buf13[0 + rel] = 0;
          }
          _cg_rel2 += 1;
        }
      }
    }
    {
      bElem _cg_beta_i010_vecbuf;
      bElem _cg_beta_i0_10_vecbuf;
      bElem _cg_beta_i01_2_vecbuf;
      bElem _cg_beta_i_110_vecbuf;
      bElem _cg_beta_i_1_10_vecbuf;
      bElem _cg_beta_i_11_2_vecbuf;
      bElem _cg_beta_j000_vecbuf;
      bElem _cg_beta_j100_vecbuf;
      bElem _cg_beta_j_100_vecbuf;
      bElem _cg_beta_j00_2_vecbuf;
      bElem _cg_beta_k000_vecbuf;
      bElem _cg_beta_k100_vecbuf;
      bElem _cg_beta_k_100_vecbuf;
      bElem _cg_beta_k0_20_vecbuf;
      {
        // New offset [0, 0, 0]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i0_10_reg;
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j_100_reg;
        bElem _cg_beta_k100_reg;
        bElem _cg_beta_k_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_i.dat[neighbor13 * beta_i.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_i.dat[neighbor10 * beta_i.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_i0_10_vecbuf
          dev_shl(_cg_beta_i0_10_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_i.dat[neighbor16 * beta_i.step + hipThreadIdx_x];
          // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_i010_vecbuf
          dev_shl(_cg_beta_i010_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          _cg_beta_j_100_vecbuf = beta_j.dat[neighbor12 * beta_j.step + hipThreadIdx_x];
          _cg_beta_j000_vecbuf = beta_j.dat[neighbor13 * beta_j.step + hipThreadIdx_x];
          _cg_beta_j100_vecbuf = beta_j.dat[neighbor14 * beta_j.step + hipThreadIdx_x];
          _cg_beta_k_100_vecbuf = beta_k.dat[neighbor12 * beta_k.step + hipThreadIdx_x];
          _cg_beta_k000_vecbuf = beta_k.dat[neighbor13 * beta_k.step + hipThreadIdx_x];
          _cg_beta_k100_vecbuf = beta_k.dat[neighbor14 * beta_k.step + hipThreadIdx_x];
          _cg_beta_i010_reg = _cg_beta_i010_vecbuf;
          _cg_beta_i0_10_reg = _cg_beta_i0_10_vecbuf;
          bElem _cg_vectmp3;
          // merge0 _cg_beta_j000_vecbuf ,_cg_beta_j100_vecbuf, 1 -> _cg_vectmp3
          dev_shl(_cg_vectmp3, _cg_beta_j000_vecbuf, _cg_beta_j100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_j100_reg = _cg_vectmp3;
          bElem _cg_vectmp4;
          // merge0 _cg_beta_j_100_vecbuf ,_cg_beta_j000_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_beta_j_100_vecbuf, _cg_beta_j000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_j_100_reg = _cg_vectmp4;
          bElem _cg_vectmp5;
          // merge0 _cg_beta_k000_vecbuf ,_cg_beta_k100_vecbuf, 1 -> _cg_vectmp5
          dev_shl(_cg_vectmp5, _cg_beta_k000_vecbuf, _cg_beta_k100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_k100_reg = _cg_vectmp5;
          bElem _cg_vectmp6;
          // merge0 _cg_beta_k_100_vecbuf ,_cg_beta_k000_vecbuf, 7 -> _cg_vectmp6
          dev_shl(_cg_vectmp6, _cg_beta_k_100_vecbuf, _cg_beta_k000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_k_100_reg = _cg_vectmp6;
        }
        buf13[0] += (_cg_beta_i010_reg - _cg_beta_i0_10_reg) * buf5[0];
        buf13[0] += (_cg_beta_j100_reg - _cg_beta_j_100_reg) * buf3[0];
        buf13[0] += (_cg_beta_k100_reg - _cg_beta_k_100_reg) * buf1[0];
      }
      {
        // New offset [1, 0, 0]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i0_10_reg;
        {
          _cg_beta_i_1_10_vecbuf = _cg_beta_i0_10_vecbuf;
          _cg_beta_i_110_vecbuf = _cg_beta_i010_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_i.dat[neighbor14 * beta_i.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_i.dat[neighbor11 * beta_i.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_i0_10_vecbuf
          dev_shl(_cg_beta_i0_10_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_i.dat[neighbor17 * beta_i.step + hipThreadIdx_x];
          // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_i010_vecbuf
          dev_shl(_cg_beta_i010_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp3;
          // merge0 _cg_beta_i_110_vecbuf ,_cg_beta_i010_vecbuf, 1 -> _cg_vectmp3
          dev_shl(_cg_vectmp3, _cg_beta_i_110_vecbuf, _cg_beta_i010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i010_reg = _cg_vectmp3;
          bElem _cg_vectmp4;
          // merge0 _cg_beta_i_1_10_vecbuf ,_cg_beta_i0_10_vecbuf, 1 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_beta_i_1_10_vecbuf, _cg_beta_i0_10_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i0_10_reg = _cg_vectmp4;
        }
        buf13[0] += (_cg_beta_i010_reg - _cg_beta_i0_10_reg) * buf7[0];
      }
      {
        // New offset [-1, 1, 0]
        bElem _cg_beta_k100_reg;
        bElem _cg_beta_k1_20_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_k.dat[neighbor13 * beta_k.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_k.dat[neighbor10 * beta_k.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_k0_20_vecbuf
          dev_shl(_cg_beta_k0_20_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_k.dat[neighbor16 * beta_k.step + hipThreadIdx_x];
          // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_k000_vecbuf
          dev_shl(_cg_beta_k000_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
          _cg_beta_k1_20_reg = _cg_beta_k0_20_vecbuf;
        }
        buf13[0] += (_cg_beta_k100_reg - _cg_beta_k1_20_reg) * buf0[0];
      }
      {
        // New offset [0, 1, 0]
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_j.dat[neighbor15 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_j.dat[neighbor12 * beta_j.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j_100_vecbuf
          dev_shl(_cg_beta_j_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_j.dat[neighbor16 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = beta_j.dat[neighbor13 * beta_j.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_beta_j000_vecbuf
          dev_shl(_cg_beta_j000_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          _cg_vectmp4 = beta_j.dat[neighbor17 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp5;
          _cg_vectmp5 = beta_j.dat[neighbor14 * beta_j.step + hipThreadIdx_x];
          // merge1 _cg_vectmp5 ,_cg_vectmp4, 1 -> _cg_beta_j100_vecbuf
          dev_shl(_cg_beta_j100_vecbuf, _cg_vectmp5, _cg_vectmp4, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp6;
          // merge0 _cg_beta_j000_vecbuf ,_cg_beta_j100_vecbuf, 1 -> _cg_vectmp6
          dev_shl(_cg_vectmp6, _cg_beta_j000_vecbuf, _cg_beta_j100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_j100_reg = _cg_vectmp6;
          bElem _cg_vectmp7;
          // merge0 _cg_beta_j_100_vecbuf ,_cg_beta_j000_vecbuf, 7 -> _cg_vectmp7
          dev_shl(_cg_vectmp7, _cg_beta_j_100_vecbuf, _cg_beta_j000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_j_100_reg = _cg_vectmp7;
        }
        buf13[0] += (_cg_beta_j100_reg - _cg_beta_j_100_reg) * buf9[0];
      }
      {
        // New offset [0, -1, 1]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i01_2_reg;
        {
          _cg_beta_i01_2_vecbuf = beta_i.dat[neighbor4 * beta_i.step + 448 + hipThreadIdx_x];
          _cg_beta_i010_vecbuf = beta_i.dat[neighbor13 * beta_i.step + 64 + hipThreadIdx_x];
          _cg_beta_i010_reg = _cg_beta_i010_vecbuf;
          _cg_beta_i01_2_reg = _cg_beta_i01_2_vecbuf;
        }
        buf13[0] += (_cg_beta_i010_reg - _cg_beta_i01_2_reg) * buf4[0];
      }
      {
        // New offset [1, -1, 1]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i01_2_reg;
        {
          _cg_beta_i_11_2_vecbuf = _cg_beta_i01_2_vecbuf;
          _cg_beta_i_110_vecbuf = _cg_beta_i010_vecbuf;
          _cg_beta_i01_2_vecbuf = beta_i.dat[neighbor5 * beta_i.step + 448 + hipThreadIdx_x];
          _cg_beta_i010_vecbuf = beta_i.dat[neighbor14 * beta_i.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_beta_i_110_vecbuf ,_cg_beta_i010_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_beta_i_110_vecbuf, _cg_beta_i010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i010_reg = _cg_vectmp0;
          bElem _cg_vectmp1;
          // merge0 _cg_beta_i_11_2_vecbuf ,_cg_beta_i01_2_vecbuf, 1 -> _cg_vectmp1
          dev_shl(_cg_vectmp1, _cg_beta_i_11_2_vecbuf, _cg_beta_i01_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i01_2_reg = _cg_vectmp1;
        }
        buf13[0] += (_cg_beta_i010_reg - _cg_beta_i01_2_reg) * buf8[0];
      }
      {
        // New offset [-1, 0, 1]
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j10_2_reg;
        {
          _cg_beta_j00_2_vecbuf = beta_j.dat[neighbor4 * beta_j.step + 448 + hipThreadIdx_x];
          _cg_beta_j000_vecbuf = beta_j.dat[neighbor13 * beta_j.step + 64 + hipThreadIdx_x];
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_beta_j10_2_reg = _cg_beta_j00_2_vecbuf;
        }
        buf13[0] += (_cg_beta_j100_reg - _cg_beta_j10_2_reg) * buf2[0];
      }
      {
        // New offset [0, 0, 1]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i0_10_reg;
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j_100_reg;
        bElem _cg_beta_k100_reg;
        bElem _cg_beta_k_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_i.dat[neighbor13 * beta_i.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_i.dat[neighbor10 * beta_i.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_i0_10_vecbuf
          dev_shl(_cg_beta_i0_10_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_i.dat[neighbor16 * beta_i.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_i010_vecbuf
          dev_shl(_cg_beta_i010_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          _cg_beta_j_100_vecbuf = beta_j.dat[neighbor12 * beta_j.step + 64 + hipThreadIdx_x];
          _cg_beta_j100_vecbuf = beta_j.dat[neighbor14 * beta_j.step + 64 + hipThreadIdx_x];
          _cg_beta_k_100_vecbuf = beta_k.dat[neighbor12 * beta_k.step + 64 + hipThreadIdx_x];
          _cg_beta_k000_vecbuf = beta_k.dat[neighbor13 * beta_k.step + 64 + hipThreadIdx_x];
          _cg_beta_k100_vecbuf = beta_k.dat[neighbor14 * beta_k.step + 64 + hipThreadIdx_x];
          _cg_beta_i010_reg = _cg_beta_i010_vecbuf;
          _cg_beta_i0_10_reg = _cg_beta_i0_10_vecbuf;
          bElem _cg_vectmp3;
          // merge0 _cg_beta_j000_vecbuf ,_cg_beta_j100_vecbuf, 1 -> _cg_vectmp3
          dev_shl(_cg_vectmp3, _cg_beta_j000_vecbuf, _cg_beta_j100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_j100_reg = _cg_vectmp3;
          bElem _cg_vectmp4;
          // merge0 _cg_beta_j_100_vecbuf ,_cg_beta_j000_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_beta_j_100_vecbuf, _cg_beta_j000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_j_100_reg = _cg_vectmp4;
          bElem _cg_vectmp5;
          // merge0 _cg_beta_k000_vecbuf ,_cg_beta_k100_vecbuf, 1 -> _cg_vectmp5
          dev_shl(_cg_vectmp5, _cg_beta_k000_vecbuf, _cg_beta_k100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_k100_reg = _cg_vectmp5;
          bElem _cg_vectmp6;
          // merge0 _cg_beta_k_100_vecbuf ,_cg_beta_k000_vecbuf, 7 -> _cg_vectmp6
          dev_shl(_cg_vectmp6, _cg_beta_k_100_vecbuf, _cg_beta_k000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_k_100_reg = _cg_vectmp6;
        }
        buf13[1] += (_cg_beta_i010_reg - _cg_beta_i0_10_reg) * buf5[1];
        buf13[1] += (_cg_beta_j100_reg - _cg_beta_j_100_reg) * buf3[1];
        buf13[1] += (_cg_beta_k100_reg - _cg_beta_k_100_reg) * buf1[1];
        buf13[0] += (_cg_beta_k100_reg - _cg_beta_k_100_reg) * buf11[0];
      }
      {
        // New offset [1, 0, 1]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i0_10_reg;
        {
          _cg_beta_i_1_10_vecbuf = _cg_beta_i0_10_vecbuf;
          _cg_beta_i_110_vecbuf = _cg_beta_i010_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_i.dat[neighbor14 * beta_i.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_i.dat[neighbor11 * beta_i.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_i0_10_vecbuf
          dev_shl(_cg_beta_i0_10_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_i.dat[neighbor17 * beta_i.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_i010_vecbuf
          dev_shl(_cg_beta_i010_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp3;
          // merge0 _cg_beta_i_110_vecbuf ,_cg_beta_i010_vecbuf, 1 -> _cg_vectmp3
          dev_shl(_cg_vectmp3, _cg_beta_i_110_vecbuf, _cg_beta_i010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i010_reg = _cg_vectmp3;
          bElem _cg_vectmp4;
          // merge0 _cg_beta_i_1_10_vecbuf ,_cg_beta_i0_10_vecbuf, 1 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_beta_i_1_10_vecbuf, _cg_beta_i0_10_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i0_10_reg = _cg_vectmp4;
        }
        buf13[1] += (_cg_beta_i010_reg - _cg_beta_i0_10_reg) * buf7[1];
      }
      {
        // New offset [-1, 1, 1]
        bElem _cg_beta_k100_reg;
        bElem _cg_beta_k1_20_reg;
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j10_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_k.dat[neighbor13 * beta_k.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_k.dat[neighbor10 * beta_k.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_k0_20_vecbuf
          dev_shl(_cg_beta_k0_20_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_k.dat[neighbor16 * beta_k.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_k000_vecbuf
          dev_shl(_cg_beta_k000_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp3;
          _cg_vectmp3 = beta_j.dat[neighbor7 * beta_j.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp4;
          _cg_vectmp4 = beta_j.dat[neighbor4 * beta_j.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp4 ,_cg_vectmp3, 1 -> _cg_beta_j00_2_vecbuf
          dev_shl(_cg_beta_j00_2_vecbuf, _cg_vectmp4, _cg_vectmp3, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp5;
          _cg_vectmp5 = beta_j.dat[neighbor16 * beta_j.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp6;
          _cg_vectmp6 = beta_j.dat[neighbor13 * beta_j.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp6 ,_cg_vectmp5, 1 -> _cg_beta_j000_vecbuf
          dev_shl(_cg_beta_j000_vecbuf, _cg_vectmp6, _cg_vectmp5, 56, 64, hipThreadIdx_x);
          _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
          _cg_beta_k1_20_reg = _cg_beta_k0_20_vecbuf;
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_beta_j10_2_reg = _cg_beta_j00_2_vecbuf;
        }
        buf13[1] += (_cg_beta_k100_reg - _cg_beta_k1_20_reg) * buf0[1];
        buf13[0] += (_cg_beta_j100_reg - _cg_beta_j10_2_reg) * buf10[0];
        buf13[0] += (_cg_beta_k100_reg - _cg_beta_k1_20_reg) * buf12[0];
      }
      {
        // New offset [0, 1, 1]
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_j.dat[neighbor15 * beta_j.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_j.dat[neighbor12 * beta_j.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j_100_vecbuf
          dev_shl(_cg_beta_j_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_j.dat[neighbor17 * beta_j.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = beta_j.dat[neighbor14 * beta_j.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_beta_j100_vecbuf
          dev_shl(_cg_beta_j100_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_beta_j000_vecbuf ,_cg_beta_j100_vecbuf, 1 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_beta_j000_vecbuf, _cg_beta_j100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_j100_reg = _cg_vectmp4;
          bElem _cg_vectmp5;
          // merge0 _cg_beta_j_100_vecbuf ,_cg_beta_j000_vecbuf, 7 -> _cg_vectmp5
          dev_shl(_cg_vectmp5, _cg_beta_j_100_vecbuf, _cg_beta_j000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_j_100_reg = _cg_vectmp5;
        }
        buf13[1] += (_cg_beta_j100_reg - _cg_beta_j_100_reg) * buf9[1];
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 6; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, -1, 2]
            bElem _cg_beta_i010_reg;
            bElem _cg_beta_i01_2_reg;
            {
              _cg_beta_i01_2_vecbuf = beta_i.dat[neighbor13 * beta_i.step + (hipThreadIdx_x + rel * 64)];
              _cg_beta_i010_vecbuf = beta_i.dat[neighbor13 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_i010_reg = _cg_beta_i010_vecbuf;
              _cg_beta_i01_2_reg = _cg_beta_i01_2_vecbuf;
            }
            buf13[1 + rel] += (_cg_beta_i010_reg - _cg_beta_i01_2_reg) * buf4[1 + rel];
          }
          {
            // New offset [1, -1, 2]
            bElem _cg_beta_i010_reg;
            bElem _cg_beta_i01_2_reg;
            {
              _cg_beta_i_11_2_vecbuf = _cg_beta_i01_2_vecbuf;
              _cg_beta_i_110_vecbuf = _cg_beta_i010_vecbuf;
              _cg_beta_i01_2_vecbuf = beta_i.dat[neighbor14 * beta_i.step + (hipThreadIdx_x + rel * 64)];
              _cg_beta_i010_vecbuf = beta_i.dat[neighbor14 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_beta_i_110_vecbuf ,_cg_beta_i010_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_beta_i_110_vecbuf, _cg_beta_i010_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_i010_reg = _cg_vectmp0;
              bElem _cg_vectmp1;
              // merge0 _cg_beta_i_11_2_vecbuf ,_cg_beta_i01_2_vecbuf, 1 -> _cg_vectmp1
              dev_shl(_cg_vectmp1, _cg_beta_i_11_2_vecbuf, _cg_beta_i01_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_i01_2_reg = _cg_vectmp1;
            }
            buf13[1 + rel] += (_cg_beta_i010_reg - _cg_beta_i01_2_reg) * buf8[1 + rel];
          }
          {
            // New offset [-1, 0, 2]
            bElem _cg_beta_j100_reg;
            bElem _cg_beta_j10_2_reg;
            {
              _cg_beta_j00_2_vecbuf = beta_j.dat[neighbor13 * beta_j.step + (hipThreadIdx_x + rel * 64)];
              _cg_beta_j000_vecbuf = beta_j.dat[neighbor13 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
              _cg_beta_j10_2_reg = _cg_beta_j00_2_vecbuf;
            }
            buf13[1 + rel] += (_cg_beta_j100_reg - _cg_beta_j10_2_reg) * buf2[1 + rel];
          }
          {
            // New offset [0, 0, 2]
            bElem _cg_beta_i010_reg;
            bElem _cg_beta_i0_10_reg;
            bElem _cg_beta_j100_reg;
            bElem _cg_beta_j_100_reg;
            bElem _cg_beta_k100_reg;
            bElem _cg_beta_k_100_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = beta_i.dat[neighbor13 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = beta_i.dat[neighbor10 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_i0_10_vecbuf
              dev_shl(_cg_beta_i0_10_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = beta_i.dat[neighbor16 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_i010_vecbuf
              dev_shl(_cg_beta_i010_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
              _cg_beta_j_100_vecbuf = beta_j.dat[neighbor12 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_j100_vecbuf = beta_j.dat[neighbor14 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_k_100_vecbuf = beta_k.dat[neighbor12 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_k000_vecbuf = beta_k.dat[neighbor13 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_k100_vecbuf = beta_k.dat[neighbor14 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_i010_reg = _cg_beta_i010_vecbuf;
              _cg_beta_i0_10_reg = _cg_beta_i0_10_vecbuf;
              bElem _cg_vectmp3;
              // merge0 _cg_beta_j000_vecbuf ,_cg_beta_j100_vecbuf, 1 -> _cg_vectmp3
              dev_shl(_cg_vectmp3, _cg_beta_j000_vecbuf, _cg_beta_j100_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_j100_reg = _cg_vectmp3;
              bElem _cg_vectmp4;
              // merge0 _cg_beta_j_100_vecbuf ,_cg_beta_j000_vecbuf, 7 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_beta_j_100_vecbuf, _cg_beta_j000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_beta_j_100_reg = _cg_vectmp4;
              bElem _cg_vectmp5;
              // merge0 _cg_beta_k000_vecbuf ,_cg_beta_k100_vecbuf, 1 -> _cg_vectmp5
              dev_shl(_cg_vectmp5, _cg_beta_k000_vecbuf, _cg_beta_k100_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_k100_reg = _cg_vectmp5;
              bElem _cg_vectmp6;
              // merge0 _cg_beta_k_100_vecbuf ,_cg_beta_k000_vecbuf, 7 -> _cg_vectmp6
              dev_shl(_cg_vectmp6, _cg_beta_k_100_vecbuf, _cg_beta_k000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_beta_k_100_reg = _cg_vectmp6;
            }
            buf13[2 + rel] += (_cg_beta_i010_reg - _cg_beta_i0_10_reg) * buf5[2 + rel];
            buf13[2 + rel] += (_cg_beta_j100_reg - _cg_beta_j_100_reg) * buf3[2 + rel];
            buf13[2 + rel] += (_cg_beta_k100_reg - _cg_beta_k_100_reg) * buf1[2 + rel];
            buf13[1 + rel] += (_cg_beta_k100_reg - _cg_beta_k_100_reg) * buf11[1 + rel];
          }
          {
            // New offset [1, 0, 2]
            bElem _cg_beta_i010_reg;
            bElem _cg_beta_i0_10_reg;
            {
              _cg_beta_i_1_10_vecbuf = _cg_beta_i0_10_vecbuf;
              _cg_beta_i_110_vecbuf = _cg_beta_i010_vecbuf;
              bElem _cg_vectmp0;
              _cg_vectmp0 = beta_i.dat[neighbor14 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = beta_i.dat[neighbor11 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_i0_10_vecbuf
              dev_shl(_cg_beta_i0_10_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = beta_i.dat[neighbor17 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_i010_vecbuf
              dev_shl(_cg_beta_i010_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp3;
              // merge0 _cg_beta_i_110_vecbuf ,_cg_beta_i010_vecbuf, 1 -> _cg_vectmp3
              dev_shl(_cg_vectmp3, _cg_beta_i_110_vecbuf, _cg_beta_i010_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_i010_reg = _cg_vectmp3;
              bElem _cg_vectmp4;
              // merge0 _cg_beta_i_1_10_vecbuf ,_cg_beta_i0_10_vecbuf, 1 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_beta_i_1_10_vecbuf, _cg_beta_i0_10_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_i0_10_reg = _cg_vectmp4;
            }
            buf13[2 + rel] += (_cg_beta_i010_reg - _cg_beta_i0_10_reg) * buf7[2 + rel];
          }
          {
            // New offset [-1, 1, 2]
            bElem _cg_beta_k100_reg;
            bElem _cg_beta_k1_20_reg;
            bElem _cg_beta_j100_reg;
            bElem _cg_beta_j10_2_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = beta_k.dat[neighbor13 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = beta_k.dat[neighbor10 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_k0_20_vecbuf
              dev_shl(_cg_beta_k0_20_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = beta_k.dat[neighbor16 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_k000_vecbuf
              dev_shl(_cg_beta_k000_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp3;
              _cg_vectmp3 = beta_j.dat[neighbor16 * beta_j.step + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp4;
              _cg_vectmp4 = beta_j.dat[neighbor13 * beta_j.step + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp4 ,_cg_vectmp3, 1 -> _cg_beta_j00_2_vecbuf
              dev_shl(_cg_beta_j00_2_vecbuf, _cg_vectmp4, _cg_vectmp3, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp5;
              _cg_vectmp5 = beta_j.dat[neighbor16 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp6;
              _cg_vectmp6 = beta_j.dat[neighbor13 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp6 ,_cg_vectmp5, 1 -> _cg_beta_j000_vecbuf
              dev_shl(_cg_beta_j000_vecbuf, _cg_vectmp6, _cg_vectmp5, 56, 64, hipThreadIdx_x);
              _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
              _cg_beta_k1_20_reg = _cg_beta_k0_20_vecbuf;
              _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
              _cg_beta_j10_2_reg = _cg_beta_j00_2_vecbuf;
            }
            buf13[2 + rel] += (_cg_beta_k100_reg - _cg_beta_k1_20_reg) * buf0[2 + rel];
            buf13[1 + rel] += (_cg_beta_j100_reg - _cg_beta_j10_2_reg) * buf10[1 + rel];
            buf13[1 + rel] += (_cg_beta_k100_reg - _cg_beta_k1_20_reg) * buf12[1 + rel];
          }
          {
            // New offset [0, 1, 2]
            bElem _cg_beta_j100_reg;
            bElem _cg_beta_j_100_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = beta_j.dat[neighbor15 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = beta_j.dat[neighbor12 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j_100_vecbuf
              dev_shl(_cg_beta_j_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = beta_j.dat[neighbor17 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = beta_j.dat[neighbor14 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_beta_j100_vecbuf
              dev_shl(_cg_beta_j100_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_beta_j000_vecbuf ,_cg_beta_j100_vecbuf, 1 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_beta_j000_vecbuf, _cg_beta_j100_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_j100_reg = _cg_vectmp4;
              bElem _cg_vectmp5;
              // merge0 _cg_beta_j_100_vecbuf ,_cg_beta_j000_vecbuf, 7 -> _cg_vectmp5
              dev_shl(_cg_vectmp5, _cg_beta_j_100_vecbuf, _cg_beta_j000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_beta_j_100_reg = _cg_vectmp5;
            }
            buf13[2 + rel] += (_cg_beta_j100_reg - _cg_beta_j_100_reg) * buf9[2 + rel];
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [0, -1, 8]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i01_2_reg;
        {
          _cg_beta_i01_2_vecbuf = beta_i.dat[neighbor13 * beta_i.step + 384 + hipThreadIdx_x];
          _cg_beta_i010_vecbuf = beta_i.dat[neighbor22 * beta_i.step + hipThreadIdx_x];
          _cg_beta_i010_reg = _cg_beta_i010_vecbuf;
          _cg_beta_i01_2_reg = _cg_beta_i01_2_vecbuf;
        }
        buf13[7] += (_cg_beta_i010_reg - _cg_beta_i01_2_reg) * buf4[7];
      }
      {
        // New offset [1, -1, 8]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i01_2_reg;
        {
          _cg_beta_i_11_2_vecbuf = _cg_beta_i01_2_vecbuf;
          _cg_beta_i_110_vecbuf = _cg_beta_i010_vecbuf;
          _cg_beta_i01_2_vecbuf = beta_i.dat[neighbor14 * beta_i.step + 384 + hipThreadIdx_x];
          _cg_beta_i010_vecbuf = beta_i.dat[neighbor23 * beta_i.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_beta_i_110_vecbuf ,_cg_beta_i010_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_beta_i_110_vecbuf, _cg_beta_i010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i010_reg = _cg_vectmp0;
          bElem _cg_vectmp1;
          // merge0 _cg_beta_i_11_2_vecbuf ,_cg_beta_i01_2_vecbuf, 1 -> _cg_vectmp1
          dev_shl(_cg_vectmp1, _cg_beta_i_11_2_vecbuf, _cg_beta_i01_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i01_2_reg = _cg_vectmp1;
        }
        buf13[7] += (_cg_beta_i010_reg - _cg_beta_i01_2_reg) * buf8[7];
      }
      {
        // New offset [-1, 0, 8]
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j10_2_reg;
        {
          _cg_beta_j00_2_vecbuf = beta_j.dat[neighbor13 * beta_j.step + 384 + hipThreadIdx_x];
          _cg_beta_j000_vecbuf = beta_j.dat[neighbor22 * beta_j.step + hipThreadIdx_x];
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_beta_j10_2_reg = _cg_beta_j00_2_vecbuf;
        }
        buf13[7] += (_cg_beta_j100_reg - _cg_beta_j10_2_reg) * buf2[7];
      }
      {
        // New offset [0, 0, 8]
        bElem _cg_beta_k100_reg;
        bElem _cg_beta_k_100_reg;
        {
          _cg_beta_k_100_vecbuf = beta_k.dat[neighbor21 * beta_k.step + hipThreadIdx_x];
          _cg_beta_k000_vecbuf = beta_k.dat[neighbor22 * beta_k.step + hipThreadIdx_x];
          _cg_beta_k100_vecbuf = beta_k.dat[neighbor23 * beta_k.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_beta_k000_vecbuf ,_cg_beta_k100_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_beta_k000_vecbuf, _cg_beta_k100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_k100_reg = _cg_vectmp0;
          bElem _cg_vectmp1;
          // merge0 _cg_beta_k_100_vecbuf ,_cg_beta_k000_vecbuf, 7 -> _cg_vectmp1
          dev_shl(_cg_vectmp1, _cg_beta_k_100_vecbuf, _cg_beta_k000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_k_100_reg = _cg_vectmp1;
        }
        buf13[7] += (_cg_beta_k100_reg - _cg_beta_k_100_reg) * buf11[7];
      }
      {
        // New offset [-1, 1, 8]
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j10_2_reg;
        bElem _cg_beta_k100_reg;
        bElem _cg_beta_k1_20_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_j.dat[neighbor16 * beta_j.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_j.dat[neighbor13 * beta_j.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j00_2_vecbuf
          dev_shl(_cg_beta_j00_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_j.dat[neighbor25 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = beta_j.dat[neighbor22 * beta_j.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_beta_j000_vecbuf
          dev_shl(_cg_beta_j000_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          _cg_vectmp4 = beta_k.dat[neighbor22 * beta_k.step + hipThreadIdx_x];
          bElem _cg_vectmp5;
          _cg_vectmp5 = beta_k.dat[neighbor19 * beta_k.step + hipThreadIdx_x];
          // merge1 _cg_vectmp5 ,_cg_vectmp4, 7 -> _cg_beta_k0_20_vecbuf
          dev_shl(_cg_beta_k0_20_vecbuf, _cg_vectmp5, _cg_vectmp4, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp6;
          _cg_vectmp6 = beta_k.dat[neighbor25 * beta_k.step + hipThreadIdx_x];
          // merge1 _cg_vectmp4 ,_cg_vectmp6, 1 -> _cg_beta_k000_vecbuf
          dev_shl(_cg_beta_k000_vecbuf, _cg_vectmp4, _cg_vectmp6, 56, 64, hipThreadIdx_x);
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_beta_j10_2_reg = _cg_beta_j00_2_vecbuf;
          _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
          _cg_beta_k1_20_reg = _cg_beta_k0_20_vecbuf;
        }
        buf13[7] += (_cg_beta_j100_reg - _cg_beta_j10_2_reg) * buf10[7];
        buf13[7] += (_cg_beta_k100_reg - _cg_beta_k1_20_reg) * buf12[7];
      }
    }
  }
  {
    {
      bElem _cg_alpha000_vecbuf;
      bElem _cg_x000_vecbuf;
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 0]
            bElem _cg_alpha000_reg;
            bElem _cg_x000_reg;
            {
              _cg_alpha000_vecbuf = alpha.dat[neighbor13 * alpha.step + (hipThreadIdx_x + rel * 64)];
              _cg_x000_vecbuf = x.dat[neighbor13 * x.step + (hipThreadIdx_x + rel * 64)];
              _cg_alpha000_reg = _cg_alpha000_vecbuf;
              _cg_x000_reg = _cg_x000_vecbuf;
            }
            buf12[0 + rel] = c[0] * _cg_alpha000_reg * _cg_x000_reg - c[1] * c[2] * (0.0833 * buf6[0 + rel] + 0.020825 * buf13[0 + rel]);
          }
          _cg_rel2 += 1;
        }
      }
    }
    bElem *out_ref = &out.dat[neighbor13 * out.step];
    for (long sti = 0; sti < 8; ++sti)
    {
      out_ref[sti * 64 + hipThreadIdx_x] = buf12[sti];
    }
  }
}
# 306 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/helmholtz4/intermediate_gen/helmholtz4.cu" 2

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
# 1 "VSTile-helmholtz45.py-HIP-8x8x64" 1
{
  bElem buf0[64];
  bElem buf1[64];
  bElem buf2[64];
  bElem buf3[64];
  bElem buf4[64];
  bElem buf5[64];
  bElem buf6[64];
  bElem buf7[64];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf0[0 + rel] = 0;
                buf1[0 + rel] = 0;
                buf2[0 + rel] = 0;
                buf3[0 + rel] = 0;
                buf4[0 + rel] = 0;
                buf5[0 + rel] = 0;
                buf6[0 + rel] = 0;
                buf7[0 + rel] = 0;
              }
              _cg_rel1 += 1;
            }
          }
          _cg_rel2 += 8;
        }
      }
    }
    {
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 7; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf2[0 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + -2);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf2[7] -= x(i + hipThreadIdx_x, j + 7, k + -2);
      }
      {
        buf5[0] -= x(i + hipThreadIdx_x + -1, j, k + -1);
      }
      {
        buf5[1] -= x(i + hipThreadIdx_x + -1, j + 1, k + -1);
        buf7[0] -= x(i + hipThreadIdx_x, j + -1, k + -1);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf2[0 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf2[8 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf3[0 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf5[0 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf5[2 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + -1);
            buf7[0 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf7[1 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf2[6] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + -1);
        buf2[14] -= x(i + hipThreadIdx_x, j + 6, k + -1);
        buf3[6] += x(i + hipThreadIdx_x, j + 6, k + -1);
        buf5[6] += x(i + hipThreadIdx_x, j + 6, k + -1);
        buf7[6] += x(i + hipThreadIdx_x, j + 6, k + -1);
        buf7[7] -= x(i + hipThreadIdx_x, j + 6, k + -1);
      }
      {
        buf2[7] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + -1);
        buf2[15] -= x(i + hipThreadIdx_x, j + 7, k + -1);
        buf3[7] += x(i + hipThreadIdx_x, j + 7, k + -1);
        buf5[7] += x(i + hipThreadIdx_x, j + 7, k + -1);
        buf7[7] += x(i + hipThreadIdx_x, j + 7, k + -1);
      }
      {
        buf4[0] -= x(i + hipThreadIdx_x + -1, j + -1, k);
        buf6[0] -= x(i + hipThreadIdx_x + -1, j + -1, k);
      }
      {
        buf4[1] -= x(i + hipThreadIdx_x + -1, j, k);
        buf6[1] -= x(i + hipThreadIdx_x + -1, j, k);
        buf6[0] += x(i + hipThreadIdx_x + -1, j, k);
        buf5[8] -= x(i + hipThreadIdx_x + -1, j, k);
      }
      {
        buf0[0] += 15.0 * x(i + hipThreadIdx_x, j + -1, k);
        buf0[0] -= x(i + hipThreadIdx_x, j + -1, k);
        buf1[0] += x(i + hipThreadIdx_x, j + -1, k);
        buf4[0] += x(i + hipThreadIdx_x + -1, j + 1, k);
        buf4[0] += x(i + hipThreadIdx_x, j + -1, k);
        buf4[2] -= x(i + hipThreadIdx_x + -1, j + 1, k);
        buf6[0] += x(i + hipThreadIdx_x + 1, j + -1, k);
        buf6[2] -= x(i + hipThreadIdx_x + -1, j + 1, k);
        buf6[1] += x(i + hipThreadIdx_x + -1, j + 1, k);
        buf5[9] -= x(i + hipThreadIdx_x + -1, j + 1, k);
        buf7[8] -= x(i + hipThreadIdx_x, j + -1, k);
      }
      {
        buf0[1] += 15.0 * x(i + hipThreadIdx_x, j, k);
        buf0[1] -= x(i + hipThreadIdx_x, j, k);
        buf0[0] -= 15.0 * x(i + hipThreadIdx_x, j, k);
        buf1[1] += x(i + hipThreadIdx_x, j, k);
        buf1[0] -= 15.0 * x(i + hipThreadIdx_x, j, k);
        buf4[1] += x(i + hipThreadIdx_x + -1, j + 2, k);
        buf4[1] += x(i + hipThreadIdx_x, j, k);
        buf4[3] -= x(i + hipThreadIdx_x + -1, j + 2, k);
        buf6[1] += x(i + hipThreadIdx_x + 1, j, k);
        buf6[0] -= x(i + hipThreadIdx_x + 1, j, k);
        buf6[3] -= x(i + hipThreadIdx_x + -1, j + 2, k);
        buf6[2] += x(i + hipThreadIdx_x + -1, j + 2, k);
        buf2[0] -= 15.0 * x(i + hipThreadIdx_x, j, k);
        buf2[8] += 15.0 * x(i + hipThreadIdx_x, j, k);
        buf2[16] -= x(i + hipThreadIdx_x, j, k);
        buf3[0] -= 15.0 * x(i + hipThreadIdx_x, j, k);
        buf3[8] += x(i + hipThreadIdx_x, j, k);
        buf5[8] += x(i + hipThreadIdx_x, j, k);
        buf5[10] -= x(i + hipThreadIdx_x + -1, j + 2, k);
        buf7[8] += x(i + hipThreadIdx_x, j, k);
        buf7[9] -= x(i + hipThreadIdx_x, j, k);
      }
      {
        buf0[2] += 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf0[2] -= x(i + hipThreadIdx_x, j + 1, k);
        buf0[1] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf0[0] += x(i + hipThreadIdx_x, j + 1, k);
        buf1[2] += x(i + hipThreadIdx_x, j + 1, k);
        buf1[1] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf1[0] += 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf4[2] += x(i + hipThreadIdx_x + -1, j + 3, k);
        buf4[2] += x(i + hipThreadIdx_x, j + 1, k);
        buf4[0] -= x(i + hipThreadIdx_x, j + 1, k);
        buf4[4] -= x(i + hipThreadIdx_x + -1, j + 3, k);
        buf6[2] += x(i + hipThreadIdx_x + 1, j + 1, k);
        buf6[1] -= x(i + hipThreadIdx_x + 1, j + 1, k);
        buf6[4] -= x(i + hipThreadIdx_x + -1, j + 3, k);
        buf6[3] += x(i + hipThreadIdx_x + -1, j + 3, k);
        buf2[1] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf2[9] += 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf2[17] -= x(i + hipThreadIdx_x, j + 1, k);
        buf3[1] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k);
        buf3[9] += x(i + hipThreadIdx_x, j + 1, k);
        buf5[9] += x(i + hipThreadIdx_x, j + 1, k);
        buf5[11] -= x(i + hipThreadIdx_x + -1, j + 3, k);
        buf7[9] += x(i + hipThreadIdx_x, j + 1, k);
        buf7[10] -= x(i + hipThreadIdx_x, j + 1, k);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 3; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[3 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf0[3 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf0[2 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf0[1 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf1[3 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf1[2 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf1[1 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf1[0 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf4[3 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k);
            buf4[3 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf4[1 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf4[5 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k);
            buf6[3 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k);
            buf6[2 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k);
            buf6[5 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k);
            buf6[4 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k);
            buf2[2 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf2[10 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf2[18 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf3[2 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf3[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf5[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf5[12 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k);
            buf7[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf7[11 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[6] += 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf0[6] -= x(i + hipThreadIdx_x, j + 5, k);
        buf0[5] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf0[4] += x(i + hipThreadIdx_x, j + 5, k);
        buf1[6] += x(i + hipThreadIdx_x, j + 5, k);
        buf1[5] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf1[4] += 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf1[3] -= x(i + hipThreadIdx_x, j + 5, k);
        buf4[6] += x(i + hipThreadIdx_x + -1, j + 7, k);
        buf4[6] += x(i + hipThreadIdx_x, j + 5, k);
        buf4[4] -= x(i + hipThreadIdx_x, j + 5, k);
        buf6[6] += x(i + hipThreadIdx_x + 1, j + 5, k);
        buf6[5] -= x(i + hipThreadIdx_x + 1, j + 5, k);
        buf6[7] += x(i + hipThreadIdx_x + -1, j + 7, k);
        buf2[5] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf2[13] += 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf2[21] -= x(i + hipThreadIdx_x, j + 5, k);
        buf3[5] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k);
        buf3[13] += x(i + hipThreadIdx_x, j + 5, k);
        buf5[13] += x(i + hipThreadIdx_x, j + 5, k);
        buf5[15] -= x(i + hipThreadIdx_x + -1, j + 7, k);
        buf7[13] += x(i + hipThreadIdx_x, j + 5, k);
        buf7[14] -= x(i + hipThreadIdx_x, j + 5, k);
      }
      {
        buf0[7] += 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf0[7] -= x(i + hipThreadIdx_x, j + 6, k);
        buf0[6] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf0[5] += x(i + hipThreadIdx_x, j + 6, k);
        buf1[7] += x(i + hipThreadIdx_x, j + 6, k);
        buf1[6] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf1[5] += 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf1[4] -= x(i + hipThreadIdx_x, j + 6, k);
        buf4[7] += x(i + hipThreadIdx_x + -1, j + 8, k);
        buf4[7] += x(i + hipThreadIdx_x, j + 6, k);
        buf4[5] -= x(i + hipThreadIdx_x, j + 6, k);
        buf6[7] += x(i + hipThreadIdx_x + 1, j + 6, k);
        buf6[6] -= x(i + hipThreadIdx_x + 1, j + 6, k);
        buf2[6] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf2[14] += 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf2[22] -= x(i + hipThreadIdx_x, j + 6, k);
        buf3[6] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k);
        buf3[14] += x(i + hipThreadIdx_x, j + 6, k);
        buf5[14] += x(i + hipThreadIdx_x, j + 6, k);
        buf7[14] += x(i + hipThreadIdx_x, j + 6, k);
        buf7[15] -= x(i + hipThreadIdx_x, j + 6, k);
      }
      {
        buf0[7] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k);
        buf0[6] += x(i + hipThreadIdx_x, j + 7, k);
        buf1[7] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k);
        buf1[6] += 15.0 * x(i + hipThreadIdx_x, j + 7, k);
        buf1[5] -= x(i + hipThreadIdx_x, j + 7, k);
        buf2[7] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k);
        buf2[15] += 15.0 * x(i + hipThreadIdx_x, j + 7, k);
        buf2[23] -= x(i + hipThreadIdx_x, j + 7, k);
        buf3[7] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k);
        buf3[15] += x(i + hipThreadIdx_x, j + 7, k);
        buf6[7] -= x(i + hipThreadIdx_x + 1, j + 7, k);
        buf4[6] -= x(i + hipThreadIdx_x, j + 7, k);
        buf5[15] += x(i + hipThreadIdx_x, j + 7, k);
        buf7[15] += x(i + hipThreadIdx_x, j + 7, k);
      }
      {
        buf0[7] += x(i + hipThreadIdx_x, j + 8, k);
        buf1[7] += 15.0 * x(i + hipThreadIdx_x, j + 8, k);
        buf1[6] -= x(i + hipThreadIdx_x, j + 8, k);
        buf4[7] -= x(i + hipThreadIdx_x, j + 8, k);
      }
      {
        buf1[7] -= x(i + hipThreadIdx_x, j + 9, k);
      }
      {
        buf4[8] -= x(i + hipThreadIdx_x + -1, j + -1, k + 1);
        buf6[8] -= x(i + hipThreadIdx_x + -1, j + -1, k + 1);
      }
      {
        buf4[9] -= x(i + hipThreadIdx_x + -1, j, k + 1);
        buf6[9] -= x(i + hipThreadIdx_x + -1, j, k + 1);
        buf6[8] += x(i + hipThreadIdx_x + -1, j, k + 1);
        buf5[0] += x(i + hipThreadIdx_x + -1, j, k + 1);
        buf5[16] -= x(i + hipThreadIdx_x + -1, j, k + 1);
      }
      {
        buf0[8] += 15.0 * x(i + hipThreadIdx_x, j + -1, k + 1);
        buf0[8] -= x(i + hipThreadIdx_x, j + -1, k + 1);
        buf1[8] += x(i + hipThreadIdx_x, j + -1, k + 1);
        buf4[8] += x(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf4[8] += x(i + hipThreadIdx_x, j + -1, k + 1);
        buf4[10] -= x(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf6[8] += x(i + hipThreadIdx_x + 1, j + -1, k + 1);
        buf6[10] -= x(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf6[9] += x(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf5[1] += x(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf5[17] -= x(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf7[0] += x(i + hipThreadIdx_x, j + -1, k + 1);
        buf7[16] -= x(i + hipThreadIdx_x, j + -1, k + 1);
      }
      {
        buf0[9] += 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf0[9] -= x(i + hipThreadIdx_x, j, k + 1);
        buf0[8] -= 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf1[9] += x(i + hipThreadIdx_x, j, k + 1);
        buf1[8] -= 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf4[9] += x(i + hipThreadIdx_x + -1, j + 2, k + 1);
        buf4[9] += x(i + hipThreadIdx_x, j, k + 1);
        buf4[11] -= x(i + hipThreadIdx_x + -1, j + 2, k + 1);
        buf6[9] += x(i + hipThreadIdx_x + 1, j, k + 1);
        buf6[8] -= x(i + hipThreadIdx_x + 1, j, k + 1);
        buf6[11] -= x(i + hipThreadIdx_x + -1, j + 2, k + 1);
        buf6[10] += x(i + hipThreadIdx_x + -1, j + 2, k + 1);
        buf2[8] -= 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf2[16] += 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf2[24] -= x(i + hipThreadIdx_x, j, k + 1);
        buf2[0] += x(i + hipThreadIdx_x, j, k + 1);
        buf3[8] -= 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf3[16] += x(i + hipThreadIdx_x, j, k + 1);
        buf3[0] += 15.0 * x(i + hipThreadIdx_x, j, k + 1);
        buf5[16] += x(i + hipThreadIdx_x, j, k + 1);
        buf5[0] -= x(i + hipThreadIdx_x, j, k + 1);
        buf5[2] += x(i + hipThreadIdx_x + -1, j + 2, k + 1);
        buf5[18] -= x(i + hipThreadIdx_x + -1, j + 2, k + 1);
        buf7[16] += x(i + hipThreadIdx_x, j, k + 1);
        buf7[0] -= x(i + hipThreadIdx_x, j, k + 1);
        buf7[1] += x(i + hipThreadIdx_x, j, k + 1);
        buf7[17] -= x(i + hipThreadIdx_x, j, k + 1);
      }
      {
        buf0[10] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[10] -= x(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[9] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[8] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf1[10] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf1[9] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf1[8] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf4[10] += x(i + hipThreadIdx_x + -1, j + 3, k + 1);
        buf4[10] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf4[8] -= x(i + hipThreadIdx_x, j + 1, k + 1);
        buf4[12] -= x(i + hipThreadIdx_x + -1, j + 3, k + 1);
        buf6[10] += x(i + hipThreadIdx_x + 1, j + 1, k + 1);
        buf6[9] -= x(i + hipThreadIdx_x + 1, j + 1, k + 1);
        buf6[12] -= x(i + hipThreadIdx_x + -1, j + 3, k + 1);
        buf6[11] += x(i + hipThreadIdx_x + -1, j + 3, k + 1);
        buf2[9] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf2[17] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf2[25] -= x(i + hipThreadIdx_x, j + 1, k + 1);
        buf2[1] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf3[9] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf3[17] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf3[1] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 1);
        buf5[17] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf5[1] -= x(i + hipThreadIdx_x, j + 1, k + 1);
        buf5[3] += x(i + hipThreadIdx_x + -1, j + 3, k + 1);
        buf5[19] -= x(i + hipThreadIdx_x + -1, j + 3, k + 1);
        buf7[17] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf7[1] -= x(i + hipThreadIdx_x, j + 1, k + 1);
        buf7[2] += x(i + hipThreadIdx_x, j + 1, k + 1);
        buf7[18] -= x(i + hipThreadIdx_x, j + 1, k + 1);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 3; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[11 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf0[11 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf0[10 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf0[9 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf1[11 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf1[10 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf1[9 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf1[8 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf4[11 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 1);
            buf4[11 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf4[9 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf4[13 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 1);
            buf6[11 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1);
            buf6[10 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1);
            buf6[13 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 1);
            buf6[12 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 1);
            buf2[10 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf2[18 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf2[26 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf2[2 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf3[10 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf3[18 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf3[2 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf5[18 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf5[2 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf5[4 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 1);
            buf5[20 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 1);
            buf7[18 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf7[2 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf7[3 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf7[19 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[14] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[14] -= x(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[13] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[12] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf1[14] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf1[13] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf1[12] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf1[11] -= x(i + hipThreadIdx_x, j + 5, k + 1);
        buf4[14] += x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf4[14] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf4[12] -= x(i + hipThreadIdx_x, j + 5, k + 1);
        buf6[14] += x(i + hipThreadIdx_x + 1, j + 5, k + 1);
        buf6[13] -= x(i + hipThreadIdx_x + 1, j + 5, k + 1);
        buf6[15] += x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf2[13] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf2[21] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf2[29] -= x(i + hipThreadIdx_x, j + 5, k + 1);
        buf2[5] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf3[13] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf3[21] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf3[5] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 1);
        buf5[21] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf5[5] -= x(i + hipThreadIdx_x, j + 5, k + 1);
        buf5[7] += x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf5[23] -= x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf7[21] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf7[5] -= x(i + hipThreadIdx_x, j + 5, k + 1);
        buf7[6] += x(i + hipThreadIdx_x, j + 5, k + 1);
        buf7[22] -= x(i + hipThreadIdx_x, j + 5, k + 1);
      }
      {
        buf0[15] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[15] -= x(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[14] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[13] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf1[15] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf1[14] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf1[13] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf1[12] -= x(i + hipThreadIdx_x, j + 6, k + 1);
        buf4[15] += x(i + hipThreadIdx_x + -1, j + 8, k + 1);
        buf4[15] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf4[13] -= x(i + hipThreadIdx_x, j + 6, k + 1);
        buf6[15] += x(i + hipThreadIdx_x + 1, j + 6, k + 1);
        buf6[14] -= x(i + hipThreadIdx_x + 1, j + 6, k + 1);
        buf2[14] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf2[22] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf2[30] -= x(i + hipThreadIdx_x, j + 6, k + 1);
        buf2[6] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf3[14] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf3[22] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf3[6] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 1);
        buf5[22] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf5[6] -= x(i + hipThreadIdx_x, j + 6, k + 1);
        buf7[22] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf7[6] -= x(i + hipThreadIdx_x, j + 6, k + 1);
        buf7[7] += x(i + hipThreadIdx_x, j + 6, k + 1);
        buf7[23] -= x(i + hipThreadIdx_x, j + 6, k + 1);
      }
      {
        buf0[15] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[14] += x(i + hipThreadIdx_x, j + 7, k + 1);
        buf1[15] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf1[14] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf1[13] -= x(i + hipThreadIdx_x, j + 7, k + 1);
        buf2[15] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf2[23] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf2[31] -= x(i + hipThreadIdx_x, j + 7, k + 1);
        buf2[7] += x(i + hipThreadIdx_x, j + 7, k + 1);
        buf3[15] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf3[23] += x(i + hipThreadIdx_x, j + 7, k + 1);
        buf3[7] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 1);
        buf6[15] -= x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf4[14] -= x(i + hipThreadIdx_x, j + 7, k + 1);
        buf5[23] += x(i + hipThreadIdx_x, j + 7, k + 1);
        buf5[7] -= x(i + hipThreadIdx_x, j + 7, k + 1);
        buf7[23] += x(i + hipThreadIdx_x, j + 7, k + 1);
        buf7[7] -= x(i + hipThreadIdx_x, j + 7, k + 1);
      }
      {
        buf0[15] += x(i + hipThreadIdx_x, j + 8, k + 1);
        buf1[15] += 15.0 * x(i + hipThreadIdx_x, j + 8, k + 1);
        buf1[14] -= x(i + hipThreadIdx_x, j + 8, k + 1);
        buf4[15] -= x(i + hipThreadIdx_x, j + 8, k + 1);
      }
      {
        buf1[15] -= x(i + hipThreadIdx_x, j + 9, k + 1);
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 4; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            buf4[16 + rel] -= x(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 2);
            buf6[16 + rel] -= x(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 2);
          }
          {
            buf4[17 + rel] -= x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
            buf6[17 + rel] -= x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
            buf6[16 + rel] += x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
            buf5[8 + rel] += x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
            buf5[24 + rel] -= x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
          }
          {
            buf0[16 + rel] += 15.0 * x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2);
            buf0[16 + rel] -= x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2);
            buf1[16 + rel] += x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2);
            buf4[16 + rel] += x(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2);
            buf4[16 + rel] += x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2);
            buf4[18 + rel] -= x(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2);
            buf6[16 + rel] += x(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 2);
            buf6[18 + rel] -= x(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2);
            buf6[17 + rel] += x(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2);
            buf5[9 + rel] += x(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2);
            buf5[25 + rel] -= x(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2);
            buf7[8 + rel] += x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2);
            buf7[24 + rel] -= x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2);
          }
          {
            buf0[17 + rel] += 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf0[17 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf0[16 + rel] -= 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf1[17 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf1[16 + rel] -= 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf4[17 + rel] += x(i + hipThreadIdx_x + -1, j + 2, k + _cg_idx2 + 2);
            buf4[17 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf4[19 + rel] -= x(i + hipThreadIdx_x + -1, j + 2, k + _cg_idx2 + 2);
            buf6[17 + rel] += x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf6[16 + rel] -= x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf6[19 + rel] -= x(i + hipThreadIdx_x + -1, j + 2, k + _cg_idx2 + 2);
            buf6[18 + rel] += x(i + hipThreadIdx_x + -1, j + 2, k + _cg_idx2 + 2);
            buf2[16 + rel] -= 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf2[24 + rel] += 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf2[32 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf2[8 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf3[16 + rel] -= 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf3[24 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf3[8 + rel] += 15.0 * x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf3[0 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf5[24 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf5[8 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf5[10 + rel] += x(i + hipThreadIdx_x + -1, j + 2, k + _cg_idx2 + 2);
            buf5[26 + rel] -= x(i + hipThreadIdx_x + -1, j + 2, k + _cg_idx2 + 2);
            buf7[24 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf7[8 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf7[9 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf7[25 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
          }
          {
            buf0[18 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf0[18 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf0[17 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf0[16 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf1[18 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf1[17 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf1[16 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf4[18 + rel] += x(i + hipThreadIdx_x + -1, j + 3, k + _cg_idx2 + 2);
            buf4[18 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf4[16 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf4[20 + rel] -= x(i + hipThreadIdx_x + -1, j + 3, k + _cg_idx2 + 2);
            buf6[18 + rel] += x(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2);
            buf6[17 + rel] -= x(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2);
            buf6[20 + rel] -= x(i + hipThreadIdx_x + -1, j + 3, k + _cg_idx2 + 2);
            buf6[19 + rel] += x(i + hipThreadIdx_x + -1, j + 3, k + _cg_idx2 + 2);
            buf2[17 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf2[25 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf2[33 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf2[9 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf3[17 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf3[25 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf3[9 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf3[1 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf5[25 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf5[9 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf5[11 + rel] += x(i + hipThreadIdx_x + -1, j + 3, k + _cg_idx2 + 2);
            buf5[27 + rel] -= x(i + hipThreadIdx_x + -1, j + 3, k + _cg_idx2 + 2);
            buf7[25 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf7[9 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf7[10 + rel] += x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf7[26 + rel] -= x(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
          }
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 3; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf0[19 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[19 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[18 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[17 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf1[19 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf1[18 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf1[17 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf1[16 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf4[19 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + _cg_idx2 + 2);
                buf4[19 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf4[17 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf4[21 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + _cg_idx2 + 2);
                buf6[19 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf6[18 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf6[21 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + _cg_idx2 + 2);
                buf6[20 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + _cg_idx2 + 2);
                buf2[18 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf2[26 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf2[34 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf2[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf3[18 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf3[26 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf3[10 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf3[2 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf5[26 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf5[10 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf5[12 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + _cg_idx2 + 2);
                buf5[28 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + _cg_idx2 + 2);
                buf7[26 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf7[10 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf7[11 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf7[27 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
              }
              _cg_rel1 += 1;
            }
          }
          {
            buf0[22 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf0[22 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf0[21 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf0[20 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf1[22 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf1[21 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf1[20 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf1[19 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf4[22 + rel] += x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf4[22 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf4[20 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf6[22 + rel] += x(i + hipThreadIdx_x + 1, j + 5, k + _cg_idx2 + 2);
            buf6[21 + rel] -= x(i + hipThreadIdx_x + 1, j + 5, k + _cg_idx2 + 2);
            buf6[23 + rel] += x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf2[21 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf2[29 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf2[37 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf2[13 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf3[21 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf3[29 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf3[13 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf3[5 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf5[29 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf5[13 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf5[15 + rel] += x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf5[31 + rel] -= x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf7[29 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf7[13 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf7[14 + rel] += x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
            buf7[30 + rel] -= x(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 2);
          }
          {
            buf0[23 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf0[23 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf0[22 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf0[21 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf1[23 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf1[22 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf1[21 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf1[20 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf4[23 + rel] += x(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 2);
            buf4[23 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf4[21 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf6[23 + rel] += x(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2);
            buf6[22 + rel] -= x(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2);
            buf2[22 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf2[30 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf2[38 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf2[14 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf3[22 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf3[30 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf3[14 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf3[6 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf5[30 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf5[14 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf7[30 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf7[14 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf7[15 + rel] += x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf7[31 + rel] -= x(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
          }
          {
            buf0[23 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf0[22 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf1[23 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf1[22 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf1[21 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf2[23 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf2[31 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf2[39 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf2[15 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf3[23 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf3[31 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf3[15 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf3[7 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf6[23 + rel] -= x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf4[22 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf5[31 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf5[15 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf7[31 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf7[15 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
          }
          {
            buf0[23 + rel] += x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2);
            buf1[23 + rel] += 15.0 * x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2);
            buf1[22 + rel] -= x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2);
            buf4[23 + rel] -= x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2);
          }
          {
            buf1[23 + rel] -= x(i + hipThreadIdx_x, j + 9, k + _cg_idx2 + 2);
          }
          _cg_rel2 += 8;
        }
      }
      {
        buf4[48] -= x(i + hipThreadIdx_x + -1, j + -1, k + 6);
        buf6[48] -= x(i + hipThreadIdx_x + -1, j + -1, k + 6);
      }
      {
        buf4[49] -= x(i + hipThreadIdx_x + -1, j, k + 6);
        buf6[49] -= x(i + hipThreadIdx_x + -1, j, k + 6);
        buf6[48] += x(i + hipThreadIdx_x + -1, j, k + 6);
        buf5[40] += x(i + hipThreadIdx_x + -1, j, k + 6);
        buf5[56] -= x(i + hipThreadIdx_x + -1, j, k + 6);
      }
      {
        buf0[48] += 15.0 * x(i + hipThreadIdx_x, j + -1, k + 6);
        buf0[48] -= x(i + hipThreadIdx_x, j + -1, k + 6);
        buf1[48] += x(i + hipThreadIdx_x, j + -1, k + 6);
        buf4[48] += x(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf4[48] += x(i + hipThreadIdx_x, j + -1, k + 6);
        buf4[50] -= x(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf6[48] += x(i + hipThreadIdx_x + 1, j + -1, k + 6);
        buf6[50] -= x(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf6[49] += x(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf5[41] += x(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf5[57] -= x(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf7[40] += x(i + hipThreadIdx_x, j + -1, k + 6);
        buf7[56] -= x(i + hipThreadIdx_x, j + -1, k + 6);
      }
      {
        buf0[49] += 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf0[49] -= x(i + hipThreadIdx_x, j, k + 6);
        buf0[48] -= 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf1[49] += x(i + hipThreadIdx_x, j, k + 6);
        buf1[48] -= 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf4[49] += x(i + hipThreadIdx_x + -1, j + 2, k + 6);
        buf4[49] += x(i + hipThreadIdx_x, j, k + 6);
        buf4[51] -= x(i + hipThreadIdx_x + -1, j + 2, k + 6);
        buf6[49] += x(i + hipThreadIdx_x + 1, j, k + 6);
        buf6[48] -= x(i + hipThreadIdx_x + 1, j, k + 6);
        buf6[51] -= x(i + hipThreadIdx_x + -1, j + 2, k + 6);
        buf6[50] += x(i + hipThreadIdx_x + -1, j + 2, k + 6);
        buf2[48] -= 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf2[56] += 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf2[40] += x(i + hipThreadIdx_x, j, k + 6);
        buf3[48] -= 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf3[56] += x(i + hipThreadIdx_x, j, k + 6);
        buf3[40] += 15.0 * x(i + hipThreadIdx_x, j, k + 6);
        buf3[32] -= x(i + hipThreadIdx_x, j, k + 6);
        buf5[56] += x(i + hipThreadIdx_x, j, k + 6);
        buf5[40] -= x(i + hipThreadIdx_x, j, k + 6);
        buf5[42] += x(i + hipThreadIdx_x + -1, j + 2, k + 6);
        buf5[58] -= x(i + hipThreadIdx_x + -1, j + 2, k + 6);
        buf7[56] += x(i + hipThreadIdx_x, j, k + 6);
        buf7[40] -= x(i + hipThreadIdx_x, j, k + 6);
        buf7[41] += x(i + hipThreadIdx_x, j, k + 6);
        buf7[57] -= x(i + hipThreadIdx_x, j, k + 6);
      }
      {
        buf0[50] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[50] -= x(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[49] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[48] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf1[50] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf1[49] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf1[48] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf4[50] += x(i + hipThreadIdx_x + -1, j + 3, k + 6);
        buf4[50] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf4[48] -= x(i + hipThreadIdx_x, j + 1, k + 6);
        buf4[52] -= x(i + hipThreadIdx_x + -1, j + 3, k + 6);
        buf6[50] += x(i + hipThreadIdx_x + 1, j + 1, k + 6);
        buf6[49] -= x(i + hipThreadIdx_x + 1, j + 1, k + 6);
        buf6[52] -= x(i + hipThreadIdx_x + -1, j + 3, k + 6);
        buf6[51] += x(i + hipThreadIdx_x + -1, j + 3, k + 6);
        buf2[49] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf2[57] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf2[41] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf3[49] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf3[57] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf3[41] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 6);
        buf3[33] -= x(i + hipThreadIdx_x, j + 1, k + 6);
        buf5[57] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf5[41] -= x(i + hipThreadIdx_x, j + 1, k + 6);
        buf5[43] += x(i + hipThreadIdx_x + -1, j + 3, k + 6);
        buf5[59] -= x(i + hipThreadIdx_x + -1, j + 3, k + 6);
        buf7[57] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf7[41] -= x(i + hipThreadIdx_x, j + 1, k + 6);
        buf7[42] += x(i + hipThreadIdx_x, j + 1, k + 6);
        buf7[58] -= x(i + hipThreadIdx_x, j + 1, k + 6);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 3; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[51 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf0[51 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf0[50 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf0[49 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf1[51 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf1[50 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf1[49 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf1[48 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf4[51 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 6);
            buf4[51 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf4[49 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf4[53 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 6);
            buf6[51 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6);
            buf6[50 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6);
            buf6[53 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 6);
            buf6[52 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 6);
            buf2[50 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf2[58 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf2[42 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf3[50 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf3[58 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf3[42 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf3[34 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf5[58 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf5[42 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf5[44 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 6);
            buf5[60 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 6);
            buf7[58 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf7[42 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf7[43 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf7[59 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[54] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[54] -= x(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[53] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[52] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf1[54] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf1[53] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf1[52] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf1[51] -= x(i + hipThreadIdx_x, j + 5, k + 6);
        buf4[54] += x(i + hipThreadIdx_x + -1, j + 7, k + 6);
        buf4[54] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf4[52] -= x(i + hipThreadIdx_x, j + 5, k + 6);
        buf6[54] += x(i + hipThreadIdx_x + 1, j + 5, k + 6);
        buf6[53] -= x(i + hipThreadIdx_x + 1, j + 5, k + 6);
        buf6[55] += x(i + hipThreadIdx_x + -1, j + 7, k + 6);
        buf2[53] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf2[61] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf2[45] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf3[53] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf3[61] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf3[45] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 6);
        buf3[37] -= x(i + hipThreadIdx_x, j + 5, k + 6);
        buf5[61] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf5[45] -= x(i + hipThreadIdx_x, j + 5, k + 6);
        buf5[47] += x(i + hipThreadIdx_x + -1, j + 7, k + 6);
        buf5[63] -= x(i + hipThreadIdx_x + -1, j + 7, k + 6);
        buf7[61] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf7[45] -= x(i + hipThreadIdx_x, j + 5, k + 6);
        buf7[46] += x(i + hipThreadIdx_x, j + 5, k + 6);
        buf7[62] -= x(i + hipThreadIdx_x, j + 5, k + 6);
      }
      {
        buf0[55] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[55] -= x(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[54] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[53] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf1[55] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf1[54] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf1[53] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf1[52] -= x(i + hipThreadIdx_x, j + 6, k + 6);
        buf4[55] += x(i + hipThreadIdx_x + -1, j + 8, k + 6);
        buf4[55] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf4[53] -= x(i + hipThreadIdx_x, j + 6, k + 6);
        buf6[55] += x(i + hipThreadIdx_x + 1, j + 6, k + 6);
        buf6[54] -= x(i + hipThreadIdx_x + 1, j + 6, k + 6);
        buf2[54] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf2[62] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf2[46] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf3[54] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf3[62] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf3[46] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 6);
        buf3[38] -= x(i + hipThreadIdx_x, j + 6, k + 6);
        buf5[62] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf5[46] -= x(i + hipThreadIdx_x, j + 6, k + 6);
        buf7[62] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf7[46] -= x(i + hipThreadIdx_x, j + 6, k + 6);
        buf7[47] += x(i + hipThreadIdx_x, j + 6, k + 6);
        buf7[63] -= x(i + hipThreadIdx_x, j + 6, k + 6);
      }
      {
        buf0[55] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[54] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf1[55] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf1[54] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf1[53] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf2[55] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf2[63] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf2[47] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf3[55] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf3[63] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf3[47] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 6);
        buf3[39] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf6[55] -= x(i + hipThreadIdx_x + 1, j + 7, k + 6);
        buf4[54] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf5[63] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf5[47] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf7[63] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf7[47] -= x(i + hipThreadIdx_x, j + 7, k + 6);
      }
      {
        buf0[55] += x(i + hipThreadIdx_x, j + 8, k + 6);
        buf1[55] += 15.0 * x(i + hipThreadIdx_x, j + 8, k + 6);
        buf1[54] -= x(i + hipThreadIdx_x, j + 8, k + 6);
        buf4[55] -= x(i + hipThreadIdx_x, j + 8, k + 6);
      }
      {
        buf1[55] -= x(i + hipThreadIdx_x, j + 9, k + 6);
      }
      {
        buf4[56] -= x(i + hipThreadIdx_x + -1, j + -1, k + 7);
        buf6[56] -= x(i + hipThreadIdx_x + -1, j + -1, k + 7);
      }
      {
        buf4[57] -= x(i + hipThreadIdx_x + -1, j, k + 7);
        buf6[57] -= x(i + hipThreadIdx_x + -1, j, k + 7);
        buf6[56] += x(i + hipThreadIdx_x + -1, j, k + 7);
        buf5[48] += x(i + hipThreadIdx_x + -1, j, k + 7);
      }
      {
        buf0[56] += 15.0 * x(i + hipThreadIdx_x, j + -1, k + 7);
        buf0[56] -= x(i + hipThreadIdx_x, j + -1, k + 7);
        buf1[56] += x(i + hipThreadIdx_x, j + -1, k + 7);
        buf4[56] += x(i + hipThreadIdx_x + -1, j + 1, k + 7);
        buf4[56] += x(i + hipThreadIdx_x, j + -1, k + 7);
        buf4[58] -= x(i + hipThreadIdx_x + -1, j + 1, k + 7);
        buf6[56] += x(i + hipThreadIdx_x + 1, j + -1, k + 7);
        buf6[58] -= x(i + hipThreadIdx_x + -1, j + 1, k + 7);
        buf6[57] += x(i + hipThreadIdx_x + -1, j + 1, k + 7);
        buf5[49] += x(i + hipThreadIdx_x + -1, j + 1, k + 7);
        buf7[48] += x(i + hipThreadIdx_x, j + -1, k + 7);
      }
      {
        buf0[57] += 15.0 * x(i + hipThreadIdx_x, j, k + 7);
        buf0[57] -= x(i + hipThreadIdx_x, j, k + 7);
        buf0[56] -= 15.0 * x(i + hipThreadIdx_x, j, k + 7);
        buf1[57] += x(i + hipThreadIdx_x, j, k + 7);
        buf1[56] -= 15.0 * x(i + hipThreadIdx_x, j, k + 7);
        buf4[57] += x(i + hipThreadIdx_x + -1, j + 2, k + 7);
        buf4[57] += x(i + hipThreadIdx_x, j, k + 7);
        buf4[59] -= x(i + hipThreadIdx_x + -1, j + 2, k + 7);
        buf6[57] += x(i + hipThreadIdx_x + 1, j, k + 7);
        buf6[56] -= x(i + hipThreadIdx_x + 1, j, k + 7);
        buf6[59] -= x(i + hipThreadIdx_x + -1, j + 2, k + 7);
        buf6[58] += x(i + hipThreadIdx_x + -1, j + 2, k + 7);
        buf2[56] -= 15.0 * x(i + hipThreadIdx_x, j, k + 7);
        buf2[48] += x(i + hipThreadIdx_x, j, k + 7);
        buf3[56] -= 15.0 * x(i + hipThreadIdx_x, j, k + 7);
        buf3[48] += 15.0 * x(i + hipThreadIdx_x, j, k + 7);
        buf3[40] -= x(i + hipThreadIdx_x, j, k + 7);
        buf5[48] -= x(i + hipThreadIdx_x, j, k + 7);
        buf5[50] += x(i + hipThreadIdx_x + -1, j + 2, k + 7);
        buf7[48] -= x(i + hipThreadIdx_x, j, k + 7);
        buf7[49] += x(i + hipThreadIdx_x, j, k + 7);
      }
      {
        buf0[58] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[58] -= x(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[57] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[56] += x(i + hipThreadIdx_x, j + 1, k + 7);
        buf1[58] += x(i + hipThreadIdx_x, j + 1, k + 7);
        buf1[57] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf1[56] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf4[58] += x(i + hipThreadIdx_x + -1, j + 3, k + 7);
        buf4[58] += x(i + hipThreadIdx_x, j + 1, k + 7);
        buf4[56] -= x(i + hipThreadIdx_x, j + 1, k + 7);
        buf4[60] -= x(i + hipThreadIdx_x + -1, j + 3, k + 7);
        buf6[58] += x(i + hipThreadIdx_x + 1, j + 1, k + 7);
        buf6[57] -= x(i + hipThreadIdx_x + 1, j + 1, k + 7);
        buf6[60] -= x(i + hipThreadIdx_x + -1, j + 3, k + 7);
        buf6[59] += x(i + hipThreadIdx_x + -1, j + 3, k + 7);
        buf2[57] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf2[49] += x(i + hipThreadIdx_x, j + 1, k + 7);
        buf3[57] -= 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf3[49] += 15.0 * x(i + hipThreadIdx_x, j + 1, k + 7);
        buf3[41] -= x(i + hipThreadIdx_x, j + 1, k + 7);
        buf5[49] -= x(i + hipThreadIdx_x, j + 1, k + 7);
        buf5[51] += x(i + hipThreadIdx_x + -1, j + 3, k + 7);
        buf7[49] -= x(i + hipThreadIdx_x, j + 1, k + 7);
        buf7[50] += x(i + hipThreadIdx_x, j + 1, k + 7);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 3; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[59 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf0[59 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf0[58 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf0[57 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf1[59 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf1[58 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf1[57 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf1[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf4[59 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 7);
            buf4[59 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf4[57 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf4[61 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 7);
            buf6[59 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7);
            buf6[58 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7);
            buf6[61 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 7);
            buf6[60 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 7);
            buf2[58 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf2[50 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf3[58 + rel] -= 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf3[50 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf3[42 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf5[50 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf5[52 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 4, k + 7);
            buf7[50 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf7[51 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[62] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[62] -= x(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[61] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[60] += x(i + hipThreadIdx_x, j + 5, k + 7);
        buf1[62] += x(i + hipThreadIdx_x, j + 5, k + 7);
        buf1[61] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf1[60] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf1[59] -= x(i + hipThreadIdx_x, j + 5, k + 7);
        buf4[62] += x(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf4[62] += x(i + hipThreadIdx_x, j + 5, k + 7);
        buf4[60] -= x(i + hipThreadIdx_x, j + 5, k + 7);
        buf6[62] += x(i + hipThreadIdx_x + 1, j + 5, k + 7);
        buf6[61] -= x(i + hipThreadIdx_x + 1, j + 5, k + 7);
        buf6[63] += x(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf2[61] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf2[53] += x(i + hipThreadIdx_x, j + 5, k + 7);
        buf3[61] -= 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf3[53] += 15.0 * x(i + hipThreadIdx_x, j + 5, k + 7);
        buf3[45] -= x(i + hipThreadIdx_x, j + 5, k + 7);
        buf5[53] -= x(i + hipThreadIdx_x, j + 5, k + 7);
        buf5[55] += x(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf7[53] -= x(i + hipThreadIdx_x, j + 5, k + 7);
        buf7[54] += x(i + hipThreadIdx_x, j + 5, k + 7);
      }
      {
        buf0[63] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[63] -= x(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[62] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[61] += x(i + hipThreadIdx_x, j + 6, k + 7);
        buf1[63] += x(i + hipThreadIdx_x, j + 6, k + 7);
        buf1[62] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf1[61] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf1[60] -= x(i + hipThreadIdx_x, j + 6, k + 7);
        buf4[63] += x(i + hipThreadIdx_x + -1, j + 8, k + 7);
        buf4[63] += x(i + hipThreadIdx_x, j + 6, k + 7);
        buf4[61] -= x(i + hipThreadIdx_x, j + 6, k + 7);
        buf6[63] += x(i + hipThreadIdx_x + 1, j + 6, k + 7);
        buf6[62] -= x(i + hipThreadIdx_x + 1, j + 6, k + 7);
        buf2[62] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf2[54] += x(i + hipThreadIdx_x, j + 6, k + 7);
        buf3[62] -= 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf3[54] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 7);
        buf3[46] -= x(i + hipThreadIdx_x, j + 6, k + 7);
        buf5[54] -= x(i + hipThreadIdx_x, j + 6, k + 7);
        buf7[54] -= x(i + hipThreadIdx_x, j + 6, k + 7);
        buf7[55] += x(i + hipThreadIdx_x, j + 6, k + 7);
      }
      {
        buf0[63] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[62] += x(i + hipThreadIdx_x, j + 7, k + 7);
        buf1[63] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 7);
        buf1[62] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 7);
        buf1[61] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf2[63] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 7);
        buf2[55] += x(i + hipThreadIdx_x, j + 7, k + 7);
        buf3[63] -= 15.0 * x(i + hipThreadIdx_x, j + 7, k + 7);
        buf3[55] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 7);
        buf3[47] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf6[63] -= x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf4[62] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf5[55] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf7[55] -= x(i + hipThreadIdx_x, j + 7, k + 7);
      }
      {
        buf0[63] += x(i + hipThreadIdx_x, j + 8, k + 7);
        buf1[63] += 15.0 * x(i + hipThreadIdx_x, j + 8, k + 7);
        buf1[62] -= x(i + hipThreadIdx_x, j + 8, k + 7);
        buf4[63] -= x(i + hipThreadIdx_x, j + 8, k + 7);
      }
      {
        buf1[63] -= x(i + hipThreadIdx_x, j + 9, k + 7);
      }
      {
        buf5[56] += x(i + hipThreadIdx_x + -1, j, k + 8);
      }
      {
        buf5[57] += x(i + hipThreadIdx_x + -1, j + 1, k + 8);
        buf7[56] += x(i + hipThreadIdx_x, j + -1, k + 8);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf2[56 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf3[56 + rel] += 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf3[48 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf5[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf5[58 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 8);
            buf7[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf7[57 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf2[62] += x(i + hipThreadIdx_x, j + 6, k + 8);
        buf3[62] += 15.0 * x(i + hipThreadIdx_x, j + 6, k + 8);
        buf3[54] -= x(i + hipThreadIdx_x, j + 6, k + 8);
        buf5[62] -= x(i + hipThreadIdx_x, j + 6, k + 8);
        buf7[62] -= x(i + hipThreadIdx_x, j + 6, k + 8);
        buf7[63] += x(i + hipThreadIdx_x, j + 6, k + 8);
      }
      {
        buf2[63] += x(i + hipThreadIdx_x, j + 7, k + 8);
        buf3[63] += 15.0 * x(i + hipThreadIdx_x, j + 7, k + 8);
        buf3[55] -= x(i + hipThreadIdx_x, j + 7, k + 8);
        buf5[63] -= x(i + hipThreadIdx_x, j + 7, k + 8);
        buf7[63] -= x(i + hipThreadIdx_x, j + 7, k + 8);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 7; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf3[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1, k + 9);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf3[63] -= x(i + hipThreadIdx_x, j + 7, k + 9);
      }
    }
  }
  bElem buf8[64];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf8[0 + rel] = 0;
              }
              _cg_rel1 += 1;
            }
          }
          _cg_rel2 += 8;
        }
      }
    }
    {
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf8[0 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + -1);
            buf8[0 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + -1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 7; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf8[8 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2);
                buf8[8 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + _cg_idx2);
                buf8[0 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2);
                buf8[0 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + _cg_idx2);
              }
              _cg_rel1 += 1;
            }
          }
          _cg_rel2 += 8;
        }
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf8[56 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + 7);
            buf8[56 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + 7);
          }
          _cg_rel1 += 1;
        }
      }
    }
  }
  bElem buf9[64];
  bElem buf10[64];
  bElem buf11[64];
  bElem buf12[64];
  bElem buf13[64];
  bElem buf14[64];
  bElem buf15[64];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf9[0 + rel] = 0;
                buf10[0 + rel] = 0;
                buf11[0 + rel] = 0;
                buf12[0 + rel] = 0;
                buf13[0 + rel] = 0;
                buf14[0 + rel] = 0;
                buf15[0 + rel] = 0;
              }
              _cg_rel1 += 1;
            }
          }
          _cg_rel2 += 8;
        }
      }
    }
    {
      {
        buf11[0] -= x(i + hipThreadIdx_x + 1, j, k + -1);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 7; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf11[1 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + -1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf9[0] -= x(i + hipThreadIdx_x, j + -1, k + -1);
        buf10[0] -= x(i + hipThreadIdx_x + 1, j + -1, k);
      }
      {
        buf9[1] -= x(i + hipThreadIdx_x, j, k + -1);
        buf10[1] -= x(i + hipThreadIdx_x + 1, j, k);
        buf11[8] -= x(i + hipThreadIdx_x + 1, j, k);
        buf11[0] += x(i + hipThreadIdx_x, j, k + -1);
        buf12[0] -= x(i + hipThreadIdx_x + 1, j, k);
        buf12[0] += x(i + hipThreadIdx_x + -1, j, k);
        buf13[0] += x(i + hipThreadIdx_x, j, k + -1);
        buf14[0] -= x(i + hipThreadIdx_x + 1, j, k);
        buf14[0] += x(i + hipThreadIdx_x + -1, j, k);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf9[0 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf9[2 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf10[0 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf10[2 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf12[0 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf12[0 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
            buf12[1 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf12[1 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
            buf13[0 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf13[1 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf11[9 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf11[1 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf14[1 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf14[1 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf9[6] += x(i + hipThreadIdx_x, j + 7, k + -1);
        buf10[6] += x(i + hipThreadIdx_x + 1, j + 7, k);
        buf12[6] += x(i + hipThreadIdx_x + 1, j + 7, k);
        buf12[6] -= x(i + hipThreadIdx_x + -1, j + 7, k);
        buf12[7] -= x(i + hipThreadIdx_x + 1, j + 7, k);
        buf12[7] += x(i + hipThreadIdx_x + -1, j + 7, k);
        buf13[6] -= x(i + hipThreadIdx_x, j + 7, k + -1);
        buf13[7] += x(i + hipThreadIdx_x, j + 7, k + -1);
        buf11[15] -= x(i + hipThreadIdx_x + 1, j + 7, k);
        buf11[7] += x(i + hipThreadIdx_x, j + 7, k + -1);
        buf14[7] -= x(i + hipThreadIdx_x + 1, j + 7, k);
        buf14[7] += x(i + hipThreadIdx_x + -1, j + 7, k);
      }
      {
        buf9[7] += x(i + hipThreadIdx_x, j + 8, k + -1);
        buf10[7] += x(i + hipThreadIdx_x + 1, j + 8, k);
        buf12[7] += x(i + hipThreadIdx_x + 1, j + 8, k);
        buf12[7] -= x(i + hipThreadIdx_x + -1, j + 8, k);
        buf13[7] -= x(i + hipThreadIdx_x, j + 8, k + -1);
      }
      {
        buf9[8] -= x(i + hipThreadIdx_x, j + -1, k);
        buf9[0] += x(i + hipThreadIdx_x, j + -1, k);
        buf10[8] -= x(i + hipThreadIdx_x + 1, j + -1, k + 1);
        buf10[0] += x(i + hipThreadIdx_x, j + -1, k);
        buf15[0] += x(i + hipThreadIdx_x, j + -1, k);
      }
      {
        buf9[9] -= x(i + hipThreadIdx_x, j, k);
        buf9[1] += x(i + hipThreadIdx_x, j, k);
        buf10[9] -= x(i + hipThreadIdx_x + 1, j, k + 1);
        buf10[1] += x(i + hipThreadIdx_x, j, k);
        buf15[1] += x(i + hipThreadIdx_x, j, k);
        buf11[0] += x(i + hipThreadIdx_x + 1, j, k + 1);
        buf11[16] -= x(i + hipThreadIdx_x + 1, j, k + 1);
        buf11[8] += x(i + hipThreadIdx_x, j, k);
        buf14[0] += x(i + hipThreadIdx_x + 1, j, k + 1);
        buf14[0] -= x(i + hipThreadIdx_x + -1, j, k + 1);
        buf14[8] -= x(i + hipThreadIdx_x + 1, j, k + 1);
        buf14[8] += x(i + hipThreadIdx_x + -1, j, k + 1);
        buf12[8] -= x(i + hipThreadIdx_x + 1, j, k + 1);
        buf12[8] += x(i + hipThreadIdx_x + -1, j, k + 1);
        buf13[8] += x(i + hipThreadIdx_x, j, k);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf9[8 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf9[0 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf9[10 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf9[2 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf10[8 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf10[0 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf10[10 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf10[2 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf12[8 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf12[8 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 1);
            buf12[9 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf12[9 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 1);
            buf13[8 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf13[9 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf15[0 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf15[2 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf11[1 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf11[17 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf11[9 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf14[1 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf14[1 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 1);
            buf14[9 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 1);
            buf14[9 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf9[14] += x(i + hipThreadIdx_x, j + 7, k);
        buf9[6] -= x(i + hipThreadIdx_x, j + 7, k);
        buf10[14] += x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf10[6] -= x(i + hipThreadIdx_x, j + 7, k);
        buf12[14] += x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf12[14] -= x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf12[15] -= x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf12[15] += x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf13[14] -= x(i + hipThreadIdx_x, j + 7, k);
        buf13[15] += x(i + hipThreadIdx_x, j + 7, k);
        buf15[6] -= x(i + hipThreadIdx_x, j + 7, k);
        buf11[7] += x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf11[23] -= x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf11[15] += x(i + hipThreadIdx_x, j + 7, k);
        buf14[7] += x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf14[7] -= x(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf14[15] -= x(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf14[15] += x(i + hipThreadIdx_x + -1, j + 7, k + 1);
      }
      {
        buf9[15] += x(i + hipThreadIdx_x, j + 8, k);
        buf9[7] -= x(i + hipThreadIdx_x, j + 8, k);
        buf10[15] += x(i + hipThreadIdx_x + 1, j + 8, k + 1);
        buf10[7] -= x(i + hipThreadIdx_x, j + 8, k);
        buf12[15] += x(i + hipThreadIdx_x + 1, j + 8, k + 1);
        buf12[15] -= x(i + hipThreadIdx_x + -1, j + 8, k + 1);
        buf13[15] -= x(i + hipThreadIdx_x, j + 8, k);
        buf15[7] -= x(i + hipThreadIdx_x, j + 8, k);
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 5; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            buf9[16 + rel] -= x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
            buf9[8 + rel] += x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
            buf10[16 + rel] -= x(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 2);
            buf10[8 + rel] += x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
            buf15[8 + rel] += x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
            buf15[0 + rel] -= x(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
          }
          {
            buf9[17 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf9[9 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf10[17 + rel] -= x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf10[9 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf15[9 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf15[1 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf11[8 + rel] += x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf11[0 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf11[24 + rel] -= x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf11[16 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf14[8 + rel] += x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf14[8 + rel] -= x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
            buf14[16 + rel] -= x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf14[16 + rel] += x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
            buf13[0 + rel] -= x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf13[16 + rel] += x(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf12[16 + rel] -= x(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf12[16 + rel] += x(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
          }
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf9[16 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf9[8 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf9[18 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf9[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf10[16 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf10[8 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf10[18 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf10[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf12[16 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf12[16 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf12[17 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf12[17 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf13[16 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf13[1 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf13[17 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf13[0 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf15[8 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf15[10 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf15[0 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf15[2 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf11[9 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf11[1 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf11[25 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf11[17 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf14[9 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf14[9 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf14[17 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
                buf14[17 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 2);
              }
              _cg_rel1 += 1;
            }
          }
          {
            buf9[22 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf9[14 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf10[22 + rel] += x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf10[14 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf12[22 + rel] += x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf12[22 + rel] -= x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf12[23 + rel] -= x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf12[23 + rel] += x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf13[22 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf13[7 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf13[23 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf13[6 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf15[14 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf15[6 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf11[15 + rel] += x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf11[7 + rel] -= x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf11[31 + rel] -= x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf11[23 + rel] += x(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf14[15 + rel] += x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf14[15 + rel] -= x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf14[23 + rel] -= x(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf14[23 + rel] += x(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
          }
          {
            buf9[23 + rel] += x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf9[15 + rel] -= x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf10[23 + rel] += x(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 2);
            buf10[15 + rel] -= x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf12[23 + rel] += x(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 2);
            buf12[23 + rel] -= x(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 2);
            buf13[23 + rel] -= x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf13[7 + rel] += x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf15[15 + rel] -= x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf15[7 + rel] += x(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
          }
          _cg_rel2 += 8;
        }
      }
      {
        buf9[56] -= x(i + hipThreadIdx_x, j + -1, k + 6);
        buf9[48] += x(i + hipThreadIdx_x, j + -1, k + 6);
        buf10[56] -= x(i + hipThreadIdx_x + 1, j + -1, k + 7);
        buf10[48] += x(i + hipThreadIdx_x, j + -1, k + 6);
        buf15[48] += x(i + hipThreadIdx_x, j + -1, k + 6);
        buf15[40] -= x(i + hipThreadIdx_x, j + -1, k + 6);
      }
      {
        buf9[57] -= x(i + hipThreadIdx_x, j, k + 6);
        buf9[49] += x(i + hipThreadIdx_x, j, k + 6);
        buf10[57] -= x(i + hipThreadIdx_x + 1, j, k + 7);
        buf10[49] += x(i + hipThreadIdx_x, j, k + 6);
        buf15[49] += x(i + hipThreadIdx_x, j, k + 6);
        buf15[41] -= x(i + hipThreadIdx_x, j, k + 6);
        buf11[48] += x(i + hipThreadIdx_x + 1, j, k + 7);
        buf11[40] -= x(i + hipThreadIdx_x, j, k + 6);
        buf11[56] += x(i + hipThreadIdx_x, j, k + 6);
        buf14[48] += x(i + hipThreadIdx_x + 1, j, k + 7);
        buf14[48] -= x(i + hipThreadIdx_x + -1, j, k + 7);
        buf14[56] -= x(i + hipThreadIdx_x + 1, j, k + 7);
        buf14[56] += x(i + hipThreadIdx_x + -1, j, k + 7);
        buf13[40] -= x(i + hipThreadIdx_x, j, k + 6);
        buf13[56] += x(i + hipThreadIdx_x, j, k + 6);
        buf12[56] -= x(i + hipThreadIdx_x + 1, j, k + 7);
        buf12[56] += x(i + hipThreadIdx_x + -1, j, k + 7);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf9[56 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf9[48 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf9[58 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf9[50 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf10[56 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf10[48 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf10[58 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf10[50 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf12[56 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf12[56 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf12[57 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf12[57 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf13[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf13[41 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf13[57 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf13[40 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf15[48 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf15[50 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf15[40 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf15[42 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf11[49 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf11[41 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf11[57 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 6);
            buf14[49 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf14[49 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf14[57 + rel] -= x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf14[57 + rel] += x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf9[62] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf9[54] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf10[62] += x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf10[54] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf12[62] += x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf12[62] -= x(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf12[63] -= x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf12[63] += x(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf13[62] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf13[47] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf13[63] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf13[46] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf15[54] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf15[46] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf11[55] += x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf11[47] -= x(i + hipThreadIdx_x, j + 7, k + 6);
        buf11[63] += x(i + hipThreadIdx_x, j + 7, k + 6);
        buf14[55] += x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf14[55] -= x(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf14[63] -= x(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf14[63] += x(i + hipThreadIdx_x + -1, j + 7, k + 7);
      }
      {
        buf9[63] += x(i + hipThreadIdx_x, j + 8, k + 6);
        buf9[55] -= x(i + hipThreadIdx_x, j + 8, k + 6);
        buf10[63] += x(i + hipThreadIdx_x + 1, j + 8, k + 7);
        buf10[55] -= x(i + hipThreadIdx_x, j + 8, k + 6);
        buf12[63] += x(i + hipThreadIdx_x + 1, j + 8, k + 7);
        buf12[63] -= x(i + hipThreadIdx_x + -1, j + 8, k + 7);
        buf13[63] -= x(i + hipThreadIdx_x, j + 8, k + 6);
        buf13[47] += x(i + hipThreadIdx_x, j + 8, k + 6);
        buf15[55] -= x(i + hipThreadIdx_x, j + 8, k + 6);
        buf15[47] += x(i + hipThreadIdx_x, j + 8, k + 6);
      }
      {
        buf9[56] += x(i + hipThreadIdx_x, j + -1, k + 7);
        buf10[56] += x(i + hipThreadIdx_x, j + -1, k + 7);
        buf15[56] += x(i + hipThreadIdx_x, j + -1, k + 7);
        buf15[48] -= x(i + hipThreadIdx_x, j + -1, k + 7);
      }
      {
        buf9[57] += x(i + hipThreadIdx_x, j, k + 7);
        buf10[57] += x(i + hipThreadIdx_x, j, k + 7);
        buf15[57] += x(i + hipThreadIdx_x, j, k + 7);
        buf15[49] -= x(i + hipThreadIdx_x, j, k + 7);
        buf11[56] += x(i + hipThreadIdx_x + 1, j, k + 8);
        buf11[48] -= x(i + hipThreadIdx_x, j, k + 7);
        buf14[56] += x(i + hipThreadIdx_x + 1, j, k + 8);
        buf14[56] -= x(i + hipThreadIdx_x + -1, j, k + 8);
        buf13[48] -= x(i + hipThreadIdx_x, j, k + 7);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf9[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf9[58 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf10[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf10[58 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf15[56 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf15[58 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf15[48 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf15[50 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf11[57 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 8);
            buf11[49 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf14[57 + rel] += x(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 8);
            buf14[57 + rel] -= x(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 8);
            buf13[49 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf13[48 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf9[62] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf10[62] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf15[62] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf15[54] += x(i + hipThreadIdx_x, j + 7, k + 7);
        buf11[63] += x(i + hipThreadIdx_x + 1, j + 7, k + 8);
        buf11[55] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf14[63] += x(i + hipThreadIdx_x + 1, j + 7, k + 8);
        buf14[63] -= x(i + hipThreadIdx_x + -1, j + 7, k + 8);
        buf13[55] -= x(i + hipThreadIdx_x, j + 7, k + 7);
        buf13[54] += x(i + hipThreadIdx_x, j + 7, k + 7);
      }
      {
        buf9[63] -= x(i + hipThreadIdx_x, j + 8, k + 7);
        buf10[63] -= x(i + hipThreadIdx_x, j + 8, k + 7);
        buf15[63] -= x(i + hipThreadIdx_x, j + 8, k + 7);
        buf15[55] += x(i + hipThreadIdx_x, j + 8, k + 7);
        buf13[55] += x(i + hipThreadIdx_x, j + 8, k + 7);
      }
      {
        buf15[56] -= x(i + hipThreadIdx_x, j + -1, k + 8);
      }
      {
        buf11[56] -= x(i + hipThreadIdx_x, j, k + 8);
        buf13[56] -= x(i + hipThreadIdx_x, j, k + 8);
        buf15[57] -= x(i + hipThreadIdx_x, j, k + 8);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf11[57 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
            buf13[57 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
            buf13[56 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
            buf15[56 + rel] += x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
            buf15[58 + rel] -= x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf11[63] -= x(i + hipThreadIdx_x, j + 7, k + 8);
        buf13[63] -= x(i + hipThreadIdx_x, j + 7, k + 8);
        buf13[62] += x(i + hipThreadIdx_x, j + 7, k + 8);
        buf15[62] += x(i + hipThreadIdx_x, j + 7, k + 8);
      }
      {
        buf13[63] += x(i + hipThreadIdx_x, j + 8, k + 8);
        buf15[63] += x(i + hipThreadIdx_x, j + 8, k + 8);
      }
    }
  }
  bElem buf16[64];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf16[0 + rel] = 0;
              }
              _cg_rel1 += 1;
            }
          }
          _cg_rel2 += 8;
        }
      }
    }
    {
      {
        buf16[0] += (beta_i(i + hipThreadIdx_x, j + 1, k) - beta_i(i + hipThreadIdx_x, j + -1, k)) * buf4[0];
        buf16[0] += (beta_j(i + hipThreadIdx_x + 1, j, k) - beta_j(i + hipThreadIdx_x + -1, j, k)) * buf6[0];
        buf16[0] += (beta_j(i + hipThreadIdx_x, j, k + 1) - beta_j(i + hipThreadIdx_x, j, k + -1)) * buf7[0];
        buf16[0] += (beta_k(i + hipThreadIdx_x + 1, j, k) - beta_k(i + hipThreadIdx_x + -1, j, k)) * buf8[0];
        buf16[0] += (beta_k(i + hipThreadIdx_x, j + 1, k) - beta_k(i + hipThreadIdx_x, j + -1, k)) * buf9[0];
        buf16[0] += (beta_i(i + hipThreadIdx_x + 1, j + 1, k) - beta_i(i + hipThreadIdx_x + 1, j + -1, k)) * buf10[0];
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 7; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf16[1 + rel] += (beta_i(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) - beta_i(i + hipThreadIdx_x, j + _cg_idx1, k)) * buf4[1 + rel];
            buf16[1 + rel] += (beta_j(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k) - beta_j(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k)) * buf6[1 + rel];
            buf16[1 + rel] += (beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 1) - beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1)) * buf7[1 + rel];
            buf16[1 + rel] += (beta_k(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k) - beta_k(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k)) * buf8[1 + rel];
            buf16[1 + rel] += (beta_k(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) - beta_k(i + hipThreadIdx_x, j + _cg_idx1, k)) * buf9[1 + rel];
            buf16[1 + rel] += (beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k) - beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k)) * buf10[1 + rel];
            buf16[0 + rel] += (beta_j(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k) - beta_j(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k)) * buf12[0 + rel];
            buf16[0 + rel] += (beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 1) - beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1)) * buf13[0 + rel];
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf16[7] += (beta_j(i + hipThreadIdx_x + 1, j + 8, k) - beta_j(i + hipThreadIdx_x + -1, j + 8, k)) * buf12[7];
        buf16[7] += (beta_j(i + hipThreadIdx_x, j + 8, k + 1) - beta_j(i + hipThreadIdx_x, j + 8, k + -1)) * buf13[7];
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 7; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            buf16[8 + rel] += (beta_i(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1)) * buf4[8 + rel];
            buf16[8 + rel] += (beta_j(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1) - beta_j(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1)) * buf6[8 + rel];
            buf16[8 + rel] += (beta_j(i + hipThreadIdx_x, j, k + _cg_idx2 + 2) - beta_j(i + hipThreadIdx_x, j, k + _cg_idx2)) * buf7[8 + rel];
            buf16[8 + rel] += (beta_k(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1)) * buf8[8 + rel];
            buf16[8 + rel] += (beta_k(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1)) * buf9[8 + rel];
            buf16[8 + rel] += (beta_i(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 1)) * buf10[8 + rel];
            buf16[0 + rel] += (beta_k(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1)) * buf14[0 + rel];
            buf16[0 + rel] += (beta_k(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1)) * buf15[0 + rel];
          }
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 7; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf16[9 + rel] += (beta_i(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1)) * buf4[9 + rel];
                buf16[9 + rel] += (beta_j(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) - beta_j(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1)) * buf6[9 + rel];
                buf16[9 + rel] += (beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 2) - beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2)) * buf7[9 + rel];
                buf16[9 + rel] += (beta_k(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1)) * buf8[9 + rel];
                buf16[9 + rel] += (beta_k(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1)) * buf9[9 + rel];
                buf16[9 + rel] += (beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2 + 1)) * buf10[9 + rel];
                buf16[0 + rel] += (beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + -1)) * buf5[0 + rel];
                buf16[0 + rel] += (beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2 + -1)) * buf11[0 + rel];
                buf16[8 + rel] += (beta_j(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) - beta_j(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1)) * buf12[8 + rel];
                buf16[8 + rel] += (beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 2) - beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2)) * buf13[8 + rel];
                buf16[1 + rel] += (beta_k(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1)) * buf14[1 + rel];
                buf16[1 + rel] += (beta_k(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1)) * buf15[1 + rel];
              }
              _cg_rel1 += 1;
            }
          }
          {
            buf16[7 + rel] += (beta_i(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + -1)) * buf5[7 + rel];
            buf16[7 + rel] += (beta_i(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1) - beta_i(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + -1)) * buf11[7 + rel];
            buf16[15 + rel] += (beta_j(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 1) - beta_j(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 1)) * buf12[15 + rel];
            buf16[15 + rel] += (beta_j(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2) - beta_j(i + hipThreadIdx_x, j + 8, k + _cg_idx2)) * buf13[15 + rel];
          }
          _cg_rel2 += 8;
        }
      }
      {
        buf16[56] += (beta_k(i + hipThreadIdx_x + 1, j, k + 8) - beta_k(i + hipThreadIdx_x + -1, j, k + 8)) * buf14[56];
        buf16[56] += (beta_k(i + hipThreadIdx_x, j + 1, k + 8) - beta_k(i + hipThreadIdx_x, j + -1, k + 8)) * buf15[56];
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 7; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf16[56 + rel] += (beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + 8) - beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + 6)) * buf5[56 + rel];
            buf16[56 + rel] += (beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + 8) - beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + 6)) * buf11[56 + rel];
            buf16[57 + rel] += (beta_k(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 8) - beta_k(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 8)) * buf14[57 + rel];
            buf16[57 + rel] += (beta_k(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 8) - beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + 8)) * buf15[57 + rel];
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf16[63] += (beta_i(i + hipThreadIdx_x, j + 7, k + 8) - beta_i(i + hipThreadIdx_x, j + 7, k + 6)) * buf5[63];
        buf16[63] += (beta_i(i + hipThreadIdx_x + 1, j + 7, k + 8) - beta_i(i + hipThreadIdx_x + 1, j + 7, k + 6)) * buf11[63];
      }
    }
  }
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf15[0 + rel] = c[0] * alpha(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) - c[1] * c[2] * (0.0833 * (beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * (15.0 * x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + _cg_idx2) - 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) - x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + _cg_idx2) + x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2)) + beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2) * (15.0 * x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2) - 15.0 * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) - x(i + hipThreadIdx_x + 2, j + _cg_idx1, k + _cg_idx2) + x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + _cg_idx2)) + beta_j(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * buf0[0 + rel] + beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2) * buf1[0 + rel] + beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * buf2[0 + rel] + beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1) * buf3[0 + rel]) + 0.020825 * buf16[0 + rel]);
              }
              _cg_rel1 += 1;
            }
          }
          _cg_rel2 += 8;
        }
      }
    }
    {
      long rel = 0;
      for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
      {
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          for (long _cg_idx0 = hipThreadIdx_x; _cg_idx0 < 64; _cg_idx0 += 64, ++rel)
          {
            out(i + _cg_idx0, j + _cg_idx1, k + _cg_idx2) = buf15[rel];
          }
        }
      }
    }
  }
}
# 380 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/helmholtz4/intermediate_gen/helmholtz4.cu" 2

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
# 1 "VSBrick-helmholtz45.py-HIP-8x8x8-8x8" 1
{
  auto *binfo = out.bInfo;
  long neighbor0 = binfo->adj[b][0];
  long neighbor1 = binfo->adj[b][1];
  long neighbor2 = binfo->adj[b][2];
  long neighbor3 = binfo->adj[b][3];
  long neighbor4 = binfo->adj[b][4];
  long neighbor5 = binfo->adj[b][5];
  long neighbor6 = binfo->adj[b][6];
  long neighbor7 = binfo->adj[b][7];
  long neighbor8 = binfo->adj[b][8];
  long neighbor9 = binfo->adj[b][9];
  long neighbor10 = binfo->adj[b][10];
  long neighbor11 = binfo->adj[b][11];
  long neighbor12 = binfo->adj[b][12];
  long neighbor13 = b;
  long neighbor14 = binfo->adj[b][14];
  long neighbor15 = binfo->adj[b][15];
  long neighbor16 = binfo->adj[b][16];
  long neighbor17 = binfo->adj[b][17];
  long neighbor18 = binfo->adj[b][18];
  long neighbor19 = binfo->adj[b][19];
  long neighbor20 = binfo->adj[b][20];
  long neighbor21 = binfo->adj[b][21];
  long neighbor22 = binfo->adj[b][22];
  long neighbor23 = binfo->adj[b][23];
  long neighbor24 = binfo->adj[b][24];
  long neighbor25 = binfo->adj[b][25];
  long neighbor26 = binfo->adj[b][26];
  bElem buf0[8];
  bElem buf1[8];
  bElem buf2[8];
  bElem buf3[8];
  bElem buf4[8];
  bElem buf5[8];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 0]
            buf0[0 + rel] = 0;
            buf1[0 + rel] = 0;
            buf2[0 + rel] = 0;
            buf3[0 + rel] = 0;
            buf4[0 + rel] = 0;
            buf5[0 + rel] = 0;
          }
          _cg_rel2 += 1;
        }
      }
    }
    {
      bElem _cg_x_100_vecbuf;
      bElem _cg_x000_vecbuf;
      {
        // New offset [1, 0, -2]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor4 * x.step + 384 + hipThreadIdx_x];
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf4[0] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 0, -1]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf4[0] += 15.0 * _cg_x_100_reg;
        buf4[1] -= _cg_x_100_reg;
        buf5[0] += _cg_x_100_reg;
      }
      {
        // New offset [1, -1, 0]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[0] += 15.0 * _cg_x_100_reg;
        buf2[0] -= _cg_x_100_reg;
        buf3[0] += _cg_x_100_reg;
      }
      {
        // New offset [0, 0, 0]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[0] += 15.0 * _cg_x_100_reg;
        buf0[0] -= _cg_x_100_reg;
        buf1[0] += _cg_x_100_reg;
      }
      {
        // New offset [1, 0, 0]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = _cg_x000_vecbuf;
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf0[0] -= 15.0 * _cg_x_100_reg;
        buf1[0] -= 15.0 * _cg_x_100_reg;
        buf2[0] -= 15.0 * _cg_x_100_reg;
        buf3[0] -= 15.0 * _cg_x_100_reg;
        buf4[0] -= 15.0 * _cg_x_100_reg;
        buf4[1] += 15.0 * _cg_x_100_reg;
        buf4[2] -= _cg_x_100_reg;
        buf5[0] -= 15.0 * _cg_x_100_reg;
        buf5[1] += _cg_x_100_reg;
      }
      {
        // New offset [2, 0, 0]
        bElem _cg_x_100_reg;
        {
          _cg_x000_vecbuf = x.dat[neighbor14 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[0] += _cg_x_100_reg;
        buf1[0] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [3, 0, 0]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf1[0] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 1, 0]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[0] += _cg_x_100_reg;
        buf3[0] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [1, 2, 0]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf3[0] -= _cg_x_100_reg;
      }
      {
        // New offset [1, -1, 1]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[1] += 15.0 * _cg_x_100_reg;
        buf2[1] -= _cg_x_100_reg;
        buf3[1] += _cg_x_100_reg;
      }
      {
        // New offset [0, 0, 1]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + 64 + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor13 * x.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[1] += 15.0 * _cg_x_100_reg;
        buf0[1] -= _cg_x_100_reg;
        buf1[1] += _cg_x_100_reg;
      }
      {
        // New offset [1, 0, 1]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = _cg_x000_vecbuf;
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf0[1] -= 15.0 * _cg_x_100_reg;
        buf1[1] -= 15.0 * _cg_x_100_reg;
        buf2[1] -= 15.0 * _cg_x_100_reg;
        buf3[1] -= 15.0 * _cg_x_100_reg;
        buf4[1] -= 15.0 * _cg_x_100_reg;
        buf4[2] += 15.0 * _cg_x_100_reg;
        buf4[3] -= _cg_x_100_reg;
        buf4[0] += _cg_x_100_reg;
        buf5[1] -= 15.0 * _cg_x_100_reg;
        buf5[2] += _cg_x_100_reg;
        buf5[0] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [2, 0, 1]
        bElem _cg_x_100_reg;
        {
          _cg_x000_vecbuf = x.dat[neighbor14 * x.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[1] += _cg_x_100_reg;
        buf1[1] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [3, 0, 1]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf1[1] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 1, 1]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[1] += _cg_x_100_reg;
        buf3[1] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [1, 2, 1]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf3[1] -= _cg_x_100_reg;
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 4; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [1, -1, 2]
            bElem _cg_x_100_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor13 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor10 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_100_vecbuf
              dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              _cg_x_100_reg = _cg_x_100_vecbuf;
            }
            buf2[2 + rel] += 15.0 * _cg_x_100_reg;
            buf2[2 + rel] -= _cg_x_100_reg;
            buf3[2 + rel] += _cg_x_100_reg;
          }
          {
            // New offset [0, 0, 2]
            bElem _cg_x_100_reg;
            {
              _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_x000_vecbuf = x.dat[neighbor13 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_x_100_reg = _cg_vectmp0;
            }
            buf0[2 + rel] += 15.0 * _cg_x_100_reg;
            buf0[2 + rel] -= _cg_x_100_reg;
            buf1[2 + rel] += _cg_x_100_reg;
          }
          {
            // New offset [1, 0, 2]
            bElem _cg_x_100_reg;
            {
              _cg_x_100_vecbuf = _cg_x000_vecbuf;
              _cg_x_100_reg = _cg_x_100_vecbuf;
            }
            buf0[2 + rel] -= 15.0 * _cg_x_100_reg;
            buf1[2 + rel] -= 15.0 * _cg_x_100_reg;
            buf2[2 + rel] -= 15.0 * _cg_x_100_reg;
            buf3[2 + rel] -= 15.0 * _cg_x_100_reg;
            buf4[2 + rel] -= 15.0 * _cg_x_100_reg;
            buf4[3 + rel] += 15.0 * _cg_x_100_reg;
            buf4[4 + rel] -= _cg_x_100_reg;
            buf4[1 + rel] += _cg_x_100_reg;
            buf5[2 + rel] -= 15.0 * _cg_x_100_reg;
            buf5[3 + rel] += _cg_x_100_reg;
            buf5[1 + rel] += 15.0 * _cg_x_100_reg;
            buf5[0 + rel] -= _cg_x_100_reg;
          }
          {
            // New offset [2, 0, 2]
            bElem _cg_x_100_reg;
            {
              _cg_x000_vecbuf = x.dat[neighbor14 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_x_100_reg = _cg_vectmp0;
            }
            buf0[2 + rel] += _cg_x_100_reg;
            buf1[2 + rel] += 15.0 * _cg_x_100_reg;
          }
          {
            // New offset [3, 0, 2]
            bElem _cg_x_100_reg;
            {
              bElem _cg_vectmp0;
              // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 2 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 6, 8, hipThreadIdx_x & 7);
              _cg_x_100_reg = _cg_vectmp0;
            }
            buf1[2 + rel] -= _cg_x_100_reg;
          }
          {
            // New offset [1, 1, 2]
            bElem _cg_x_100_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor16 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor13 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_100_vecbuf
              dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              _cg_x_100_reg = _cg_x_100_vecbuf;
            }
            buf2[2 + rel] += _cg_x_100_reg;
            buf3[2 + rel] += 15.0 * _cg_x_100_reg;
          }
          {
            // New offset [1, 2, 2]
            bElem _cg_x_100_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor16 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor13 * x.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_x_100_vecbuf
              dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
              _cg_x_100_reg = _cg_x_100_vecbuf;
            }
            buf3[2 + rel] -= _cg_x_100_reg;
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [1, -1, 6]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[6] += 15.0 * _cg_x_100_reg;
        buf2[6] -= _cg_x_100_reg;
        buf3[6] += _cg_x_100_reg;
      }
      {
        // New offset [0, 0, 6]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + 384 + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor13 * x.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[6] += 15.0 * _cg_x_100_reg;
        buf0[6] -= _cg_x_100_reg;
        buf1[6] += _cg_x_100_reg;
      }
      {
        // New offset [1, 0, 6]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = _cg_x000_vecbuf;
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf0[6] -= 15.0 * _cg_x_100_reg;
        buf1[6] -= 15.0 * _cg_x_100_reg;
        buf2[6] -= 15.0 * _cg_x_100_reg;
        buf3[6] -= 15.0 * _cg_x_100_reg;
        buf4[6] -= 15.0 * _cg_x_100_reg;
        buf4[7] += 15.0 * _cg_x_100_reg;
        buf4[5] += _cg_x_100_reg;
        buf5[6] -= 15.0 * _cg_x_100_reg;
        buf5[7] += _cg_x_100_reg;
        buf5[5] += 15.0 * _cg_x_100_reg;
        buf5[4] -= _cg_x_100_reg;
      }
      {
        // New offset [2, 0, 6]
        bElem _cg_x_100_reg;
        {
          _cg_x000_vecbuf = x.dat[neighbor14 * x.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[6] += _cg_x_100_reg;
        buf1[6] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [3, 0, 6]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf1[6] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 1, 6]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[6] += _cg_x_100_reg;
        buf3[6] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [1, 2, 6]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf3[6] -= _cg_x_100_reg;
      }
      {
        // New offset [1, -1, 7]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[7] += 15.0 * _cg_x_100_reg;
        buf2[7] -= _cg_x_100_reg;
        buf3[7] += _cg_x_100_reg;
      }
      {
        // New offset [0, 0, 7]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + 448 + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[7] += 15.0 * _cg_x_100_reg;
        buf0[7] -= _cg_x_100_reg;
        buf1[7] += _cg_x_100_reg;
      }
      {
        // New offset [1, 0, 7]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = _cg_x000_vecbuf;
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf0[7] -= 15.0 * _cg_x_100_reg;
        buf1[7] -= 15.0 * _cg_x_100_reg;
        buf2[7] -= 15.0 * _cg_x_100_reg;
        buf3[7] -= 15.0 * _cg_x_100_reg;
        buf4[7] -= 15.0 * _cg_x_100_reg;
        buf4[6] += _cg_x_100_reg;
        buf5[7] -= 15.0 * _cg_x_100_reg;
        buf5[6] += 15.0 * _cg_x_100_reg;
        buf5[5] -= _cg_x_100_reg;
      }
      {
        // New offset [2, 0, 7]
        bElem _cg_x_100_reg;
        {
          _cg_x000_vecbuf = x.dat[neighbor14 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_x_100_reg;
        buf1[7] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [3, 0, 7]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_x_100_reg = _cg_vectmp0;
        }
        buf1[7] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 1, 7]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf2[7] += _cg_x_100_reg;
        buf3[7] += 15.0 * _cg_x_100_reg;
      }
      {
        // New offset [1, 2, 7]
        bElem _cg_x_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_x_100_vecbuf
          dev_shl(_cg_x_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf3[7] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 0, 8]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf4[7] += _cg_x_100_reg;
        buf5[7] += 15.0 * _cg_x_100_reg;
        buf5[6] -= _cg_x_100_reg;
      }
      {
        // New offset [1, 0, 9]
        bElem _cg_x_100_reg;
        {
          _cg_x_100_vecbuf = x.dat[neighbor22 * x.step + 64 + hipThreadIdx_x];
          _cg_x_100_reg = _cg_x_100_vecbuf;
        }
        buf5[7] -= _cg_x_100_reg;
      }
    }
  }
  bElem buf6[8];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 0]
            buf6[0 + rel] = 0;
          }
          _cg_rel2 += 1;
        }
      }
    }
    {
      bElem _cg_beta_i000_vecbuf;
      bElem _cg_beta_i_100_vecbuf;
      bElem _cg_beta_j000_vecbuf;
      bElem _cg_beta_k000_vecbuf;
      {
        // New offset [0, 0, 0]
        bElem _cg_beta_i000_reg;
        bElem _cg_beta_j000_reg;
        bElem _cg_beta_k000_reg;
        {
          _cg_beta_i000_vecbuf = beta_i.dat[neighbor13 * beta_i.step + hipThreadIdx_x];
          _cg_beta_j000_vecbuf = beta_j.dat[neighbor13 * beta_j.step + hipThreadIdx_x];
          _cg_beta_k000_vecbuf = beta_k.dat[neighbor13 * beta_k.step + hipThreadIdx_x];
          _cg_beta_i000_reg = _cg_beta_i000_vecbuf;
          _cg_beta_j000_reg = _cg_beta_j000_vecbuf;
          _cg_beta_k000_reg = _cg_beta_k000_vecbuf;
        }
        buf6[0] += _cg_beta_i000_reg * buf0[0];
        buf6[0] += _cg_beta_j000_reg * buf2[0];
        buf6[0] += _cg_beta_k000_reg * buf4[0];
      }
      {
        // New offset [1, 0, 0]
        bElem _cg_beta_i000_reg;
        {
          _cg_beta_i_100_vecbuf = _cg_beta_i000_vecbuf;
          _cg_beta_i000_vecbuf = beta_i.dat[neighbor14 * beta_i.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_beta_i_100_vecbuf ,_cg_beta_i000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_beta_i_100_vecbuf, _cg_beta_i000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i000_reg = _cg_vectmp0;
        }
        buf6[0] += _cg_beta_i000_reg * buf1[0];
      }
      {
        // New offset [0, 1, 0]
        bElem _cg_beta_j000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_j.dat[neighbor16 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_j.dat[neighbor13 * beta_j.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j000_vecbuf
          dev_shl(_cg_beta_j000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_beta_j000_reg = _cg_beta_j000_vecbuf;
        }
        buf6[0] += _cg_beta_j000_reg * buf3[0];
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 7; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 1]
            bElem _cg_beta_i000_reg;
            bElem _cg_beta_j000_reg;
            bElem _cg_beta_k000_reg;
            {
              _cg_beta_i000_vecbuf = beta_i.dat[neighbor13 * beta_i.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_j000_vecbuf = beta_j.dat[neighbor13 * beta_j.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_k000_vecbuf = beta_k.dat[neighbor13 * beta_k.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_i000_reg = _cg_beta_i000_vecbuf;
              _cg_beta_j000_reg = _cg_beta_j000_vecbuf;
              _cg_beta_k000_reg = _cg_beta_k000_vecbuf;
            }
            buf6[1 + rel] += _cg_beta_i000_reg * buf0[1 + rel];
            buf6[1 + rel] += _cg_beta_j000_reg * buf2[1 + rel];
            buf6[1 + rel] += _cg_beta_k000_reg * buf4[1 + rel];
            buf6[0 + rel] += _cg_beta_k000_reg * buf5[0 + rel];
          }
          {
            // New offset [1, 0, 1]
            bElem _cg_beta_i000_reg;
            {
              _cg_beta_i_100_vecbuf = _cg_beta_i000_vecbuf;
              _cg_beta_i000_vecbuf = beta_i.dat[neighbor14 * beta_i.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_beta_i_100_vecbuf ,_cg_beta_i000_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_beta_i_100_vecbuf, _cg_beta_i000_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_i000_reg = _cg_vectmp0;
            }
            buf6[1 + rel] += _cg_beta_i000_reg * buf1[1 + rel];
          }
          {
            // New offset [0, 1, 1]
            bElem _cg_beta_j000_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = beta_j.dat[neighbor16 * beta_j.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = beta_j.dat[neighbor13 * beta_j.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j000_vecbuf
              dev_shl(_cg_beta_j000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              _cg_beta_j000_reg = _cg_beta_j000_vecbuf;
            }
            buf6[1 + rel] += _cg_beta_j000_reg * buf3[1 + rel];
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [0, 0, 8]
        bElem _cg_beta_k000_reg;
        {
          _cg_beta_k000_vecbuf = beta_k.dat[neighbor22 * beta_k.step + hipThreadIdx_x];
          _cg_beta_k000_reg = _cg_beta_k000_vecbuf;
        }
        buf6[7] += _cg_beta_k000_reg * buf5[7];
      }
    }
  }
  bElem buf7[8];
  bElem buf8[8];
  bElem buf9[8];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 0]
            buf5[0 + rel] = 0;
            buf4[0 + rel] = 0;
            buf3[0 + rel] = 0;
            buf2[0 + rel] = 0;
            buf1[0 + rel] = 0;
            buf0[0 + rel] = 0;
            buf7[0 + rel] = 0;
            buf8[0 + rel] = 0;
            buf9[0 + rel] = 0;
          }
          _cg_rel2 += 1;
        }
      }
    }
    {
      bElem _cg_x_110_vecbuf;
      bElem _cg_x010_vecbuf;
      {
        // New offset [1, -2, -1]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor1 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf2[0] -= _cg_x_110_reg;
        buf0[0] -= _cg_x_110_reg;
      }
      {
        // New offset [0, -1, -1]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = x.dat[neighbor3 * x.step + 448 + hipThreadIdx_x];
          _cg_x010_vecbuf = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf4[0] -= _cg_x_110_reg;
        buf1[0] -= _cg_x_110_reg;
      }
      {
        // New offset [1, -1, -1]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf4[0] += _cg_x_110_reg;
        buf2[0] += _cg_x_110_reg;
        buf8[0] += _cg_x_110_reg;
      }
      {
        // New offset [2, -1, -1]
        bElem _cg_x_110_reg;
        {
          _cg_x010_vecbuf = x.dat[neighbor5 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf1[0] += _cg_x_110_reg;
        buf8[0] -= _cg_x_110_reg;
      }
      {
        // New offset [1, 0, -1]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor7 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf0[0] += _cg_x_110_reg;
      }
      {
        // New offset [0, -2, 0]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor12 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor9 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = x.dat[neighbor10 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp4;
        }
        buf5[0] -= _cg_x_110_reg;
        buf3[0] -= _cg_x_110_reg;
      }
      {
        // New offset [1, -2, 0]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf5[0] += _cg_x_110_reg;
        buf0[0] += _cg_x_110_reg;
        buf0[1] -= _cg_x_110_reg;
        buf7[0] += _cg_x_110_reg;
        buf2[1] -= _cg_x_110_reg;
      }
      {
        // New offset [2, -2, 0]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor14 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor11 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp2;
        }
        buf3[0] += _cg_x_110_reg;
        buf7[0] -= _cg_x_110_reg;
      }
      {
        // New offset [0, -1, 0]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = x.dat[neighbor12 * x.step + hipThreadIdx_x];
          _cg_x010_vecbuf = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf4[1] -= _cg_x_110_reg;
        buf1[1] -= _cg_x_110_reg;
        buf1[0] += _cg_x_110_reg;
        buf3[0] += _cg_x_110_reg;
        buf9[0] += _cg_x_110_reg;
      }
      {
        // New offset [1, -1, 0]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf4[1] += _cg_x_110_reg;
        buf2[1] += _cg_x_110_reg;
        buf8[1] += _cg_x_110_reg;
      }
      {
        // New offset [2, -1, 0]
        bElem _cg_x_110_reg;
        {
          _cg_x010_vecbuf = x.dat[neighbor14 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf3[0] -= _cg_x_110_reg;
        buf1[0] -= _cg_x_110_reg;
        buf1[1] += _cg_x_110_reg;
        buf9[0] -= _cg_x_110_reg;
        buf8[1] -= _cg_x_110_reg;
      }
      {
        // New offset [0, 0, 0]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor15 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor12 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = x.dat[neighbor16 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp4;
        }
        buf5[0] += _cg_x_110_reg;
        buf9[0] -= _cg_x_110_reg;
      }
      {
        // New offset [1, 0, 0]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf5[0] -= _cg_x_110_reg;
        buf0[0] -= _cg_x_110_reg;
        buf0[1] += _cg_x_110_reg;
        buf7[0] -= _cg_x_110_reg;
      }
      {
        // New offset [2, 0, 0]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor17 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor14 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp2;
        }
        buf7[0] += _cg_x_110_reg;
        buf9[0] += _cg_x_110_reg;
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 6; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, -2, 1]
            bElem _cg_x_110_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor12 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor9 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_110_vecbuf
              dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = x.dat[neighbor10 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_x010_vecbuf
              dev_shl(_cg_x010_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_x_110_reg = _cg_vectmp4;
            }
            buf5[1 + rel] -= _cg_x_110_reg;
            buf3[1 + rel] -= _cg_x_110_reg;
          }
          {
            // New offset [1, -2, 1]
            bElem _cg_x_110_reg;
            {
              _cg_x_110_vecbuf = _cg_x010_vecbuf;
              _cg_x_110_reg = _cg_x_110_vecbuf;
            }
            buf5[1 + rel] += _cg_x_110_reg;
            buf0[1 + rel] += _cg_x_110_reg;
            buf0[2 + rel] -= _cg_x_110_reg;
            buf7[1 + rel] += _cg_x_110_reg;
            buf2[0 + rel] += _cg_x_110_reg;
            buf2[2 + rel] -= _cg_x_110_reg;
          }
          {
            // New offset [2, -2, 1]
            bElem _cg_x_110_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor14 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor11 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x010_vecbuf
              dev_shl(_cg_x010_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_x_110_reg = _cg_vectmp2;
            }
            buf3[1 + rel] += _cg_x_110_reg;
            buf7[1 + rel] -= _cg_x_110_reg;
          }
          {
            // New offset [0, -1, 1]
            bElem _cg_x_110_reg;
            {
              _cg_x_110_vecbuf = x.dat[neighbor12 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_x010_vecbuf = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_x_110_reg = _cg_vectmp0;
            }
            buf4[0 + rel] += _cg_x_110_reg;
            buf4[2 + rel] -= _cg_x_110_reg;
            buf1[2 + rel] -= _cg_x_110_reg;
            buf1[1 + rel] += _cg_x_110_reg;
            buf3[1 + rel] += _cg_x_110_reg;
            buf9[1 + rel] += _cg_x_110_reg;
          }
          {
            // New offset [1, -1, 1]
            bElem _cg_x_110_reg;
            {
              _cg_x_110_vecbuf = _cg_x010_vecbuf;
              _cg_x_110_reg = _cg_x_110_vecbuf;
            }
            buf4[0 + rel] -= _cg_x_110_reg;
            buf4[2 + rel] += _cg_x_110_reg;
            buf2[0 + rel] -= _cg_x_110_reg;
            buf2[2 + rel] += _cg_x_110_reg;
            buf8[0 + rel] -= _cg_x_110_reg;
            buf8[2 + rel] += _cg_x_110_reg;
          }
          {
            // New offset [2, -1, 1]
            bElem _cg_x_110_reg;
            {
              _cg_x010_vecbuf = x.dat[neighbor14 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_x_110_reg = _cg_vectmp0;
            }
            buf3[1 + rel] -= _cg_x_110_reg;
            buf1[1 + rel] -= _cg_x_110_reg;
            buf1[2 + rel] += _cg_x_110_reg;
            buf9[1 + rel] -= _cg_x_110_reg;
            buf8[2 + rel] -= _cg_x_110_reg;
            buf8[0 + rel] += _cg_x_110_reg;
          }
          {
            // New offset [0, 0, 1]
            bElem _cg_x_110_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor15 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor12 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_110_vecbuf
              dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = x.dat[neighbor16 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_x010_vecbuf
              dev_shl(_cg_x010_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_x_110_reg = _cg_vectmp4;
            }
            buf5[1 + rel] += _cg_x_110_reg;
            buf9[1 + rel] -= _cg_x_110_reg;
          }
          {
            // New offset [1, 0, 1]
            bElem _cg_x_110_reg;
            {
              _cg_x_110_vecbuf = _cg_x010_vecbuf;
              _cg_x_110_reg = _cg_x_110_vecbuf;
            }
            buf5[1 + rel] -= _cg_x_110_reg;
            buf0[1 + rel] -= _cg_x_110_reg;
            buf0[2 + rel] += _cg_x_110_reg;
            buf7[1 + rel] -= _cg_x_110_reg;
          }
          {
            // New offset [2, 0, 1]
            bElem _cg_x_110_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor17 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor14 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x010_vecbuf
              dev_shl(_cg_x010_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_x_110_reg = _cg_vectmp2;
            }
            buf7[1 + rel] += _cg_x_110_reg;
            buf9[1 + rel] += _cg_x_110_reg;
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [0, -2, 7]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor12 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor9 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = x.dat[neighbor10 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp4;
        }
        buf5[7] -= _cg_x_110_reg;
        buf3[7] -= _cg_x_110_reg;
      }
      {
        // New offset [1, -2, 7]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf5[7] += _cg_x_110_reg;
        buf0[7] += _cg_x_110_reg;
        buf7[7] += _cg_x_110_reg;
        buf2[6] += _cg_x_110_reg;
      }
      {
        // New offset [2, -2, 7]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor14 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor11 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp2;
        }
        buf3[7] += _cg_x_110_reg;
        buf7[7] -= _cg_x_110_reg;
      }
      {
        // New offset [0, -1, 7]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = x.dat[neighbor12 * x.step + 448 + hipThreadIdx_x];
          _cg_x010_vecbuf = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf4[6] += _cg_x_110_reg;
        buf3[7] += _cg_x_110_reg;
        buf1[7] += _cg_x_110_reg;
        buf9[7] += _cg_x_110_reg;
      }
      {
        // New offset [1, -1, 7]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf4[6] -= _cg_x_110_reg;
        buf2[6] -= _cg_x_110_reg;
        buf8[6] -= _cg_x_110_reg;
      }
      {
        // New offset [2, -1, 7]
        bElem _cg_x_110_reg;
        {
          _cg_x010_vecbuf = x.dat[neighbor14 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf3[7] -= _cg_x_110_reg;
        buf1[7] -= _cg_x_110_reg;
        buf9[7] -= _cg_x_110_reg;
        buf8[6] += _cg_x_110_reg;
      }
      {
        // New offset [0, 0, 7]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor15 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor12 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = x.dat[neighbor16 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp4;
        }
        buf5[7] += _cg_x_110_reg;
        buf9[7] -= _cg_x_110_reg;
      }
      {
        // New offset [1, 0, 7]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf5[7] -= _cg_x_110_reg;
        buf0[7] -= _cg_x_110_reg;
        buf7[7] -= _cg_x_110_reg;
      }
      {
        // New offset [2, 0, 7]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor17 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor14 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x010_vecbuf
          dev_shl(_cg_x010_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp2;
        }
        buf7[7] += _cg_x_110_reg;
        buf9[7] += _cg_x_110_reg;
      }
      {
        // New offset [1, -2, 8]
        bElem _cg_x_110_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor19 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x_110_vecbuf
          dev_shl(_cg_x_110_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf2[7] += _cg_x_110_reg;
      }
      {
        // New offset [0, -1, 8]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = x.dat[neighbor21 * x.step + hipThreadIdx_x];
          _cg_x010_vecbuf = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf4[7] += _cg_x_110_reg;
      }
      {
        // New offset [1, -1, 8]
        bElem _cg_x_110_reg;
        {
          _cg_x_110_vecbuf = _cg_x010_vecbuf;
          _cg_x_110_reg = _cg_x_110_vecbuf;
        }
        buf4[7] -= _cg_x_110_reg;
        buf2[7] -= _cg_x_110_reg;
        buf8[7] -= _cg_x_110_reg;
      }
      {
        // New offset [2, -1, 8]
        bElem _cg_x_110_reg;
        {
          _cg_x010_vecbuf = x.dat[neighbor23 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_110_vecbuf ,_cg_x010_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_110_vecbuf, _cg_x010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x_110_reg = _cg_vectmp0;
        }
        buf8[7] += _cg_x_110_reg;
      }
    }
  }
  bElem buf10[8];
  bElem buf11[8];
  bElem buf12[8];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 0]
            buf10[0 + rel] = 0;
            buf11[0 + rel] = 0;
            buf12[0 + rel] = 0;
          }
          _cg_rel2 += 1;
        }
      }
    }
    {
      bElem _cg_x011_vecbuf;
      bElem _cg_x_111_vecbuf;
      {
        // New offset [0, -1, -2]
        bElem _cg_x011_reg;
        {
          _cg_x011_vecbuf = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[0] += _cg_x011_reg;
      }
      {
        // New offset [0, 0, -2]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor7 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[0] -= _cg_x011_reg;
      }
      {
        // New offset [0, -2, -1]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf12[0] += _cg_x011_reg;
      }
      {
        // New offset [-1, -1, -1]
        bElem _cg_x011_reg;
        {
          _cg_x_111_vecbuf = x.dat[neighbor12 * x.step + hipThreadIdx_x];
          _cg_x011_vecbuf = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x011_reg = _cg_vectmp0;
        }
        buf11[0] += _cg_x011_reg;
      }
      {
        // New offset [0, -1, -1]
        bElem _cg_x011_reg;
        {
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[1] += _cg_x011_reg;
      }
      {
        // New offset [1, -1, -1]
        bElem _cg_x011_reg;
        {
          _cg_x_111_vecbuf = _cg_x011_vecbuf;
          _cg_x011_vecbuf = x.dat[neighbor14 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x011_reg = _cg_vectmp0;
        }
        buf11[0] -= _cg_x011_reg;
      }
      {
        // New offset [0, 0, -1]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[1] -= _cg_x011_reg;
        buf12[0] -= _cg_x011_reg;
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 6; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, -2, 0]
            bElem _cg_x011_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor10 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x011_vecbuf
              dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              _cg_x011_reg = _cg_x011_vecbuf;
            }
            buf12[0 + rel] -= _cg_x011_reg;
            buf12[1 + rel] += _cg_x011_reg;
          }
          {
            // New offset [-1, -1, 0]
            bElem _cg_x011_reg;
            {
              _cg_x_111_vecbuf = x.dat[neighbor12 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_x011_vecbuf = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_x011_reg = _cg_vectmp0;
            }
            buf11[0 + rel] -= _cg_x011_reg;
            buf11[1 + rel] += _cg_x011_reg;
          }
          {
            // New offset [0, -1, 0]
            bElem _cg_x011_reg;
            {
              _cg_x011_reg = _cg_x011_vecbuf;
            }
            buf10[0 + rel] -= _cg_x011_reg;
            buf10[2 + rel] += _cg_x011_reg;
          }
          {
            // New offset [1, -1, 0]
            bElem _cg_x011_reg;
            {
              _cg_x_111_vecbuf = _cg_x011_vecbuf;
              _cg_x011_vecbuf = x.dat[neighbor14 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_x011_reg = _cg_vectmp0;
            }
            buf11[0 + rel] += _cg_x011_reg;
            buf11[1 + rel] -= _cg_x011_reg;
          }
          {
            // New offset [0, 0, 0]
            bElem _cg_x011_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor16 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x011_vecbuf
              dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              _cg_x011_reg = _cg_x011_vecbuf;
            }
            buf10[0 + rel] += _cg_x011_reg;
            buf10[2 + rel] -= _cg_x011_reg;
            buf12[0 + rel] += _cg_x011_reg;
            buf12[1 + rel] -= _cg_x011_reg;
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [0, -2, 6]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf12[6] -= _cg_x011_reg;
        buf12[7] += _cg_x011_reg;
      }
      {
        // New offset [-1, -1, 6]
        bElem _cg_x011_reg;
        {
          _cg_x_111_vecbuf = x.dat[neighbor12 * x.step + 448 + hipThreadIdx_x];
          _cg_x011_vecbuf = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x011_reg = _cg_vectmp0;
        }
        buf11[6] -= _cg_x011_reg;
        buf11[7] += _cg_x011_reg;
      }
      {
        // New offset [0, -1, 6]
        bElem _cg_x011_reg;
        {
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[6] -= _cg_x011_reg;
      }
      {
        // New offset [1, -1, 6]
        bElem _cg_x011_reg;
        {
          _cg_x_111_vecbuf = _cg_x011_vecbuf;
          _cg_x011_vecbuf = x.dat[neighbor14 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x011_reg = _cg_vectmp0;
        }
        buf11[6] += _cg_x011_reg;
        buf11[7] -= _cg_x011_reg;
      }
      {
        // New offset [0, 0, 6]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[6] += _cg_x011_reg;
        buf12[6] += _cg_x011_reg;
        buf12[7] -= _cg_x011_reg;
      }
      {
        // New offset [0, -2, 7]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor19 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf12[7] -= _cg_x011_reg;
      }
      {
        // New offset [-1, -1, 7]
        bElem _cg_x011_reg;
        {
          _cg_x_111_vecbuf = x.dat[neighbor21 * x.step + hipThreadIdx_x];
          _cg_x011_vecbuf = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x011_reg = _cg_vectmp0;
        }
        buf11[7] -= _cg_x011_reg;
      }
      {
        // New offset [0, -1, 7]
        bElem _cg_x011_reg;
        {
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[7] -= _cg_x011_reg;
      }
      {
        // New offset [1, -1, 7]
        bElem _cg_x011_reg;
        {
          _cg_x_111_vecbuf = _cg_x011_vecbuf;
          _cg_x011_vecbuf = x.dat[neighbor23 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_x_111_vecbuf ,_cg_x011_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_111_vecbuf, _cg_x011_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x011_reg = _cg_vectmp0;
        }
        buf11[7] += _cg_x011_reg;
      }
      {
        // New offset [0, 0, 7]
        bElem _cg_x011_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor25 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x011_vecbuf
          dev_shl(_cg_x011_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_x011_reg = _cg_x011_vecbuf;
        }
        buf10[7] += _cg_x011_reg;
        buf12[7] += _cg_x011_reg;
      }
    }
  }
  bElem buf13[8];
  {
    {
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 0]
            buf13[0 + rel] = 0;
          }
          _cg_rel2 += 1;
        }
      }
    }
    {
      bElem _cg_beta_i010_vecbuf;
      bElem _cg_beta_i0_10_vecbuf;
      bElem _cg_beta_i01_2_vecbuf;
      bElem _cg_beta_i_110_vecbuf;
      bElem _cg_beta_i_1_10_vecbuf;
      bElem _cg_beta_i_11_2_vecbuf;
      bElem _cg_beta_j000_vecbuf;
      bElem _cg_beta_j100_vecbuf;
      bElem _cg_beta_j_100_vecbuf;
      bElem _cg_beta_j00_2_vecbuf;
      bElem _cg_beta_k000_vecbuf;
      bElem _cg_beta_k100_vecbuf;
      bElem _cg_beta_k_100_vecbuf;
      bElem _cg_beta_k0_20_vecbuf;
      {
        // New offset [0, 0, 0]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i0_10_reg;
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j_100_reg;
        bElem _cg_beta_k100_reg;
        bElem _cg_beta_k_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_i.dat[neighbor13 * beta_i.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_i.dat[neighbor10 * beta_i.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_i0_10_vecbuf
          dev_shl(_cg_beta_i0_10_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_i.dat[neighbor16 * beta_i.step + hipThreadIdx_x];
          // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_i010_vecbuf
          dev_shl(_cg_beta_i010_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          _cg_beta_j_100_vecbuf = beta_j.dat[neighbor12 * beta_j.step + hipThreadIdx_x];
          _cg_beta_j000_vecbuf = beta_j.dat[neighbor13 * beta_j.step + hipThreadIdx_x];
          _cg_beta_j100_vecbuf = beta_j.dat[neighbor14 * beta_j.step + hipThreadIdx_x];
          _cg_beta_k_100_vecbuf = beta_k.dat[neighbor12 * beta_k.step + hipThreadIdx_x];
          _cg_beta_k000_vecbuf = beta_k.dat[neighbor13 * beta_k.step + hipThreadIdx_x];
          _cg_beta_k100_vecbuf = beta_k.dat[neighbor14 * beta_k.step + hipThreadIdx_x];
          _cg_beta_i010_reg = _cg_beta_i010_vecbuf;
          _cg_beta_i0_10_reg = _cg_beta_i0_10_vecbuf;
          bElem _cg_vectmp3;
          // merge0 _cg_beta_j000_vecbuf ,_cg_beta_j100_vecbuf, 1 -> _cg_vectmp3
          dev_shl(_cg_vectmp3, _cg_beta_j000_vecbuf, _cg_beta_j100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_j100_reg = _cg_vectmp3;
          bElem _cg_vectmp4;
          // merge0 _cg_beta_j_100_vecbuf ,_cg_beta_j000_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_beta_j_100_vecbuf, _cg_beta_j000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_j_100_reg = _cg_vectmp4;
          bElem _cg_vectmp5;
          // merge0 _cg_beta_k000_vecbuf ,_cg_beta_k100_vecbuf, 1 -> _cg_vectmp5
          dev_shl(_cg_vectmp5, _cg_beta_k000_vecbuf, _cg_beta_k100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_k100_reg = _cg_vectmp5;
          bElem _cg_vectmp6;
          // merge0 _cg_beta_k_100_vecbuf ,_cg_beta_k000_vecbuf, 7 -> _cg_vectmp6
          dev_shl(_cg_vectmp6, _cg_beta_k_100_vecbuf, _cg_beta_k000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_k_100_reg = _cg_vectmp6;
        }
        buf13[0] += (_cg_beta_i010_reg - _cg_beta_i0_10_reg) * buf5[0];
        buf13[0] += (_cg_beta_j100_reg - _cg_beta_j_100_reg) * buf3[0];
        buf13[0] += (_cg_beta_k100_reg - _cg_beta_k_100_reg) * buf1[0];
      }
      {
        // New offset [1, 0, 0]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i0_10_reg;
        {
          _cg_beta_i_1_10_vecbuf = _cg_beta_i0_10_vecbuf;
          _cg_beta_i_110_vecbuf = _cg_beta_i010_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_i.dat[neighbor14 * beta_i.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_i.dat[neighbor11 * beta_i.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_i0_10_vecbuf
          dev_shl(_cg_beta_i0_10_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_i.dat[neighbor17 * beta_i.step + hipThreadIdx_x];
          // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_i010_vecbuf
          dev_shl(_cg_beta_i010_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp3;
          // merge0 _cg_beta_i_110_vecbuf ,_cg_beta_i010_vecbuf, 1 -> _cg_vectmp3
          dev_shl(_cg_vectmp3, _cg_beta_i_110_vecbuf, _cg_beta_i010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i010_reg = _cg_vectmp3;
          bElem _cg_vectmp4;
          // merge0 _cg_beta_i_1_10_vecbuf ,_cg_beta_i0_10_vecbuf, 1 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_beta_i_1_10_vecbuf, _cg_beta_i0_10_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i0_10_reg = _cg_vectmp4;
        }
        buf13[0] += (_cg_beta_i010_reg - _cg_beta_i0_10_reg) * buf7[0];
      }
      {
        // New offset [-1, 1, 0]
        bElem _cg_beta_k100_reg;
        bElem _cg_beta_k1_20_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_k.dat[neighbor13 * beta_k.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_k.dat[neighbor10 * beta_k.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_k0_20_vecbuf
          dev_shl(_cg_beta_k0_20_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_k.dat[neighbor16 * beta_k.step + hipThreadIdx_x];
          // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_k000_vecbuf
          dev_shl(_cg_beta_k000_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
          _cg_beta_k1_20_reg = _cg_beta_k0_20_vecbuf;
        }
        buf13[0] += (_cg_beta_k100_reg - _cg_beta_k1_20_reg) * buf0[0];
      }
      {
        // New offset [0, 1, 0]
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_j.dat[neighbor15 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_j.dat[neighbor12 * beta_j.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j_100_vecbuf
          dev_shl(_cg_beta_j_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_j.dat[neighbor16 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = beta_j.dat[neighbor13 * beta_j.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_beta_j000_vecbuf
          dev_shl(_cg_beta_j000_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          _cg_vectmp4 = beta_j.dat[neighbor17 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp5;
          _cg_vectmp5 = beta_j.dat[neighbor14 * beta_j.step + hipThreadIdx_x];
          // merge1 _cg_vectmp5 ,_cg_vectmp4, 1 -> _cg_beta_j100_vecbuf
          dev_shl(_cg_beta_j100_vecbuf, _cg_vectmp5, _cg_vectmp4, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp6;
          // merge0 _cg_beta_j000_vecbuf ,_cg_beta_j100_vecbuf, 1 -> _cg_vectmp6
          dev_shl(_cg_vectmp6, _cg_beta_j000_vecbuf, _cg_beta_j100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_j100_reg = _cg_vectmp6;
          bElem _cg_vectmp7;
          // merge0 _cg_beta_j_100_vecbuf ,_cg_beta_j000_vecbuf, 7 -> _cg_vectmp7
          dev_shl(_cg_vectmp7, _cg_beta_j_100_vecbuf, _cg_beta_j000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_j_100_reg = _cg_vectmp7;
        }
        buf13[0] += (_cg_beta_j100_reg - _cg_beta_j_100_reg) * buf9[0];
      }
      {
        // New offset [0, -1, 1]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i01_2_reg;
        {
          _cg_beta_i01_2_vecbuf = beta_i.dat[neighbor4 * beta_i.step + 448 + hipThreadIdx_x];
          _cg_beta_i010_vecbuf = beta_i.dat[neighbor13 * beta_i.step + 64 + hipThreadIdx_x];
          _cg_beta_i010_reg = _cg_beta_i010_vecbuf;
          _cg_beta_i01_2_reg = _cg_beta_i01_2_vecbuf;
        }
        buf13[0] += (_cg_beta_i010_reg - _cg_beta_i01_2_reg) * buf4[0];
      }
      {
        // New offset [1, -1, 1]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i01_2_reg;
        {
          _cg_beta_i_11_2_vecbuf = _cg_beta_i01_2_vecbuf;
          _cg_beta_i_110_vecbuf = _cg_beta_i010_vecbuf;
          _cg_beta_i01_2_vecbuf = beta_i.dat[neighbor5 * beta_i.step + 448 + hipThreadIdx_x];
          _cg_beta_i010_vecbuf = beta_i.dat[neighbor14 * beta_i.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_beta_i_110_vecbuf ,_cg_beta_i010_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_beta_i_110_vecbuf, _cg_beta_i010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i010_reg = _cg_vectmp0;
          bElem _cg_vectmp1;
          // merge0 _cg_beta_i_11_2_vecbuf ,_cg_beta_i01_2_vecbuf, 1 -> _cg_vectmp1
          dev_shl(_cg_vectmp1, _cg_beta_i_11_2_vecbuf, _cg_beta_i01_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i01_2_reg = _cg_vectmp1;
        }
        buf13[0] += (_cg_beta_i010_reg - _cg_beta_i01_2_reg) * buf8[0];
      }
      {
        // New offset [-1, 0, 1]
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j10_2_reg;
        {
          _cg_beta_j00_2_vecbuf = beta_j.dat[neighbor4 * beta_j.step + 448 + hipThreadIdx_x];
          _cg_beta_j000_vecbuf = beta_j.dat[neighbor13 * beta_j.step + 64 + hipThreadIdx_x];
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_beta_j10_2_reg = _cg_beta_j00_2_vecbuf;
        }
        buf13[0] += (_cg_beta_j100_reg - _cg_beta_j10_2_reg) * buf2[0];
      }
      {
        // New offset [0, 0, 1]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i0_10_reg;
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j_100_reg;
        bElem _cg_beta_k100_reg;
        bElem _cg_beta_k_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_i.dat[neighbor13 * beta_i.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_i.dat[neighbor10 * beta_i.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_i0_10_vecbuf
          dev_shl(_cg_beta_i0_10_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_i.dat[neighbor16 * beta_i.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_i010_vecbuf
          dev_shl(_cg_beta_i010_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          _cg_beta_j_100_vecbuf = beta_j.dat[neighbor12 * beta_j.step + 64 + hipThreadIdx_x];
          _cg_beta_j100_vecbuf = beta_j.dat[neighbor14 * beta_j.step + 64 + hipThreadIdx_x];
          _cg_beta_k_100_vecbuf = beta_k.dat[neighbor12 * beta_k.step + 64 + hipThreadIdx_x];
          _cg_beta_k000_vecbuf = beta_k.dat[neighbor13 * beta_k.step + 64 + hipThreadIdx_x];
          _cg_beta_k100_vecbuf = beta_k.dat[neighbor14 * beta_k.step + 64 + hipThreadIdx_x];
          _cg_beta_i010_reg = _cg_beta_i010_vecbuf;
          _cg_beta_i0_10_reg = _cg_beta_i0_10_vecbuf;
          bElem _cg_vectmp3;
          // merge0 _cg_beta_j000_vecbuf ,_cg_beta_j100_vecbuf, 1 -> _cg_vectmp3
          dev_shl(_cg_vectmp3, _cg_beta_j000_vecbuf, _cg_beta_j100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_j100_reg = _cg_vectmp3;
          bElem _cg_vectmp4;
          // merge0 _cg_beta_j_100_vecbuf ,_cg_beta_j000_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_beta_j_100_vecbuf, _cg_beta_j000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_j_100_reg = _cg_vectmp4;
          bElem _cg_vectmp5;
          // merge0 _cg_beta_k000_vecbuf ,_cg_beta_k100_vecbuf, 1 -> _cg_vectmp5
          dev_shl(_cg_vectmp5, _cg_beta_k000_vecbuf, _cg_beta_k100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_k100_reg = _cg_vectmp5;
          bElem _cg_vectmp6;
          // merge0 _cg_beta_k_100_vecbuf ,_cg_beta_k000_vecbuf, 7 -> _cg_vectmp6
          dev_shl(_cg_vectmp6, _cg_beta_k_100_vecbuf, _cg_beta_k000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_k_100_reg = _cg_vectmp6;
        }
        buf13[1] += (_cg_beta_i010_reg - _cg_beta_i0_10_reg) * buf5[1];
        buf13[1] += (_cg_beta_j100_reg - _cg_beta_j_100_reg) * buf3[1];
        buf13[1] += (_cg_beta_k100_reg - _cg_beta_k_100_reg) * buf1[1];
        buf13[0] += (_cg_beta_k100_reg - _cg_beta_k_100_reg) * buf11[0];
      }
      {
        // New offset [1, 0, 1]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i0_10_reg;
        {
          _cg_beta_i_1_10_vecbuf = _cg_beta_i0_10_vecbuf;
          _cg_beta_i_110_vecbuf = _cg_beta_i010_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_i.dat[neighbor14 * beta_i.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_i.dat[neighbor11 * beta_i.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_i0_10_vecbuf
          dev_shl(_cg_beta_i0_10_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_i.dat[neighbor17 * beta_i.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_i010_vecbuf
          dev_shl(_cg_beta_i010_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp3;
          // merge0 _cg_beta_i_110_vecbuf ,_cg_beta_i010_vecbuf, 1 -> _cg_vectmp3
          dev_shl(_cg_vectmp3, _cg_beta_i_110_vecbuf, _cg_beta_i010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i010_reg = _cg_vectmp3;
          bElem _cg_vectmp4;
          // merge0 _cg_beta_i_1_10_vecbuf ,_cg_beta_i0_10_vecbuf, 1 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_beta_i_1_10_vecbuf, _cg_beta_i0_10_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i0_10_reg = _cg_vectmp4;
        }
        buf13[1] += (_cg_beta_i010_reg - _cg_beta_i0_10_reg) * buf7[1];
      }
      {
        // New offset [-1, 1, 1]
        bElem _cg_beta_k100_reg;
        bElem _cg_beta_k1_20_reg;
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j10_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_k.dat[neighbor13 * beta_k.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_k.dat[neighbor10 * beta_k.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_k0_20_vecbuf
          dev_shl(_cg_beta_k0_20_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_k.dat[neighbor16 * beta_k.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_k000_vecbuf
          dev_shl(_cg_beta_k000_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp3;
          _cg_vectmp3 = beta_j.dat[neighbor7 * beta_j.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp4;
          _cg_vectmp4 = beta_j.dat[neighbor4 * beta_j.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp4 ,_cg_vectmp3, 1 -> _cg_beta_j00_2_vecbuf
          dev_shl(_cg_beta_j00_2_vecbuf, _cg_vectmp4, _cg_vectmp3, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp5;
          _cg_vectmp5 = beta_j.dat[neighbor16 * beta_j.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp6;
          _cg_vectmp6 = beta_j.dat[neighbor13 * beta_j.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp6 ,_cg_vectmp5, 1 -> _cg_beta_j000_vecbuf
          dev_shl(_cg_beta_j000_vecbuf, _cg_vectmp6, _cg_vectmp5, 56, 64, hipThreadIdx_x);
          _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
          _cg_beta_k1_20_reg = _cg_beta_k0_20_vecbuf;
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_beta_j10_2_reg = _cg_beta_j00_2_vecbuf;
        }
        buf13[1] += (_cg_beta_k100_reg - _cg_beta_k1_20_reg) * buf0[1];
        buf13[0] += (_cg_beta_j100_reg - _cg_beta_j10_2_reg) * buf10[0];
        buf13[0] += (_cg_beta_k100_reg - _cg_beta_k1_20_reg) * buf12[0];
      }
      {
        // New offset [0, 1, 1]
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j_100_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_j.dat[neighbor15 * beta_j.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_j.dat[neighbor12 * beta_j.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j_100_vecbuf
          dev_shl(_cg_beta_j_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_j.dat[neighbor17 * beta_j.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = beta_j.dat[neighbor14 * beta_j.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_beta_j100_vecbuf
          dev_shl(_cg_beta_j100_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_beta_j000_vecbuf ,_cg_beta_j100_vecbuf, 1 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_beta_j000_vecbuf, _cg_beta_j100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_j100_reg = _cg_vectmp4;
          bElem _cg_vectmp5;
          // merge0 _cg_beta_j_100_vecbuf ,_cg_beta_j000_vecbuf, 7 -> _cg_vectmp5
          dev_shl(_cg_vectmp5, _cg_beta_j_100_vecbuf, _cg_beta_j000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_j_100_reg = _cg_vectmp5;
        }
        buf13[1] += (_cg_beta_j100_reg - _cg_beta_j_100_reg) * buf9[1];
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 6; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, -1, 2]
            bElem _cg_beta_i010_reg;
            bElem _cg_beta_i01_2_reg;
            {
              _cg_beta_i01_2_vecbuf = beta_i.dat[neighbor13 * beta_i.step + (hipThreadIdx_x + rel * 64)];
              _cg_beta_i010_vecbuf = beta_i.dat[neighbor13 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_i010_reg = _cg_beta_i010_vecbuf;
              _cg_beta_i01_2_reg = _cg_beta_i01_2_vecbuf;
            }
            buf13[1 + rel] += (_cg_beta_i010_reg - _cg_beta_i01_2_reg) * buf4[1 + rel];
          }
          {
            // New offset [1, -1, 2]
            bElem _cg_beta_i010_reg;
            bElem _cg_beta_i01_2_reg;
            {
              _cg_beta_i_11_2_vecbuf = _cg_beta_i01_2_vecbuf;
              _cg_beta_i_110_vecbuf = _cg_beta_i010_vecbuf;
              _cg_beta_i01_2_vecbuf = beta_i.dat[neighbor14 * beta_i.step + (hipThreadIdx_x + rel * 64)];
              _cg_beta_i010_vecbuf = beta_i.dat[neighbor14 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_beta_i_110_vecbuf ,_cg_beta_i010_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_beta_i_110_vecbuf, _cg_beta_i010_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_i010_reg = _cg_vectmp0;
              bElem _cg_vectmp1;
              // merge0 _cg_beta_i_11_2_vecbuf ,_cg_beta_i01_2_vecbuf, 1 -> _cg_vectmp1
              dev_shl(_cg_vectmp1, _cg_beta_i_11_2_vecbuf, _cg_beta_i01_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_i01_2_reg = _cg_vectmp1;
            }
            buf13[1 + rel] += (_cg_beta_i010_reg - _cg_beta_i01_2_reg) * buf8[1 + rel];
          }
          {
            // New offset [-1, 0, 2]
            bElem _cg_beta_j100_reg;
            bElem _cg_beta_j10_2_reg;
            {
              _cg_beta_j00_2_vecbuf = beta_j.dat[neighbor13 * beta_j.step + (hipThreadIdx_x + rel * 64)];
              _cg_beta_j000_vecbuf = beta_j.dat[neighbor13 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
              _cg_beta_j10_2_reg = _cg_beta_j00_2_vecbuf;
            }
            buf13[1 + rel] += (_cg_beta_j100_reg - _cg_beta_j10_2_reg) * buf2[1 + rel];
          }
          {
            // New offset [0, 0, 2]
            bElem _cg_beta_i010_reg;
            bElem _cg_beta_i0_10_reg;
            bElem _cg_beta_j100_reg;
            bElem _cg_beta_j_100_reg;
            bElem _cg_beta_k100_reg;
            bElem _cg_beta_k_100_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = beta_i.dat[neighbor13 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = beta_i.dat[neighbor10 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_i0_10_vecbuf
              dev_shl(_cg_beta_i0_10_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = beta_i.dat[neighbor16 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_i010_vecbuf
              dev_shl(_cg_beta_i010_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
              _cg_beta_j_100_vecbuf = beta_j.dat[neighbor12 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_j100_vecbuf = beta_j.dat[neighbor14 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_k_100_vecbuf = beta_k.dat[neighbor12 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_k000_vecbuf = beta_k.dat[neighbor13 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_k100_vecbuf = beta_k.dat[neighbor14 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_i010_reg = _cg_beta_i010_vecbuf;
              _cg_beta_i0_10_reg = _cg_beta_i0_10_vecbuf;
              bElem _cg_vectmp3;
              // merge0 _cg_beta_j000_vecbuf ,_cg_beta_j100_vecbuf, 1 -> _cg_vectmp3
              dev_shl(_cg_vectmp3, _cg_beta_j000_vecbuf, _cg_beta_j100_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_j100_reg = _cg_vectmp3;
              bElem _cg_vectmp4;
              // merge0 _cg_beta_j_100_vecbuf ,_cg_beta_j000_vecbuf, 7 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_beta_j_100_vecbuf, _cg_beta_j000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_beta_j_100_reg = _cg_vectmp4;
              bElem _cg_vectmp5;
              // merge0 _cg_beta_k000_vecbuf ,_cg_beta_k100_vecbuf, 1 -> _cg_vectmp5
              dev_shl(_cg_vectmp5, _cg_beta_k000_vecbuf, _cg_beta_k100_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_k100_reg = _cg_vectmp5;
              bElem _cg_vectmp6;
              // merge0 _cg_beta_k_100_vecbuf ,_cg_beta_k000_vecbuf, 7 -> _cg_vectmp6
              dev_shl(_cg_vectmp6, _cg_beta_k_100_vecbuf, _cg_beta_k000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_beta_k_100_reg = _cg_vectmp6;
            }
            buf13[2 + rel] += (_cg_beta_i010_reg - _cg_beta_i0_10_reg) * buf5[2 + rel];
            buf13[2 + rel] += (_cg_beta_j100_reg - _cg_beta_j_100_reg) * buf3[2 + rel];
            buf13[2 + rel] += (_cg_beta_k100_reg - _cg_beta_k_100_reg) * buf1[2 + rel];
            buf13[1 + rel] += (_cg_beta_k100_reg - _cg_beta_k_100_reg) * buf11[1 + rel];
          }
          {
            // New offset [1, 0, 2]
            bElem _cg_beta_i010_reg;
            bElem _cg_beta_i0_10_reg;
            {
              _cg_beta_i_1_10_vecbuf = _cg_beta_i0_10_vecbuf;
              _cg_beta_i_110_vecbuf = _cg_beta_i010_vecbuf;
              bElem _cg_vectmp0;
              _cg_vectmp0 = beta_i.dat[neighbor14 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = beta_i.dat[neighbor11 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_i0_10_vecbuf
              dev_shl(_cg_beta_i0_10_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = beta_i.dat[neighbor17 * beta_i.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_i010_vecbuf
              dev_shl(_cg_beta_i010_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp3;
              // merge0 _cg_beta_i_110_vecbuf ,_cg_beta_i010_vecbuf, 1 -> _cg_vectmp3
              dev_shl(_cg_vectmp3, _cg_beta_i_110_vecbuf, _cg_beta_i010_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_i010_reg = _cg_vectmp3;
              bElem _cg_vectmp4;
              // merge0 _cg_beta_i_1_10_vecbuf ,_cg_beta_i0_10_vecbuf, 1 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_beta_i_1_10_vecbuf, _cg_beta_i0_10_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_i0_10_reg = _cg_vectmp4;
            }
            buf13[2 + rel] += (_cg_beta_i010_reg - _cg_beta_i0_10_reg) * buf7[2 + rel];
          }
          {
            // New offset [-1, 1, 2]
            bElem _cg_beta_k100_reg;
            bElem _cg_beta_k1_20_reg;
            bElem _cg_beta_j100_reg;
            bElem _cg_beta_j10_2_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = beta_k.dat[neighbor13 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = beta_k.dat[neighbor10 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_beta_k0_20_vecbuf
              dev_shl(_cg_beta_k0_20_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = beta_k.dat[neighbor16 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp0 ,_cg_vectmp2, 1 -> _cg_beta_k000_vecbuf
              dev_shl(_cg_beta_k000_vecbuf, _cg_vectmp0, _cg_vectmp2, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp3;
              _cg_vectmp3 = beta_j.dat[neighbor16 * beta_j.step + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp4;
              _cg_vectmp4 = beta_j.dat[neighbor13 * beta_j.step + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp4 ,_cg_vectmp3, 1 -> _cg_beta_j00_2_vecbuf
              dev_shl(_cg_beta_j00_2_vecbuf, _cg_vectmp4, _cg_vectmp3, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp5;
              _cg_vectmp5 = beta_j.dat[neighbor16 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp6;
              _cg_vectmp6 = beta_j.dat[neighbor13 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp6 ,_cg_vectmp5, 1 -> _cg_beta_j000_vecbuf
              dev_shl(_cg_beta_j000_vecbuf, _cg_vectmp6, _cg_vectmp5, 56, 64, hipThreadIdx_x);
              _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
              _cg_beta_k1_20_reg = _cg_beta_k0_20_vecbuf;
              _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
              _cg_beta_j10_2_reg = _cg_beta_j00_2_vecbuf;
            }
            buf13[2 + rel] += (_cg_beta_k100_reg - _cg_beta_k1_20_reg) * buf0[2 + rel];
            buf13[1 + rel] += (_cg_beta_j100_reg - _cg_beta_j10_2_reg) * buf10[1 + rel];
            buf13[1 + rel] += (_cg_beta_k100_reg - _cg_beta_k1_20_reg) * buf12[1 + rel];
          }
          {
            // New offset [0, 1, 2]
            bElem _cg_beta_j100_reg;
            bElem _cg_beta_j_100_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = beta_j.dat[neighbor15 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = beta_j.dat[neighbor12 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j_100_vecbuf
              dev_shl(_cg_beta_j_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = beta_j.dat[neighbor17 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = beta_j.dat[neighbor14 * beta_j.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_beta_j100_vecbuf
              dev_shl(_cg_beta_j100_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_beta_j000_vecbuf ,_cg_beta_j100_vecbuf, 1 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_beta_j000_vecbuf, _cg_beta_j100_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_j100_reg = _cg_vectmp4;
              bElem _cg_vectmp5;
              // merge0 _cg_beta_j_100_vecbuf ,_cg_beta_j000_vecbuf, 7 -> _cg_vectmp5
              dev_shl(_cg_vectmp5, _cg_beta_j_100_vecbuf, _cg_beta_j000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_beta_j_100_reg = _cg_vectmp5;
            }
            buf13[2 + rel] += (_cg_beta_j100_reg - _cg_beta_j_100_reg) * buf9[2 + rel];
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [0, -1, 8]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i01_2_reg;
        {
          _cg_beta_i01_2_vecbuf = beta_i.dat[neighbor13 * beta_i.step + 384 + hipThreadIdx_x];
          _cg_beta_i010_vecbuf = beta_i.dat[neighbor22 * beta_i.step + hipThreadIdx_x];
          _cg_beta_i010_reg = _cg_beta_i010_vecbuf;
          _cg_beta_i01_2_reg = _cg_beta_i01_2_vecbuf;
        }
        buf13[7] += (_cg_beta_i010_reg - _cg_beta_i01_2_reg) * buf4[7];
      }
      {
        // New offset [1, -1, 8]
        bElem _cg_beta_i010_reg;
        bElem _cg_beta_i01_2_reg;
        {
          _cg_beta_i_11_2_vecbuf = _cg_beta_i01_2_vecbuf;
          _cg_beta_i_110_vecbuf = _cg_beta_i010_vecbuf;
          _cg_beta_i01_2_vecbuf = beta_i.dat[neighbor14 * beta_i.step + 384 + hipThreadIdx_x];
          _cg_beta_i010_vecbuf = beta_i.dat[neighbor23 * beta_i.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_beta_i_110_vecbuf ,_cg_beta_i010_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_beta_i_110_vecbuf, _cg_beta_i010_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i010_reg = _cg_vectmp0;
          bElem _cg_vectmp1;
          // merge0 _cg_beta_i_11_2_vecbuf ,_cg_beta_i01_2_vecbuf, 1 -> _cg_vectmp1
          dev_shl(_cg_vectmp1, _cg_beta_i_11_2_vecbuf, _cg_beta_i01_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i01_2_reg = _cg_vectmp1;
        }
        buf13[7] += (_cg_beta_i010_reg - _cg_beta_i01_2_reg) * buf8[7];
      }
      {
        // New offset [-1, 0, 8]
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j10_2_reg;
        {
          _cg_beta_j00_2_vecbuf = beta_j.dat[neighbor13 * beta_j.step + 384 + hipThreadIdx_x];
          _cg_beta_j000_vecbuf = beta_j.dat[neighbor22 * beta_j.step + hipThreadIdx_x];
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_beta_j10_2_reg = _cg_beta_j00_2_vecbuf;
        }
        buf13[7] += (_cg_beta_j100_reg - _cg_beta_j10_2_reg) * buf2[7];
      }
      {
        // New offset [0, 0, 8]
        bElem _cg_beta_k100_reg;
        bElem _cg_beta_k_100_reg;
        {
          _cg_beta_k_100_vecbuf = beta_k.dat[neighbor21 * beta_k.step + hipThreadIdx_x];
          _cg_beta_k000_vecbuf = beta_k.dat[neighbor22 * beta_k.step + hipThreadIdx_x];
          _cg_beta_k100_vecbuf = beta_k.dat[neighbor23 * beta_k.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_beta_k000_vecbuf ,_cg_beta_k100_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_beta_k000_vecbuf, _cg_beta_k100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_k100_reg = _cg_vectmp0;
          bElem _cg_vectmp1;
          // merge0 _cg_beta_k_100_vecbuf ,_cg_beta_k000_vecbuf, 7 -> _cg_vectmp1
          dev_shl(_cg_vectmp1, _cg_beta_k_100_vecbuf, _cg_beta_k000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_beta_k_100_reg = _cg_vectmp1;
        }
        buf13[7] += (_cg_beta_k100_reg - _cg_beta_k_100_reg) * buf11[7];
      }
      {
        // New offset [-1, 1, 8]
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_j10_2_reg;
        bElem _cg_beta_k100_reg;
        bElem _cg_beta_k1_20_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_j.dat[neighbor16 * beta_j.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_j.dat[neighbor13 * beta_j.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j00_2_vecbuf
          dev_shl(_cg_beta_j00_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = beta_j.dat[neighbor25 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = beta_j.dat[neighbor22 * beta_j.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_beta_j000_vecbuf
          dev_shl(_cg_beta_j000_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          _cg_vectmp4 = beta_k.dat[neighbor22 * beta_k.step + hipThreadIdx_x];
          bElem _cg_vectmp5;
          _cg_vectmp5 = beta_k.dat[neighbor19 * beta_k.step + hipThreadIdx_x];
          // merge1 _cg_vectmp5 ,_cg_vectmp4, 7 -> _cg_beta_k0_20_vecbuf
          dev_shl(_cg_beta_k0_20_vecbuf, _cg_vectmp5, _cg_vectmp4, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp6;
          _cg_vectmp6 = beta_k.dat[neighbor25 * beta_k.step + hipThreadIdx_x];
          // merge1 _cg_vectmp4 ,_cg_vectmp6, 1 -> _cg_beta_k000_vecbuf
          dev_shl(_cg_beta_k000_vecbuf, _cg_vectmp4, _cg_vectmp6, 56, 64, hipThreadIdx_x);
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_beta_j10_2_reg = _cg_beta_j00_2_vecbuf;
          _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
          _cg_beta_k1_20_reg = _cg_beta_k0_20_vecbuf;
        }
        buf13[7] += (_cg_beta_j100_reg - _cg_beta_j10_2_reg) * buf10[7];
        buf13[7] += (_cg_beta_k100_reg - _cg_beta_k1_20_reg) * buf12[7];
      }
    }
  }
  {
    {
      bElem _cg_alpha000_vecbuf;
      bElem _cg_x000_vecbuf;
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 8; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 0]
            bElem _cg_alpha000_reg;
            bElem _cg_x000_reg;
            {
              _cg_alpha000_vecbuf = alpha.dat[neighbor13 * alpha.step + (hipThreadIdx_x + rel * 64)];
              _cg_x000_vecbuf = x.dat[neighbor13 * x.step + (hipThreadIdx_x + rel * 64)];
              _cg_alpha000_reg = _cg_alpha000_vecbuf;
              _cg_x000_reg = _cg_x000_vecbuf;
            }
            buf12[0 + rel] = c[0] * _cg_alpha000_reg * _cg_x000_reg - c[1] * c[2] * (0.0833 * buf6[0 + rel] + 0.020825 * buf13[0 + rel]);
          }
          _cg_rel2 += 1;
        }
      }
    }
    bElem *out_ref = &out.dat[neighbor13 * out.step];
    for (long sti = 0; sti < 8; ++sti)
    {
      out_ref[sti * 64 + hipThreadIdx_x] = buf12[sti];
    }
  }
}
# 456 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/helmholtz4/intermediate_gen/helmholtz4.cu" 2

}
