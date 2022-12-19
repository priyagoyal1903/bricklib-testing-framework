# 1 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/poisson/intermediate_gen/poisson.cu"
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
# 1 "VSTile-poisson2.py-HIP-8x8x64" 1
{
  bElem buf0[64];
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
        buf0[0] -= 0.0833 * in(i + hipThreadIdx_x, j + -1, k + -1);
      }
      {
        buf0[0] -= 0.166 * in(i + hipThreadIdx_x, j, k + -1);
        buf0[0] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + -1);
        buf0[0] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + -1);
        buf0[1] -= 0.0833 * in(i + hipThreadIdx_x, j, k + -1);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[1 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf0[1 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + -1);
            buf0[1 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + -1);
            buf0[2 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[7] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + -1);
        buf0[7] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + -1);
        buf0[7] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + -1);
        buf0[6] -= 0.0833 * in(i + hipThreadIdx_x, j + 7, k + -1);
      }
      {
        buf0[7] -= 0.0833 * in(i + hipThreadIdx_x, j + 8, k + -1);
      }
      {
        buf0[0] -= 0.166 * in(i + hipThreadIdx_x, j + -1, k);
        buf0[0] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + -1, k);
        buf0[0] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + -1, k);
        buf0[8] -= 0.0833 * in(i + hipThreadIdx_x, j + -1, k);
      }
      {
        buf0[0] += 2.666 * in(i + hipThreadIdx_x, j, k);
        buf0[0] -= 0.166 * in(i + hipThreadIdx_x + 1, j, k);
        buf0[0] -= 0.166 * in(i + hipThreadIdx_x + -1, j, k);
        buf0[8] -= 0.166 * in(i + hipThreadIdx_x, j, k);
        buf0[8] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k);
        buf0[8] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k);
        buf0[1] -= 0.166 * in(i + hipThreadIdx_x, j, k);
        buf0[1] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k);
        buf0[1] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k);
        buf0[9] -= 0.0833 * in(i + hipThreadIdx_x, j, k);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[1 + rel] += 2.666 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf0[1 + rel] -= 0.166 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf0[1 + rel] -= 0.166 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
            buf0[9 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf0[9 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
            buf0[9 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf0[2 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf0[2 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
            buf0[2 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf0[0 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
            buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf0[10 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf0[8 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[7] += 2.666 * in(i + hipThreadIdx_x, j + 7, k);
        buf0[7] -= 0.166 * in(i + hipThreadIdx_x + 1, j + 7, k);
        buf0[7] -= 0.166 * in(i + hipThreadIdx_x + -1, j + 7, k);
        buf0[15] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k);
        buf0[15] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k);
        buf0[15] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k);
        buf0[6] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k);
        buf0[6] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k);
        buf0[6] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k);
        buf0[14] -= 0.0833 * in(i + hipThreadIdx_x, j + 7, k);
      }
      {
        buf0[7] -= 0.166 * in(i + hipThreadIdx_x, j + 8, k);
        buf0[7] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 8, k);
        buf0[7] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 8, k);
        buf0[15] -= 0.0833 * in(i + hipThreadIdx_x, j + 8, k);
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 6; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            buf0[8 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
            buf0[8 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 1);
            buf0[8 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 1);
            buf0[16 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
            buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
          }
          {
            buf0[8 + rel] += 2.666 * in(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf0[8 + rel] -= 0.166 * in(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1);
            buf0[8 + rel] -= 0.166 * in(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1);
            buf0[16 + rel] -= 0.166 * in(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf0[16 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1);
            buf0[16 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1);
            buf0[0 + rel] -= 0.166 * in(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1);
            buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1);
            buf0[9 + rel] -= 0.166 * in(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf0[9 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1);
            buf0[9 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1);
            buf0[17 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf0[1 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
          }
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf0[9 + rel] += 2.666 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[9 + rel] -= 0.166 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[9 + rel] -= 0.166 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[17 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[17 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[17 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[1 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[1 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[1 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[10 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[10 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[10 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[8 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[8 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[8 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[18 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[2 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[16 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
              }
              _cg_rel1 += 1;
            }
          }
          {
            buf0[15 + rel] += 2.666 * in(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf0[15 + rel] -= 0.166 * in(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1);
            buf0[15 + rel] -= 0.166 * in(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 1);
            buf0[23 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf0[23 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 1);
            buf0[23 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1);
            buf0[7 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf0[7 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 1);
            buf0[7 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1);
            buf0[14 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf0[14 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 1);
            buf0[14 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1);
            buf0[22 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf0[6 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
          }
          {
            buf0[15 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf0[15 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 1);
            buf0[15 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 1);
            buf0[23 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf0[7 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
          }
          _cg_rel2 += 8;
        }
      }
      {
        buf0[56] -= 0.166 * in(i + hipThreadIdx_x, j + -1, k + 7);
        buf0[56] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + -1, k + 7);
        buf0[56] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + -1, k + 7);
        buf0[48] -= 0.0833 * in(i + hipThreadIdx_x, j + -1, k + 7);
      }
      {
        buf0[56] += 2.666 * in(i + hipThreadIdx_x, j, k + 7);
        buf0[56] -= 0.166 * in(i + hipThreadIdx_x + 1, j, k + 7);
        buf0[56] -= 0.166 * in(i + hipThreadIdx_x + -1, j, k + 7);
        buf0[48] -= 0.166 * in(i + hipThreadIdx_x, j, k + 7);
        buf0[48] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + 7);
        buf0[48] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + 7);
        buf0[57] -= 0.166 * in(i + hipThreadIdx_x, j, k + 7);
        buf0[57] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + 7);
        buf0[57] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + 7);
        buf0[49] -= 0.0833 * in(i + hipThreadIdx_x, j, k + 7);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[57 + rel] += 2.666 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf0[57 + rel] -= 0.166 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf0[57 + rel] -= 0.166 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf0[49 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf0[49 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf0[49 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf0[58 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf0[58 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf0[58 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf0[56 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf0[56 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf0[56 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf0[50 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf0[48 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[63] += 2.666 * in(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[63] -= 0.166 * in(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf0[63] -= 0.166 * in(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf0[55] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[55] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf0[55] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf0[62] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[62] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf0[62] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf0[54] -= 0.0833 * in(i + hipThreadIdx_x, j + 7, k + 7);
      }
      {
        buf0[63] -= 0.166 * in(i + hipThreadIdx_x, j + 8, k + 7);
        buf0[63] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 8, k + 7);
        buf0[63] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 8, k + 7);
        buf0[55] -= 0.0833 * in(i + hipThreadIdx_x, j + 8, k + 7);
      }
      {
        buf0[56] -= 0.0833 * in(i + hipThreadIdx_x, j + -1, k + 8);
      }
      {
        buf0[56] -= 0.166 * in(i + hipThreadIdx_x, j, k + 8);
        buf0[56] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + 8);
        buf0[56] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + 8);
        buf0[57] -= 0.0833 * in(i + hipThreadIdx_x, j, k + 8);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[57 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
            buf0[57 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 8);
            buf0[57 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 8);
            buf0[58 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
            buf0[56 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[63] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + 8);
        buf0[63] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + 8);
        buf0[63] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + 8);
        buf0[62] -= 0.0833 * in(i + hipThreadIdx_x, j + 7, k + 8);
      }
      {
        buf0[63] -= 0.0833 * in(i + hipThreadIdx_x, j + 8, k + 8);
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
            out(i + _cg_idx0, j + _cg_idx1, k + _cg_idx2) = buf0[rel];
          }
        }
      }
    }
  }
}
# 29 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/poisson/intermediate_gen/poisson.cu" 2

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
# 1 "VSBrick-poisson2.py-HIP-8x8x8-8x8" 1
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
          }
          _cg_rel2 += 1;
        }
      }
    }
    {
      bElem _cg_in000_vecbuf;
      bElem _cg_in_100_vecbuf;
      {
        // New offset [0, -1, -1]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor4 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor1 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, 0, -1]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = in.dat[neighbor3 * in.step + 448 + hipThreadIdx_x];
          _cg_in000_vecbuf = in.dat[neighbor4 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 0, -1]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[0] -= 0.166 * _cg_in000_reg;
      }
      {
        // New offset [1, 0, -1]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          _cg_in000_vecbuf = in.dat[neighbor5 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 1, -1]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor7 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor4 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, -1, 0]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor12 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor9 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in_100_vecbuf
          dev_shl(_cg_in_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = in.dat[neighbor13 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = in.dat[neighbor10 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp4;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, -1, 0]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[0] -= 0.166 * _cg_in000_reg;
        buf0[1] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [1, -1, 0]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor14 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor11 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp2;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, 0, 0]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = in.dat[neighbor12 * in.step + hipThreadIdx_x];
          _cg_in000_vecbuf = in.dat[neighbor13 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[0] -= 0.166 * _cg_in000_reg;
        buf0[1] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 0, 0]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[0] += 2.666 * _cg_in000_reg;
        buf0[1] -= 0.166 * _cg_in000_reg;
      }
      {
        // New offset [1, 0, 0]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          _cg_in000_vecbuf = in.dat[neighbor14 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[0] -= 0.166 * _cg_in000_reg;
        buf0[1] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, 1, 0]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor15 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor12 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in_100_vecbuf
          dev_shl(_cg_in_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = in.dat[neighbor16 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = in.dat[neighbor13 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp4;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 1, 0]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[0] -= 0.166 * _cg_in000_reg;
        buf0[1] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [1, 1, 0]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor17 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor14 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp2;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 6; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [-1, -1, 1]
            bElem _cg_in000_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = in.dat[neighbor12 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = in.dat[neighbor9 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in_100_vecbuf
              dev_shl(_cg_in_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = in.dat[neighbor13 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = in.dat[neighbor10 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_in000_vecbuf
              dev_shl(_cg_in000_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_in000_reg = _cg_vectmp4;
            }
            buf0[1 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [0, -1, 1]
            bElem _cg_in000_reg;
            {
              _cg_in000_reg = _cg_in000_vecbuf;
            }
            buf0[1 + rel] -= 0.166 * _cg_in000_reg;
            buf0[2 + rel] -= 0.0833 * _cg_in000_reg;
            buf0[0 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [1, -1, 1]
            bElem _cg_in000_reg;
            {
              _cg_in_100_vecbuf = _cg_in000_vecbuf;
              bElem _cg_vectmp0;
              _cg_vectmp0 = in.dat[neighbor14 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = in.dat[neighbor11 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in000_vecbuf
              dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_in000_reg = _cg_vectmp2;
            }
            buf0[1 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [-1, 0, 1]
            bElem _cg_in000_reg;
            {
              _cg_in_100_vecbuf = in.dat[neighbor12 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_in000_vecbuf = in.dat[neighbor13 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_in000_reg = _cg_vectmp0;
            }
            buf0[1 + rel] -= 0.166 * _cg_in000_reg;
            buf0[2 + rel] -= 0.0833 * _cg_in000_reg;
            buf0[0 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [0, 0, 1]
            bElem _cg_in000_reg;
            {
              _cg_in000_reg = _cg_in000_vecbuf;
            }
            buf0[1 + rel] += 2.666 * _cg_in000_reg;
            buf0[2 + rel] -= 0.166 * _cg_in000_reg;
            buf0[0 + rel] -= 0.166 * _cg_in000_reg;
          }
          {
            // New offset [1, 0, 1]
            bElem _cg_in000_reg;
            {
              _cg_in_100_vecbuf = _cg_in000_vecbuf;
              _cg_in000_vecbuf = in.dat[neighbor14 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_in000_reg = _cg_vectmp0;
            }
            buf0[1 + rel] -= 0.166 * _cg_in000_reg;
            buf0[2 + rel] -= 0.0833 * _cg_in000_reg;
            buf0[0 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [-1, 1, 1]
            bElem _cg_in000_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = in.dat[neighbor15 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = in.dat[neighbor12 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in_100_vecbuf
              dev_shl(_cg_in_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = in.dat[neighbor16 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = in.dat[neighbor13 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_in000_vecbuf
              dev_shl(_cg_in000_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_in000_reg = _cg_vectmp4;
            }
            buf0[1 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [0, 1, 1]
            bElem _cg_in000_reg;
            {
              _cg_in000_reg = _cg_in000_vecbuf;
            }
            buf0[1 + rel] -= 0.166 * _cg_in000_reg;
            buf0[2 + rel] -= 0.0833 * _cg_in000_reg;
            buf0[0 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [1, 1, 1]
            bElem _cg_in000_reg;
            {
              _cg_in_100_vecbuf = _cg_in000_vecbuf;
              bElem _cg_vectmp0;
              _cg_vectmp0 = in.dat[neighbor17 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = in.dat[neighbor14 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in000_vecbuf
              dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_in000_reg = _cg_vectmp2;
            }
            buf0[1 + rel] -= 0.0833 * _cg_in000_reg;
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [-1, -1, 7]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor12 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor9 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in_100_vecbuf
          dev_shl(_cg_in_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = in.dat[neighbor13 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = in.dat[neighbor10 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp4;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, -1, 7]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[7] -= 0.166 * _cg_in000_reg;
        buf0[6] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [1, -1, 7]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor14 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor11 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp2;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, 0, 7]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = in.dat[neighbor12 * in.step + 448 + hipThreadIdx_x];
          _cg_in000_vecbuf = in.dat[neighbor13 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[7] -= 0.166 * _cg_in000_reg;
        buf0[6] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 0, 7]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[7] += 2.666 * _cg_in000_reg;
        buf0[6] -= 0.166 * _cg_in000_reg;
      }
      {
        // New offset [1, 0, 7]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          _cg_in000_vecbuf = in.dat[neighbor14 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[7] -= 0.166 * _cg_in000_reg;
        buf0[6] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, 1, 7]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor15 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor12 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in_100_vecbuf
          dev_shl(_cg_in_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = in.dat[neighbor16 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = in.dat[neighbor13 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp4;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 1, 7]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[7] -= 0.166 * _cg_in000_reg;
        buf0[6] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [1, 1, 7]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor17 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor14 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp2;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, -1, 8]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor22 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor19 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, 0, 8]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = in.dat[neighbor21 * in.step + hipThreadIdx_x];
          _cg_in000_vecbuf = in.dat[neighbor22 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 0, 8]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[7] -= 0.166 * _cg_in000_reg;
      }
      {
        // New offset [1, 0, 8]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          _cg_in000_vecbuf = in.dat[neighbor23 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 1, 8]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor25 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor22 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
    }
    bElem *out_ref = &out.dat[neighbor13 * out.step];
    for (long sti = 0; sti < 8; ++sti)
    {
      out_ref[sti * 64 + hipThreadIdx_x] = buf0[sti];
    }
  }
}
# 51 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/poisson/intermediate_gen/poisson.cu" 2

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
# 1 "VSTile-poisson3.py-HIP-8x8x64" 1
{
  bElem buf0[64];
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
        buf0[0] -= 0.0833 * in(i + hipThreadIdx_x, j + -1, k + -1);
      }
      {
        buf0[0] -= 0.166 * in(i + hipThreadIdx_x, j, k + -1);
        buf0[0] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + -1);
        buf0[0] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + -1);
        buf0[1] -= 0.0833 * in(i + hipThreadIdx_x, j, k + -1);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[1 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf0[1 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + -1);
            buf0[1 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + -1);
            buf0[2 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[7] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + -1);
        buf0[7] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + -1);
        buf0[7] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + -1);
        buf0[6] -= 0.0833 * in(i + hipThreadIdx_x, j + 7, k + -1);
      }
      {
        buf0[7] -= 0.0833 * in(i + hipThreadIdx_x, j + 8, k + -1);
      }
      {
        buf0[0] -= 0.166 * in(i + hipThreadIdx_x, j + -1, k);
        buf0[0] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + -1, k);
        buf0[0] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + -1, k);
        buf0[8] -= 0.0833 * in(i + hipThreadIdx_x, j + -1, k);
      }
      {
        buf0[0] += 2.666 * in(i + hipThreadIdx_x, j, k);
        buf0[0] -= 0.166 * in(i + hipThreadIdx_x + 1, j, k);
        buf0[0] -= 0.166 * in(i + hipThreadIdx_x + -1, j, k);
        buf0[8] -= 0.166 * in(i + hipThreadIdx_x, j, k);
        buf0[8] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k);
        buf0[8] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k);
        buf0[1] -= 0.166 * in(i + hipThreadIdx_x, j, k);
        buf0[1] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k);
        buf0[1] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k);
        buf0[9] -= 0.0833 * in(i + hipThreadIdx_x, j, k);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[1 + rel] += 2.666 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf0[1 + rel] -= 0.166 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf0[1 + rel] -= 0.166 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
            buf0[9 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf0[9 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
            buf0[9 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf0[2 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf0[2 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
            buf0[2 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf0[0 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
            buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf0[10 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf0[8 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[7] += 2.666 * in(i + hipThreadIdx_x, j + 7, k);
        buf0[7] -= 0.166 * in(i + hipThreadIdx_x + 1, j + 7, k);
        buf0[7] -= 0.166 * in(i + hipThreadIdx_x + -1, j + 7, k);
        buf0[15] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k);
        buf0[15] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k);
        buf0[15] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k);
        buf0[6] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k);
        buf0[6] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k);
        buf0[6] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k);
        buf0[14] -= 0.0833 * in(i + hipThreadIdx_x, j + 7, k);
      }
      {
        buf0[7] -= 0.166 * in(i + hipThreadIdx_x, j + 8, k);
        buf0[7] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 8, k);
        buf0[7] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 8, k);
        buf0[15] -= 0.0833 * in(i + hipThreadIdx_x, j + 8, k);
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 6; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            buf0[8 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
            buf0[8 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 1);
            buf0[8 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 1);
            buf0[16 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
            buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
          }
          {
            buf0[8 + rel] += 2.666 * in(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf0[8 + rel] -= 0.166 * in(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1);
            buf0[8 + rel] -= 0.166 * in(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1);
            buf0[16 + rel] -= 0.166 * in(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf0[16 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1);
            buf0[16 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1);
            buf0[0 + rel] -= 0.166 * in(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1);
            buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1);
            buf0[9 + rel] -= 0.166 * in(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf0[9 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1);
            buf0[9 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1);
            buf0[17 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf0[1 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
          }
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf0[9 + rel] += 2.666 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[9 + rel] -= 0.166 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[9 + rel] -= 0.166 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[17 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[17 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[17 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[1 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[1 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[1 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[10 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[10 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[10 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[8 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[8 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[8 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[18 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[2 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[16 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
              }
              _cg_rel1 += 1;
            }
          }
          {
            buf0[15 + rel] += 2.666 * in(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf0[15 + rel] -= 0.166 * in(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1);
            buf0[15 + rel] -= 0.166 * in(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 1);
            buf0[23 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf0[23 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 1);
            buf0[23 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1);
            buf0[7 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf0[7 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 1);
            buf0[7 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1);
            buf0[14 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf0[14 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 1);
            buf0[14 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1);
            buf0[22 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf0[6 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
          }
          {
            buf0[15 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf0[15 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 1);
            buf0[15 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 1);
            buf0[23 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf0[7 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
          }
          _cg_rel2 += 8;
        }
      }
      {
        buf0[56] -= 0.166 * in(i + hipThreadIdx_x, j + -1, k + 7);
        buf0[56] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + -1, k + 7);
        buf0[56] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + -1, k + 7);
        buf0[48] -= 0.0833 * in(i + hipThreadIdx_x, j + -1, k + 7);
      }
      {
        buf0[56] += 2.666 * in(i + hipThreadIdx_x, j, k + 7);
        buf0[56] -= 0.166 * in(i + hipThreadIdx_x + 1, j, k + 7);
        buf0[56] -= 0.166 * in(i + hipThreadIdx_x + -1, j, k + 7);
        buf0[48] -= 0.166 * in(i + hipThreadIdx_x, j, k + 7);
        buf0[48] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + 7);
        buf0[48] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + 7);
        buf0[57] -= 0.166 * in(i + hipThreadIdx_x, j, k + 7);
        buf0[57] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + 7);
        buf0[57] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + 7);
        buf0[49] -= 0.0833 * in(i + hipThreadIdx_x, j, k + 7);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[57 + rel] += 2.666 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf0[57 + rel] -= 0.166 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf0[57 + rel] -= 0.166 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf0[49 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf0[49 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf0[49 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf0[58 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf0[58 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf0[58 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf0[56 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf0[56 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf0[56 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf0[50 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf0[48 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[63] += 2.666 * in(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[63] -= 0.166 * in(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf0[63] -= 0.166 * in(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf0[55] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[55] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf0[55] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf0[62] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[62] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf0[62] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf0[54] -= 0.0833 * in(i + hipThreadIdx_x, j + 7, k + 7);
      }
      {
        buf0[63] -= 0.166 * in(i + hipThreadIdx_x, j + 8, k + 7);
        buf0[63] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 8, k + 7);
        buf0[63] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 8, k + 7);
        buf0[55] -= 0.0833 * in(i + hipThreadIdx_x, j + 8, k + 7);
      }
      {
        buf0[56] -= 0.0833 * in(i + hipThreadIdx_x, j + -1, k + 8);
      }
      {
        buf0[56] -= 0.166 * in(i + hipThreadIdx_x, j, k + 8);
        buf0[56] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + 8);
        buf0[56] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + 8);
        buf0[57] -= 0.0833 * in(i + hipThreadIdx_x, j, k + 8);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[57 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
            buf0[57 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 8);
            buf0[57 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 8);
            buf0[58 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
            buf0[56 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[63] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + 8);
        buf0[63] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + 8);
        buf0[63] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + 8);
        buf0[62] -= 0.0833 * in(i + hipThreadIdx_x, j + 7, k + 8);
      }
      {
        buf0[63] -= 0.0833 * in(i + hipThreadIdx_x, j + 8, k + 8);
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
            out(i + _cg_idx0, j + _cg_idx1, k + _cg_idx2) = buf0[rel];
          }
        }
      }
    }
  }
}
# 74 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/poisson/intermediate_gen/poisson.cu" 2

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
# 1 "VSBrick-poisson3.py-HIP-8x8x8-8x8" 1
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
          }
          _cg_rel2 += 1;
        }
      }
    }
    {
      bElem _cg_in000_vecbuf;
      bElem _cg_in_100_vecbuf;
      {
        // New offset [0, -1, -1]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor4 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor1 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, 0, -1]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = in.dat[neighbor3 * in.step + 448 + hipThreadIdx_x];
          _cg_in000_vecbuf = in.dat[neighbor4 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 0, -1]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[0] -= 0.166 * _cg_in000_reg;
      }
      {
        // New offset [1, 0, -1]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          _cg_in000_vecbuf = in.dat[neighbor5 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 1, -1]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor7 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor4 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, -1, 0]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor12 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor9 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in_100_vecbuf
          dev_shl(_cg_in_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = in.dat[neighbor13 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = in.dat[neighbor10 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp4;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, -1, 0]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[0] -= 0.166 * _cg_in000_reg;
        buf0[1] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [1, -1, 0]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor14 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor11 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp2;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, 0, 0]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = in.dat[neighbor12 * in.step + hipThreadIdx_x];
          _cg_in000_vecbuf = in.dat[neighbor13 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[0] -= 0.166 * _cg_in000_reg;
        buf0[1] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 0, 0]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[0] += 2.666 * _cg_in000_reg;
        buf0[1] -= 0.166 * _cg_in000_reg;
      }
      {
        // New offset [1, 0, 0]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          _cg_in000_vecbuf = in.dat[neighbor14 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[0] -= 0.166 * _cg_in000_reg;
        buf0[1] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, 1, 0]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor15 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor12 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in_100_vecbuf
          dev_shl(_cg_in_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = in.dat[neighbor16 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = in.dat[neighbor13 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp4;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 1, 0]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[0] -= 0.166 * _cg_in000_reg;
        buf0[1] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [1, 1, 0]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor17 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor14 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp2;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 6; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [-1, -1, 1]
            bElem _cg_in000_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = in.dat[neighbor12 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = in.dat[neighbor9 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in_100_vecbuf
              dev_shl(_cg_in_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = in.dat[neighbor13 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = in.dat[neighbor10 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_in000_vecbuf
              dev_shl(_cg_in000_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_in000_reg = _cg_vectmp4;
            }
            buf0[1 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [0, -1, 1]
            bElem _cg_in000_reg;
            {
              _cg_in000_reg = _cg_in000_vecbuf;
            }
            buf0[1 + rel] -= 0.166 * _cg_in000_reg;
            buf0[2 + rel] -= 0.0833 * _cg_in000_reg;
            buf0[0 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [1, -1, 1]
            bElem _cg_in000_reg;
            {
              _cg_in_100_vecbuf = _cg_in000_vecbuf;
              bElem _cg_vectmp0;
              _cg_vectmp0 = in.dat[neighbor14 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = in.dat[neighbor11 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in000_vecbuf
              dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_in000_reg = _cg_vectmp2;
            }
            buf0[1 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [-1, 0, 1]
            bElem _cg_in000_reg;
            {
              _cg_in_100_vecbuf = in.dat[neighbor12 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_in000_vecbuf = in.dat[neighbor13 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_in000_reg = _cg_vectmp0;
            }
            buf0[1 + rel] -= 0.166 * _cg_in000_reg;
            buf0[2 + rel] -= 0.0833 * _cg_in000_reg;
            buf0[0 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [0, 0, 1]
            bElem _cg_in000_reg;
            {
              _cg_in000_reg = _cg_in000_vecbuf;
            }
            buf0[1 + rel] += 2.666 * _cg_in000_reg;
            buf0[2 + rel] -= 0.166 * _cg_in000_reg;
            buf0[0 + rel] -= 0.166 * _cg_in000_reg;
          }
          {
            // New offset [1, 0, 1]
            bElem _cg_in000_reg;
            {
              _cg_in_100_vecbuf = _cg_in000_vecbuf;
              _cg_in000_vecbuf = in.dat[neighbor14 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_in000_reg = _cg_vectmp0;
            }
            buf0[1 + rel] -= 0.166 * _cg_in000_reg;
            buf0[2 + rel] -= 0.0833 * _cg_in000_reg;
            buf0[0 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [-1, 1, 1]
            bElem _cg_in000_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = in.dat[neighbor15 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = in.dat[neighbor12 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in_100_vecbuf
              dev_shl(_cg_in_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = in.dat[neighbor16 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = in.dat[neighbor13 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_in000_vecbuf
              dev_shl(_cg_in000_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_in000_reg = _cg_vectmp4;
            }
            buf0[1 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [0, 1, 1]
            bElem _cg_in000_reg;
            {
              _cg_in000_reg = _cg_in000_vecbuf;
            }
            buf0[1 + rel] -= 0.166 * _cg_in000_reg;
            buf0[2 + rel] -= 0.0833 * _cg_in000_reg;
            buf0[0 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [1, 1, 1]
            bElem _cg_in000_reg;
            {
              _cg_in_100_vecbuf = _cg_in000_vecbuf;
              bElem _cg_vectmp0;
              _cg_vectmp0 = in.dat[neighbor17 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = in.dat[neighbor14 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in000_vecbuf
              dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_in000_reg = _cg_vectmp2;
            }
            buf0[1 + rel] -= 0.0833 * _cg_in000_reg;
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [-1, -1, 7]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor12 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor9 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in_100_vecbuf
          dev_shl(_cg_in_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = in.dat[neighbor13 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = in.dat[neighbor10 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp4;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, -1, 7]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[7] -= 0.166 * _cg_in000_reg;
        buf0[6] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [1, -1, 7]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor14 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor11 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp2;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, 0, 7]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = in.dat[neighbor12 * in.step + 448 + hipThreadIdx_x];
          _cg_in000_vecbuf = in.dat[neighbor13 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[7] -= 0.166 * _cg_in000_reg;
        buf0[6] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 0, 7]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[7] += 2.666 * _cg_in000_reg;
        buf0[6] -= 0.166 * _cg_in000_reg;
      }
      {
        // New offset [1, 0, 7]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          _cg_in000_vecbuf = in.dat[neighbor14 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[7] -= 0.166 * _cg_in000_reg;
        buf0[6] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, 1, 7]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor15 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor12 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in_100_vecbuf
          dev_shl(_cg_in_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = in.dat[neighbor16 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = in.dat[neighbor13 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp4;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 1, 7]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[7] -= 0.166 * _cg_in000_reg;
        buf0[6] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [1, 1, 7]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor17 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor14 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp2;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, -1, 8]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor22 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor19 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, 0, 8]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = in.dat[neighbor21 * in.step + hipThreadIdx_x];
          _cg_in000_vecbuf = in.dat[neighbor22 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 0, 8]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[7] -= 0.166 * _cg_in000_reg;
      }
      {
        // New offset [1, 0, 8]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          _cg_in000_vecbuf = in.dat[neighbor23 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 1, 8]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor25 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor22 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
    }
    bElem *out_ref = &out.dat[neighbor13 * out.step];
    for (long sti = 0; sti < 8; ++sti)
    {
      out_ref[sti * 64 + hipThreadIdx_x] = buf0[sti];
    }
  }
}
# 96 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/poisson/intermediate_gen/poisson.cu" 2

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
# 1 "VSTile-poisson5.py-HIP-8x8x64" 1
{
  bElem buf0[64];
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
        buf0[0] -= 0.0833 * in(i + hipThreadIdx_x, j + -1, k + -1);
      }
      {
        buf0[0] -= 0.166 * in(i + hipThreadIdx_x, j, k + -1);
        buf0[0] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + -1);
        buf0[0] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + -1);
        buf0[1] -= 0.0833 * in(i + hipThreadIdx_x, j, k + -1);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[1 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf0[1 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + -1);
            buf0[1 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + -1);
            buf0[2 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
            buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[7] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + -1);
        buf0[7] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + -1);
        buf0[7] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + -1);
        buf0[6] -= 0.0833 * in(i + hipThreadIdx_x, j + 7, k + -1);
      }
      {
        buf0[7] -= 0.0833 * in(i + hipThreadIdx_x, j + 8, k + -1);
      }
      {
        buf0[0] -= 0.166 * in(i + hipThreadIdx_x, j + -1, k);
        buf0[0] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + -1, k);
        buf0[0] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + -1, k);
        buf0[8] -= 0.0833 * in(i + hipThreadIdx_x, j + -1, k);
      }
      {
        buf0[0] += 2.666 * in(i + hipThreadIdx_x, j, k);
        buf0[0] -= 0.166 * in(i + hipThreadIdx_x + 1, j, k);
        buf0[0] -= 0.166 * in(i + hipThreadIdx_x + -1, j, k);
        buf0[8] -= 0.166 * in(i + hipThreadIdx_x, j, k);
        buf0[8] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k);
        buf0[8] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k);
        buf0[1] -= 0.166 * in(i + hipThreadIdx_x, j, k);
        buf0[1] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k);
        buf0[1] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k);
        buf0[9] -= 0.0833 * in(i + hipThreadIdx_x, j, k);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[1 + rel] += 2.666 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf0[1 + rel] -= 0.166 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf0[1 + rel] -= 0.166 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
            buf0[9 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf0[9 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
            buf0[9 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf0[2 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf0[2 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
            buf0[2 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf0[0 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k);
            buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k);
            buf0[10 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
            buf0[8 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[7] += 2.666 * in(i + hipThreadIdx_x, j + 7, k);
        buf0[7] -= 0.166 * in(i + hipThreadIdx_x + 1, j + 7, k);
        buf0[7] -= 0.166 * in(i + hipThreadIdx_x + -1, j + 7, k);
        buf0[15] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k);
        buf0[15] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k);
        buf0[15] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k);
        buf0[6] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k);
        buf0[6] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k);
        buf0[6] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k);
        buf0[14] -= 0.0833 * in(i + hipThreadIdx_x, j + 7, k);
      }
      {
        buf0[7] -= 0.166 * in(i + hipThreadIdx_x, j + 8, k);
        buf0[7] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 8, k);
        buf0[7] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 8, k);
        buf0[15] -= 0.0833 * in(i + hipThreadIdx_x, j + 8, k);
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 6; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            buf0[8 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
            buf0[8 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 1);
            buf0[8 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 1);
            buf0[16 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
            buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1);
          }
          {
            buf0[8 + rel] += 2.666 * in(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf0[8 + rel] -= 0.166 * in(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1);
            buf0[8 + rel] -= 0.166 * in(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1);
            buf0[16 + rel] -= 0.166 * in(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf0[16 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1);
            buf0[16 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1);
            buf0[0 + rel] -= 0.166 * in(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1);
            buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1);
            buf0[9 + rel] -= 0.166 * in(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf0[9 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1);
            buf0[9 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1);
            buf0[17 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
            buf0[1 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j, k + _cg_idx2 + 1);
          }
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf0[9 + rel] += 2.666 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[9 + rel] -= 0.166 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[9 + rel] -= 0.166 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[17 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[17 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[17 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[1 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[1 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[1 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[10 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[10 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[10 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[8 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[8 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[8 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[18 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[2 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[16 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
                buf0[0 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1);
              }
              _cg_rel1 += 1;
            }
          }
          {
            buf0[15 + rel] += 2.666 * in(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf0[15 + rel] -= 0.166 * in(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1);
            buf0[15 + rel] -= 0.166 * in(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 1);
            buf0[23 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf0[23 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 1);
            buf0[23 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1);
            buf0[7 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf0[7 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 1);
            buf0[7 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1);
            buf0[14 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf0[14 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 1);
            buf0[14 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1);
            buf0[22 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
            buf0[6 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1);
          }
          {
            buf0[15 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf0[15 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 1);
            buf0[15 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 1);
            buf0[23 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
            buf0[7 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1);
          }
          _cg_rel2 += 8;
        }
      }
      {
        buf0[56] -= 0.166 * in(i + hipThreadIdx_x, j + -1, k + 7);
        buf0[56] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + -1, k + 7);
        buf0[56] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + -1, k + 7);
        buf0[48] -= 0.0833 * in(i + hipThreadIdx_x, j + -1, k + 7);
      }
      {
        buf0[56] += 2.666 * in(i + hipThreadIdx_x, j, k + 7);
        buf0[56] -= 0.166 * in(i + hipThreadIdx_x + 1, j, k + 7);
        buf0[56] -= 0.166 * in(i + hipThreadIdx_x + -1, j, k + 7);
        buf0[48] -= 0.166 * in(i + hipThreadIdx_x, j, k + 7);
        buf0[48] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + 7);
        buf0[48] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + 7);
        buf0[57] -= 0.166 * in(i + hipThreadIdx_x, j, k + 7);
        buf0[57] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + 7);
        buf0[57] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + 7);
        buf0[49] -= 0.0833 * in(i + hipThreadIdx_x, j, k + 7);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[57 + rel] += 2.666 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf0[57 + rel] -= 0.166 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf0[57 + rel] -= 0.166 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf0[49 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf0[49 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf0[49 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf0[58 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf0[58 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf0[58 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf0[56 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf0[56 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7);
            buf0[56 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7);
            buf0[50 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
            buf0[48 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[63] += 2.666 * in(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[63] -= 0.166 * in(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf0[63] -= 0.166 * in(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf0[55] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[55] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf0[55] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf0[62] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[62] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf0[62] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf0[54] -= 0.0833 * in(i + hipThreadIdx_x, j + 7, k + 7);
      }
      {
        buf0[63] -= 0.166 * in(i + hipThreadIdx_x, j + 8, k + 7);
        buf0[63] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 8, k + 7);
        buf0[63] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 8, k + 7);
        buf0[55] -= 0.0833 * in(i + hipThreadIdx_x, j + 8, k + 7);
      }
      {
        buf0[56] -= 0.0833 * in(i + hipThreadIdx_x, j + -1, k + 8);
      }
      {
        buf0[56] -= 0.166 * in(i + hipThreadIdx_x, j, k + 8);
        buf0[56] -= 0.0833 * in(i + hipThreadIdx_x + -1, j, k + 8);
        buf0[56] -= 0.0833 * in(i + hipThreadIdx_x + 1, j, k + 8);
        buf0[57] -= 0.0833 * in(i + hipThreadIdx_x, j, k + 8);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[57 + rel] -= 0.166 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
            buf0[57 + rel] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 8);
            buf0[57 + rel] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 8);
            buf0[58 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
            buf0[56 + rel] -= 0.0833 * in(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[63] -= 0.166 * in(i + hipThreadIdx_x, j + 7, k + 8);
        buf0[63] -= 0.0833 * in(i + hipThreadIdx_x + -1, j + 7, k + 8);
        buf0[63] -= 0.0833 * in(i + hipThreadIdx_x + 1, j + 7, k + 8);
        buf0[62] -= 0.0833 * in(i + hipThreadIdx_x, j + 7, k + 8);
      }
      {
        buf0[63] -= 0.0833 * in(i + hipThreadIdx_x, j + 8, k + 8);
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
            out(i + _cg_idx0, j + _cg_idx1, k + _cg_idx2) = buf0[rel];
          }
        }
      }
    }
  }
}
# 119 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/poisson/intermediate_gen/poisson.cu" 2

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
# 1 "VSBrick-poisson5.py-HIP-8x8x8-8x8" 1
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
          }
          _cg_rel2 += 1;
        }
      }
    }
    {
      bElem _cg_in000_vecbuf;
      bElem _cg_in_100_vecbuf;
      {
        // New offset [0, -1, -1]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor4 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor1 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, 0, -1]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = in.dat[neighbor3 * in.step + 448 + hipThreadIdx_x];
          _cg_in000_vecbuf = in.dat[neighbor4 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 0, -1]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[0] -= 0.166 * _cg_in000_reg;
      }
      {
        // New offset [1, 0, -1]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          _cg_in000_vecbuf = in.dat[neighbor5 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 1, -1]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor7 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor4 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, -1, 0]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor12 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor9 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in_100_vecbuf
          dev_shl(_cg_in_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = in.dat[neighbor13 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = in.dat[neighbor10 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp4;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, -1, 0]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[0] -= 0.166 * _cg_in000_reg;
        buf0[1] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [1, -1, 0]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor14 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor11 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp2;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, 0, 0]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = in.dat[neighbor12 * in.step + hipThreadIdx_x];
          _cg_in000_vecbuf = in.dat[neighbor13 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[0] -= 0.166 * _cg_in000_reg;
        buf0[1] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 0, 0]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[0] += 2.666 * _cg_in000_reg;
        buf0[1] -= 0.166 * _cg_in000_reg;
      }
      {
        // New offset [1, 0, 0]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          _cg_in000_vecbuf = in.dat[neighbor14 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[0] -= 0.166 * _cg_in000_reg;
        buf0[1] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, 1, 0]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor15 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor12 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in_100_vecbuf
          dev_shl(_cg_in_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = in.dat[neighbor16 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = in.dat[neighbor13 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp4;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 1, 0]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[0] -= 0.166 * _cg_in000_reg;
        buf0[1] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [1, 1, 0]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor17 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor14 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp2;
        }
        buf0[0] -= 0.0833 * _cg_in000_reg;
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 6; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [-1, -1, 1]
            bElem _cg_in000_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = in.dat[neighbor12 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = in.dat[neighbor9 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in_100_vecbuf
              dev_shl(_cg_in_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = in.dat[neighbor13 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = in.dat[neighbor10 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_in000_vecbuf
              dev_shl(_cg_in000_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_in000_reg = _cg_vectmp4;
            }
            buf0[1 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [0, -1, 1]
            bElem _cg_in000_reg;
            {
              _cg_in000_reg = _cg_in000_vecbuf;
            }
            buf0[1 + rel] -= 0.166 * _cg_in000_reg;
            buf0[2 + rel] -= 0.0833 * _cg_in000_reg;
            buf0[0 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [1, -1, 1]
            bElem _cg_in000_reg;
            {
              _cg_in_100_vecbuf = _cg_in000_vecbuf;
              bElem _cg_vectmp0;
              _cg_vectmp0 = in.dat[neighbor14 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = in.dat[neighbor11 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in000_vecbuf
              dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_in000_reg = _cg_vectmp2;
            }
            buf0[1 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [-1, 0, 1]
            bElem _cg_in000_reg;
            {
              _cg_in_100_vecbuf = in.dat[neighbor12 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_in000_vecbuf = in.dat[neighbor13 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_in000_reg = _cg_vectmp0;
            }
            buf0[1 + rel] -= 0.166 * _cg_in000_reg;
            buf0[2 + rel] -= 0.0833 * _cg_in000_reg;
            buf0[0 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [0, 0, 1]
            bElem _cg_in000_reg;
            {
              _cg_in000_reg = _cg_in000_vecbuf;
            }
            buf0[1 + rel] += 2.666 * _cg_in000_reg;
            buf0[2 + rel] -= 0.166 * _cg_in000_reg;
            buf0[0 + rel] -= 0.166 * _cg_in000_reg;
          }
          {
            // New offset [1, 0, 1]
            bElem _cg_in000_reg;
            {
              _cg_in_100_vecbuf = _cg_in000_vecbuf;
              _cg_in000_vecbuf = in.dat[neighbor14 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_in000_reg = _cg_vectmp0;
            }
            buf0[1 + rel] -= 0.166 * _cg_in000_reg;
            buf0[2 + rel] -= 0.0833 * _cg_in000_reg;
            buf0[0 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [-1, 1, 1]
            bElem _cg_in000_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = in.dat[neighbor15 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = in.dat[neighbor12 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in_100_vecbuf
              dev_shl(_cg_in_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = in.dat[neighbor16 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = in.dat[neighbor13 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_in000_vecbuf
              dev_shl(_cg_in000_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_in000_reg = _cg_vectmp4;
            }
            buf0[1 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [0, 1, 1]
            bElem _cg_in000_reg;
            {
              _cg_in000_reg = _cg_in000_vecbuf;
            }
            buf0[1 + rel] -= 0.166 * _cg_in000_reg;
            buf0[2 + rel] -= 0.0833 * _cg_in000_reg;
            buf0[0 + rel] -= 0.0833 * _cg_in000_reg;
          }
          {
            // New offset [1, 1, 1]
            bElem _cg_in000_reg;
            {
              _cg_in_100_vecbuf = _cg_in000_vecbuf;
              bElem _cg_vectmp0;
              _cg_vectmp0 = in.dat[neighbor17 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = in.dat[neighbor14 * in.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in000_vecbuf
              dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_in000_reg = _cg_vectmp2;
            }
            buf0[1 + rel] -= 0.0833 * _cg_in000_reg;
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [-1, -1, 7]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor12 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor9 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in_100_vecbuf
          dev_shl(_cg_in_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = in.dat[neighbor13 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = in.dat[neighbor10 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp4;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, -1, 7]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[7] -= 0.166 * _cg_in000_reg;
        buf0[6] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [1, -1, 7]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor14 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor11 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp2;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, 0, 7]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = in.dat[neighbor12 * in.step + 448 + hipThreadIdx_x];
          _cg_in000_vecbuf = in.dat[neighbor13 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[7] -= 0.166 * _cg_in000_reg;
        buf0[6] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 0, 7]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[7] += 2.666 * _cg_in000_reg;
        buf0[6] -= 0.166 * _cg_in000_reg;
      }
      {
        // New offset [1, 0, 7]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          _cg_in000_vecbuf = in.dat[neighbor14 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[7] -= 0.166 * _cg_in000_reg;
        buf0[6] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, 1, 7]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor15 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor12 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in_100_vecbuf
          dev_shl(_cg_in_100_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = in.dat[neighbor16 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = in.dat[neighbor13 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp4;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 1, 7]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[7] -= 0.166 * _cg_in000_reg;
        buf0[6] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [1, 1, 7]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor17 * in.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor14 * in.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp2;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, -1, 8]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor22 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor19 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [-1, 0, 8]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = in.dat[neighbor21 * in.step + hipThreadIdx_x];
          _cg_in000_vecbuf = in.dat[neighbor22 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 0, 8]
        bElem _cg_in000_reg;
        {
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[7] -= 0.166 * _cg_in000_reg;
      }
      {
        // New offset [1, 0, 8]
        bElem _cg_in000_reg;
        {
          _cg_in_100_vecbuf = _cg_in000_vecbuf;
          _cg_in000_vecbuf = in.dat[neighbor23 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_in_100_vecbuf ,_cg_in000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_in_100_vecbuf, _cg_in000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_in000_reg = _cg_vectmp0;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
      {
        // New offset [0, 1, 8]
        bElem _cg_in000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = in.dat[neighbor25 * in.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = in.dat[neighbor22 * in.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_in000_vecbuf
          dev_shl(_cg_in000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_in000_reg = _cg_in000_vecbuf;
        }
        buf0[7] -= 0.0833 * _cg_in000_reg;
      }
    }
    bElem *out_ref = &out.dat[neighbor13 * out.step];
    for (long sti = 0; sti < 8; ++sti)
    {
      out_ref[sti * 64 + hipThreadIdx_x] = buf0[sti];
    }
  }
}
# 141 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/poisson/intermediate_gen/poisson.cu" 2

}
