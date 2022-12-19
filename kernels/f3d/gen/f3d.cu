# 1 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/f3d/intermediate_gen/f3d.cu"
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
# 1 "VSTile-f3d1.py-HIP-8x8x64" 1
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
        buf0[0] += 0;
        buf0[0] += bIn(i + hipThreadIdx_x + -1, j + -1, k + -1) * c[0][0][0];
        buf0[0] += bIn(i + hipThreadIdx_x, j + -1, k + -1) * c[1][0][0];
        buf0[0] += bIn(i + hipThreadIdx_x + 1, j + -1, k + -1) * c[2][0][0];
      }
      {
        buf0[1] += 0;
        buf0[1] += bIn(i + hipThreadIdx_x + -1, j, k + -1) * c[0][0][0];
        buf0[1] += bIn(i + hipThreadIdx_x, j, k + -1) * c[1][0][0];
        buf0[1] += bIn(i + hipThreadIdx_x + 1, j, k + -1) * c[2][0][0];
        buf0[0] += bIn(i + hipThreadIdx_x + -1, j, k + -1) * c[0][1][0];
        buf0[0] += bIn(i + hipThreadIdx_x, j, k + -1) * c[1][1][0];
        buf0[0] += bIn(i + hipThreadIdx_x + 1, j, k + -1) * c[2][1][0];
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[2 + rel] += 0;
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + -1) * c[0][0][0];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1) * c[1][0][0];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + -1) * c[2][0][0];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + -1) * c[0][1][0];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1) * c[1][1][0];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + -1) * c[2][1][0];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + -1) * c[0][2][0];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + -1) * c[1][2][0];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + -1) * c[2][2][0];
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[7] += bIn(i + hipThreadIdx_x + -1, j + 7, k + -1) * c[0][1][0];
        buf0[7] += bIn(i + hipThreadIdx_x, j + 7, k + -1) * c[1][1][0];
        buf0[7] += bIn(i + hipThreadIdx_x + 1, j + 7, k + -1) * c[2][1][0];
        buf0[6] += bIn(i + hipThreadIdx_x + -1, j + 7, k + -1) * c[0][2][0];
        buf0[6] += bIn(i + hipThreadIdx_x, j + 7, k + -1) * c[1][2][0];
        buf0[6] += bIn(i + hipThreadIdx_x + 1, j + 7, k + -1) * c[2][2][0];
      }
      {
        buf0[7] += bIn(i + hipThreadIdx_x + -1, j + 8, k + -1) * c[0][2][0];
        buf0[7] += bIn(i + hipThreadIdx_x, j + 8, k + -1) * c[1][2][0];
        buf0[7] += bIn(i + hipThreadIdx_x + 1, j + 8, k + -1) * c[2][2][0];
      }
      {
        buf0[8] += 0;
        buf0[8] += bIn(i + hipThreadIdx_x + -1, j + -1, k) * c[0][0][0];
        buf0[8] += bIn(i + hipThreadIdx_x, j + -1, k) * c[1][0][0];
        buf0[8] += bIn(i + hipThreadIdx_x + 1, j + -1, k) * c[2][0][0];
        buf0[0] += bIn(i + hipThreadIdx_x + -1, j + -1, k) * c[0][0][1];
        buf0[0] += bIn(i + hipThreadIdx_x, j + -1, k) * c[1][0][1];
        buf0[0] += bIn(i + hipThreadIdx_x + 1, j + -1, k) * c[2][0][1];
      }
      {
        buf0[9] += 0;
        buf0[9] += bIn(i + hipThreadIdx_x + -1, j, k) * c[0][0][0];
        buf0[9] += bIn(i + hipThreadIdx_x, j, k) * c[1][0][0];
        buf0[9] += bIn(i + hipThreadIdx_x + 1, j, k) * c[2][0][0];
        buf0[1] += bIn(i + hipThreadIdx_x + -1, j, k) * c[0][0][1];
        buf0[1] += bIn(i + hipThreadIdx_x, j, k) * c[1][0][1];
        buf0[1] += bIn(i + hipThreadIdx_x + 1, j, k) * c[2][0][1];
        buf0[8] += bIn(i + hipThreadIdx_x + -1, j, k) * c[0][1][0];
        buf0[8] += bIn(i + hipThreadIdx_x, j, k) * c[1][1][0];
        buf0[8] += bIn(i + hipThreadIdx_x + 1, j, k) * c[2][1][0];
        buf0[0] += bIn(i + hipThreadIdx_x + -1, j, k) * c[0][1][1];
        buf0[0] += bIn(i + hipThreadIdx_x, j, k) * c[1][1][1];
        buf0[0] += bIn(i + hipThreadIdx_x + 1, j, k) * c[2][1][1];
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[10 + rel] += 0;
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k) * c[0][0][0];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k) * c[1][0][0];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k) * c[2][0][0];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k) * c[0][0][1];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k) * c[1][0][1];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k) * c[2][0][1];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k) * c[0][1][0];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k) * c[1][1][0];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k) * c[2][1][0];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k) * c[0][1][1];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k) * c[1][1][1];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k) * c[2][1][1];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k) * c[0][2][0];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k) * c[1][2][0];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k) * c[2][2][0];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k) * c[0][2][1];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k) * c[1][2][1];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k) * c[2][2][1];
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[15] += bIn(i + hipThreadIdx_x + -1, j + 7, k) * c[0][1][0];
        buf0[15] += bIn(i + hipThreadIdx_x, j + 7, k) * c[1][1][0];
        buf0[15] += bIn(i + hipThreadIdx_x + 1, j + 7, k) * c[2][1][0];
        buf0[7] += bIn(i + hipThreadIdx_x + -1, j + 7, k) * c[0][1][1];
        buf0[7] += bIn(i + hipThreadIdx_x, j + 7, k) * c[1][1][1];
        buf0[7] += bIn(i + hipThreadIdx_x + 1, j + 7, k) * c[2][1][1];
        buf0[14] += bIn(i + hipThreadIdx_x + -1, j + 7, k) * c[0][2][0];
        buf0[14] += bIn(i + hipThreadIdx_x, j + 7, k) * c[1][2][0];
        buf0[14] += bIn(i + hipThreadIdx_x + 1, j + 7, k) * c[2][2][0];
        buf0[6] += bIn(i + hipThreadIdx_x + -1, j + 7, k) * c[0][2][1];
        buf0[6] += bIn(i + hipThreadIdx_x, j + 7, k) * c[1][2][1];
        buf0[6] += bIn(i + hipThreadIdx_x + 1, j + 7, k) * c[2][2][1];
      }
      {
        buf0[15] += bIn(i + hipThreadIdx_x + -1, j + 8, k) * c[0][2][0];
        buf0[15] += bIn(i + hipThreadIdx_x, j + 8, k) * c[1][2][0];
        buf0[15] += bIn(i + hipThreadIdx_x + 1, j + 8, k) * c[2][2][0];
        buf0[7] += bIn(i + hipThreadIdx_x + -1, j + 8, k) * c[0][2][1];
        buf0[7] += bIn(i + hipThreadIdx_x, j + 8, k) * c[1][2][1];
        buf0[7] += bIn(i + hipThreadIdx_x + 1, j + 8, k) * c[2][2][1];
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 6; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            buf0[16 + rel] += 0;
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 1) * c[0][0][0];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1) * c[1][0][0];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 1) * c[2][0][0];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 1) * c[0][0][1];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1) * c[1][0][1];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 1) * c[2][0][1];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 1) * c[0][0][2];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 1) * c[1][0][2];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 1) * c[2][0][2];
          }
          {
            buf0[17 + rel] += 0;
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1) * c[0][0][0];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 1) * c[1][0][0];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1) * c[2][0][0];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1) * c[0][0][1];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 1) * c[1][0][1];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1) * c[2][0][1];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1) * c[0][0][2];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 1) * c[1][0][2];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1) * c[2][0][2];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1) * c[0][1][0];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 1) * c[1][1][0];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1) * c[2][1][0];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1) * c[0][1][1];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 1) * c[1][1][1];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1) * c[2][1][1];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 1) * c[0][1][2];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 1) * c[1][1][2];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 1) * c[2][1][2];
          }
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf0[18 + rel] += 0;
                buf0[18 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[0][0][0];
                buf0[18 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[1][0][0];
                buf0[18 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[2][0][0];
                buf0[10 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[0][0][1];
                buf0[10 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[1][0][1];
                buf0[10 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[2][0][1];
                buf0[2 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[0][0][2];
                buf0[2 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[1][0][2];
                buf0[2 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[2][0][2];
                buf0[17 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[0][1][0];
                buf0[17 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[1][1][0];
                buf0[17 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[2][1][0];
                buf0[9 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[0][1][1];
                buf0[9 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[1][1][1];
                buf0[9 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[2][1][1];
                buf0[1 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[0][1][2];
                buf0[1 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[1][1][2];
                buf0[1 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[2][1][2];
                buf0[16 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[0][2][0];
                buf0[16 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[1][2][0];
                buf0[16 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[2][2][0];
                buf0[8 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[0][2][1];
                buf0[8 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[1][2][1];
                buf0[8 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[2][2][1];
                buf0[0 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[0][2][2];
                buf0[0 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[1][2][2];
                buf0[0 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + _cg_idx2 + 1) * c[2][2][2];
              }
              _cg_rel1 += 1;
            }
          }
          {
            buf0[23 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 1) * c[0][1][0];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1) * c[1][1][0];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1) * c[2][1][0];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 1) * c[0][1][1];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1) * c[1][1][1];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1) * c[2][1][1];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 1) * c[0][1][2];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1) * c[1][1][2];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1) * c[2][1][2];
            buf0[22 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 1) * c[0][2][0];
            buf0[22 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1) * c[1][2][0];
            buf0[22 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1) * c[2][2][0];
            buf0[14 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 1) * c[0][2][1];
            buf0[14 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1) * c[1][2][1];
            buf0[14 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1) * c[2][2][1];
            buf0[6 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 1) * c[0][2][2];
            buf0[6 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 1) * c[1][2][2];
            buf0[6 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 1) * c[2][2][2];
          }
          {
            buf0[23 + rel] += bIn(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 1) * c[0][2][0];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1) * c[1][2][0];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 1) * c[2][2][0];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 1) * c[0][2][1];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1) * c[1][2][1];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 1) * c[2][2][1];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 1) * c[0][2][2];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 1) * c[1][2][2];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 1) * c[2][2][2];
          }
          _cg_rel2 += 8;
        }
      }
      {
        buf0[56] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 7) * c[0][0][1];
        buf0[56] += bIn(i + hipThreadIdx_x, j + -1, k + 7) * c[1][0][1];
        buf0[56] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 7) * c[2][0][1];
        buf0[48] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 7) * c[0][0][2];
        buf0[48] += bIn(i + hipThreadIdx_x, j + -1, k + 7) * c[1][0][2];
        buf0[48] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 7) * c[2][0][2];
      }
      {
        buf0[57] += bIn(i + hipThreadIdx_x + -1, j, k + 7) * c[0][0][1];
        buf0[57] += bIn(i + hipThreadIdx_x, j, k + 7) * c[1][0][1];
        buf0[57] += bIn(i + hipThreadIdx_x + 1, j, k + 7) * c[2][0][1];
        buf0[49] += bIn(i + hipThreadIdx_x + -1, j, k + 7) * c[0][0][2];
        buf0[49] += bIn(i + hipThreadIdx_x, j, k + 7) * c[1][0][2];
        buf0[49] += bIn(i + hipThreadIdx_x + 1, j, k + 7) * c[2][0][2];
        buf0[56] += bIn(i + hipThreadIdx_x + -1, j, k + 7) * c[0][1][1];
        buf0[56] += bIn(i + hipThreadIdx_x, j, k + 7) * c[1][1][1];
        buf0[56] += bIn(i + hipThreadIdx_x + 1, j, k + 7) * c[2][1][1];
        buf0[48] += bIn(i + hipThreadIdx_x + -1, j, k + 7) * c[0][1][2];
        buf0[48] += bIn(i + hipThreadIdx_x, j, k + 7) * c[1][1][2];
        buf0[48] += bIn(i + hipThreadIdx_x + 1, j, k + 7) * c[2][1][2];
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[58 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7) * c[0][0][1];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7) * c[1][0][1];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7) * c[2][0][1];
            buf0[50 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7) * c[0][0][2];
            buf0[50 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7) * c[1][0][2];
            buf0[50 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7) * c[2][0][2];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7) * c[0][1][1];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7) * c[1][1][1];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7) * c[2][1][1];
            buf0[49 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7) * c[0][1][2];
            buf0[49 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7) * c[1][1][2];
            buf0[49 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7) * c[2][1][2];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7) * c[0][2][1];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7) * c[1][2][1];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7) * c[2][2][1];
            buf0[48 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 7) * c[0][2][2];
            buf0[48 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 7) * c[1][2][2];
            buf0[48 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 7) * c[2][2][2];
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[63] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 7) * c[0][1][1];
        buf0[63] += bIn(i + hipThreadIdx_x, j + 7, k + 7) * c[1][1][1];
        buf0[63] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 7) * c[2][1][1];
        buf0[55] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 7) * c[0][1][2];
        buf0[55] += bIn(i + hipThreadIdx_x, j + 7, k + 7) * c[1][1][2];
        buf0[55] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 7) * c[2][1][2];
        buf0[62] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 7) * c[0][2][1];
        buf0[62] += bIn(i + hipThreadIdx_x, j + 7, k + 7) * c[1][2][1];
        buf0[62] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 7) * c[2][2][1];
        buf0[54] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 7) * c[0][2][2];
        buf0[54] += bIn(i + hipThreadIdx_x, j + 7, k + 7) * c[1][2][2];
        buf0[54] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 7) * c[2][2][2];
      }
      {
        buf0[63] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 7) * c[0][2][1];
        buf0[63] += bIn(i + hipThreadIdx_x, j + 8, k + 7) * c[1][2][1];
        buf0[63] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 7) * c[2][2][1];
        buf0[55] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 7) * c[0][2][2];
        buf0[55] += bIn(i + hipThreadIdx_x, j + 8, k + 7) * c[1][2][2];
        buf0[55] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 7) * c[2][2][2];
      }
      {
        buf0[56] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 8) * c[0][0][2];
        buf0[56] += bIn(i + hipThreadIdx_x, j + -1, k + 8) * c[1][0][2];
        buf0[56] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 8) * c[2][0][2];
      }
      {
        buf0[57] += bIn(i + hipThreadIdx_x + -1, j, k + 8) * c[0][0][2];
        buf0[57] += bIn(i + hipThreadIdx_x, j, k + 8) * c[1][0][2];
        buf0[57] += bIn(i + hipThreadIdx_x + 1, j, k + 8) * c[2][0][2];
        buf0[56] += bIn(i + hipThreadIdx_x + -1, j, k + 8) * c[0][1][2];
        buf0[56] += bIn(i + hipThreadIdx_x, j, k + 8) * c[1][1][2];
        buf0[56] += bIn(i + hipThreadIdx_x + 1, j, k + 8) * c[2][1][2];
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 6; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[58 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 8) * c[0][0][2];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8) * c[1][0][2];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 8) * c[2][0][2];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 8) * c[0][1][2];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8) * c[1][1][2];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 8) * c[2][1][2];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 1, k + 8) * c[0][2][2];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + 8) * c[1][2][2];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 1, k + 8) * c[2][2][2];
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[63] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 8) * c[0][1][2];
        buf0[63] += bIn(i + hipThreadIdx_x, j + 7, k + 8) * c[1][1][2];
        buf0[63] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 8) * c[2][1][2];
        buf0[62] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 8) * c[0][2][2];
        buf0[62] += bIn(i + hipThreadIdx_x, j + 7, k + 8) * c[1][2][2];
        buf0[62] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 8) * c[2][2][2];
      }
      {
        buf0[63] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 8) * c[0][2][2];
        buf0[63] += bIn(i + hipThreadIdx_x, j + 8, k + 8) * c[1][2][2];
        buf0[63] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 8) * c[2][2][2];
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
            bOut(i + _cg_idx0, j + _cg_idx1, k + _cg_idx2) = buf0[rel];
          }
        }
      }
    }
  }
}
# 51 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/f3d/intermediate_gen/f3d.cu" 2

}
#undef bIn
#undef bOut
__global__ void f3d_codegen_bricks1(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem (*c)[8][8]) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
# 1 "VSBrick-f3d1.py-HIP-8x8x8-8x8" 1
{
  auto *binfo = bOut.bInfo;
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
      bElem _cg_bIn_1_1_1_vecbuf;
      bElem _cg_bIn0_1_1_vecbuf;
      {
        // New offset [0, 0, 0]
        bElem _cg_bIn_1_1_1_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor3 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor0 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn_1_1_1_vecbuf
          dev_shl(_cg_bIn_1_1_1_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor4 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor1 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_bIn0_1_1_vecbuf
          dev_shl(_cg_bIn0_1_1_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp4;
        }
        buf0[0] += 0;
        buf0[0] += _cg_bIn_1_1_1_reg * c[0][0][0];
      }
      {
        // New offset [1, 0, 0]
        bElem _cg_bIn_1_1_1_reg;
        {
          _cg_bIn_1_1_1_vecbuf = _cg_bIn0_1_1_vecbuf;
          _cg_bIn_1_1_1_reg = _cg_bIn_1_1_1_vecbuf;
        }
        buf0[0] += _cg_bIn_1_1_1_reg * c[1][0][0];
      }
      {
        // New offset [2, 0, 0]
        bElem _cg_bIn_1_1_1_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor5 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor2 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn0_1_1_vecbuf
          dev_shl(_cg_bIn0_1_1_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp2;
        }
        buf0[0] += _cg_bIn_1_1_1_reg * c[2][0][0];
      }
      {
        // New offset [0, 1, 0]
        bElem _cg_bIn_1_1_1_reg;
        {
          _cg_bIn_1_1_1_vecbuf = bIn.dat[neighbor3 * bIn.step + 448 + hipThreadIdx_x];
          _cg_bIn0_1_1_vecbuf = bIn.dat[neighbor4 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp0;
        }
        buf0[0] += _cg_bIn_1_1_1_reg * c[0][1][0];
      }
      {
        // New offset [1, 1, 0]
        bElem _cg_bIn_1_1_1_reg;
        {
          _cg_bIn_1_1_1_vecbuf = _cg_bIn0_1_1_vecbuf;
          _cg_bIn_1_1_1_reg = _cg_bIn_1_1_1_vecbuf;
        }
        buf0[0] += _cg_bIn_1_1_1_reg * c[1][1][0];
      }
      {
        // New offset [2, 1, 0]
        bElem _cg_bIn_1_1_1_reg;
        {
          _cg_bIn0_1_1_vecbuf = bIn.dat[neighbor5 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp0;
        }
        buf0[0] += _cg_bIn_1_1_1_reg * c[2][1][0];
      }
      {
        // New offset [0, 2, 0]
        bElem _cg_bIn_1_1_1_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor6 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor3 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn_1_1_1_vecbuf
          dev_shl(_cg_bIn_1_1_1_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor7 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor4 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_bIn0_1_1_vecbuf
          dev_shl(_cg_bIn0_1_1_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp4;
        }
        buf0[0] += _cg_bIn_1_1_1_reg * c[0][2][0];
      }
      {
        // New offset [1, 2, 0]
        bElem _cg_bIn_1_1_1_reg;
        {
          _cg_bIn_1_1_1_vecbuf = _cg_bIn0_1_1_vecbuf;
          _cg_bIn_1_1_1_reg = _cg_bIn_1_1_1_vecbuf;
        }
        buf0[0] += _cg_bIn_1_1_1_reg * c[1][2][0];
      }
      {
        // New offset [2, 2, 0]
        bElem _cg_bIn_1_1_1_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor8 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor5 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn0_1_1_vecbuf
          dev_shl(_cg_bIn0_1_1_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp2;
        }
        buf0[0] += _cg_bIn_1_1_1_reg * c[2][2][0];
      }
      {
        // New offset [0, 0, 1]
        bElem _cg_bIn_1_1_1_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor12 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor9 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn_1_1_1_vecbuf
          dev_shl(_cg_bIn_1_1_1_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor10 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_bIn0_1_1_vecbuf
          dev_shl(_cg_bIn0_1_1_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp4;
        }
        buf0[1] += 0;
        buf0[1] += _cg_bIn_1_1_1_reg * c[0][0][0];
        buf0[0] += _cg_bIn_1_1_1_reg * c[0][0][1];
      }
      {
        // New offset [1, 0, 1]
        bElem _cg_bIn_1_1_1_reg;
        {
          _cg_bIn_1_1_1_vecbuf = _cg_bIn0_1_1_vecbuf;
          _cg_bIn_1_1_1_reg = _cg_bIn_1_1_1_vecbuf;
        }
        buf0[1] += _cg_bIn_1_1_1_reg * c[1][0][0];
        buf0[0] += _cg_bIn_1_1_1_reg * c[1][0][1];
      }
      {
        // New offset [2, 0, 1]
        bElem _cg_bIn_1_1_1_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor14 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor11 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn0_1_1_vecbuf
          dev_shl(_cg_bIn0_1_1_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp2;
        }
        buf0[1] += _cg_bIn_1_1_1_reg * c[2][0][0];
        buf0[0] += _cg_bIn_1_1_1_reg * c[2][0][1];
      }
      {
        // New offset [0, 1, 1]
        bElem _cg_bIn_1_1_1_reg;
        {
          _cg_bIn_1_1_1_vecbuf = bIn.dat[neighbor12 * bIn.step + hipThreadIdx_x];
          _cg_bIn0_1_1_vecbuf = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp0;
        }
        buf0[1] += _cg_bIn_1_1_1_reg * c[0][1][0];
        buf0[0] += _cg_bIn_1_1_1_reg * c[0][1][1];
      }
      {
        // New offset [1, 1, 1]
        bElem _cg_bIn_1_1_1_reg;
        {
          _cg_bIn_1_1_1_vecbuf = _cg_bIn0_1_1_vecbuf;
          _cg_bIn_1_1_1_reg = _cg_bIn_1_1_1_vecbuf;
        }
        buf0[1] += _cg_bIn_1_1_1_reg * c[1][1][0];
        buf0[0] += _cg_bIn_1_1_1_reg * c[1][1][1];
      }
      {
        // New offset [2, 1, 1]
        bElem _cg_bIn_1_1_1_reg;
        {
          _cg_bIn0_1_1_vecbuf = bIn.dat[neighbor14 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp0;
        }
        buf0[1] += _cg_bIn_1_1_1_reg * c[2][1][0];
        buf0[0] += _cg_bIn_1_1_1_reg * c[2][1][1];
      }
      {
        // New offset [0, 2, 1]
        bElem _cg_bIn_1_1_1_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor15 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor12 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn_1_1_1_vecbuf
          dev_shl(_cg_bIn_1_1_1_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor16 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_bIn0_1_1_vecbuf
          dev_shl(_cg_bIn0_1_1_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp4;
        }
        buf0[1] += _cg_bIn_1_1_1_reg * c[0][2][0];
        buf0[0] += _cg_bIn_1_1_1_reg * c[0][2][1];
      }
      {
        // New offset [1, 2, 1]
        bElem _cg_bIn_1_1_1_reg;
        {
          _cg_bIn_1_1_1_vecbuf = _cg_bIn0_1_1_vecbuf;
          _cg_bIn_1_1_1_reg = _cg_bIn_1_1_1_vecbuf;
        }
        buf0[1] += _cg_bIn_1_1_1_reg * c[1][2][0];
        buf0[0] += _cg_bIn_1_1_1_reg * c[1][2][1];
      }
      {
        // New offset [2, 2, 1]
        bElem _cg_bIn_1_1_1_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor17 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor14 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn0_1_1_vecbuf
          dev_shl(_cg_bIn0_1_1_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp2;
        }
        buf0[1] += _cg_bIn_1_1_1_reg * c[2][2][0];
        buf0[0] += _cg_bIn_1_1_1_reg * c[2][2][1];
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 6; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 2]
            bElem _cg_bIn_1_1_1_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor12 * bIn.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor9 * bIn.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn_1_1_1_vecbuf
              dev_shl(_cg_bIn_1_1_1_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = bIn.dat[neighbor13 * bIn.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = bIn.dat[neighbor10 * bIn.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_bIn0_1_1_vecbuf
              dev_shl(_cg_bIn0_1_1_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 7 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_bIn_1_1_1_reg = _cg_vectmp4;
            }
            buf0[2 + rel] += 0;
            buf0[2 + rel] += _cg_bIn_1_1_1_reg * c[0][0][0];
            buf0[1 + rel] += _cg_bIn_1_1_1_reg * c[0][0][1];
            buf0[0 + rel] += _cg_bIn_1_1_1_reg * c[0][0][2];
          }
          {
            // New offset [1, 0, 2]
            bElem _cg_bIn_1_1_1_reg;
            {
              _cg_bIn_1_1_1_vecbuf = _cg_bIn0_1_1_vecbuf;
              _cg_bIn_1_1_1_reg = _cg_bIn_1_1_1_vecbuf;
            }
            buf0[2 + rel] += _cg_bIn_1_1_1_reg * c[1][0][0];
            buf0[1 + rel] += _cg_bIn_1_1_1_reg * c[1][0][1];
            buf0[0 + rel] += _cg_bIn_1_1_1_reg * c[1][0][2];
          }
          {
            // New offset [2, 0, 2]
            bElem _cg_bIn_1_1_1_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor14 * bIn.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor11 * bIn.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn0_1_1_vecbuf
              dev_shl(_cg_bIn0_1_1_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_bIn_1_1_1_reg = _cg_vectmp2;
            }
            buf0[2 + rel] += _cg_bIn_1_1_1_reg * c[2][0][0];
            buf0[1 + rel] += _cg_bIn_1_1_1_reg * c[2][0][1];
            buf0[0 + rel] += _cg_bIn_1_1_1_reg * c[2][0][2];
          }
          {
            // New offset [0, 1, 2]
            bElem _cg_bIn_1_1_1_reg;
            {
              _cg_bIn_1_1_1_vecbuf = bIn.dat[neighbor12 * bIn.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_bIn0_1_1_vecbuf = bIn.dat[neighbor13 * bIn.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_bIn_1_1_1_reg = _cg_vectmp0;
            }
            buf0[2 + rel] += _cg_bIn_1_1_1_reg * c[0][1][0];
            buf0[1 + rel] += _cg_bIn_1_1_1_reg * c[0][1][1];
            buf0[0 + rel] += _cg_bIn_1_1_1_reg * c[0][1][2];
          }
          {
            // New offset [1, 1, 2]
            bElem _cg_bIn_1_1_1_reg;
            {
              _cg_bIn_1_1_1_vecbuf = _cg_bIn0_1_1_vecbuf;
              _cg_bIn_1_1_1_reg = _cg_bIn_1_1_1_vecbuf;
            }
            buf0[2 + rel] += _cg_bIn_1_1_1_reg * c[1][1][0];
            buf0[1 + rel] += _cg_bIn_1_1_1_reg * c[1][1][1];
            buf0[0 + rel] += _cg_bIn_1_1_1_reg * c[1][1][2];
          }
          {
            // New offset [2, 1, 2]
            bElem _cg_bIn_1_1_1_reg;
            {
              _cg_bIn0_1_1_vecbuf = bIn.dat[neighbor14 * bIn.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_bIn_1_1_1_reg = _cg_vectmp0;
            }
            buf0[2 + rel] += _cg_bIn_1_1_1_reg * c[2][1][0];
            buf0[1 + rel] += _cg_bIn_1_1_1_reg * c[2][1][1];
            buf0[0 + rel] += _cg_bIn_1_1_1_reg * c[2][1][2];
          }
          {
            // New offset [0, 2, 2]
            bElem _cg_bIn_1_1_1_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor15 * bIn.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor12 * bIn.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn_1_1_1_vecbuf
              dev_shl(_cg_bIn_1_1_1_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = bIn.dat[neighbor16 * bIn.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = bIn.dat[neighbor13 * bIn.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_bIn0_1_1_vecbuf
              dev_shl(_cg_bIn0_1_1_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 7 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_bIn_1_1_1_reg = _cg_vectmp4;
            }
            buf0[2 + rel] += _cg_bIn_1_1_1_reg * c[0][2][0];
            buf0[1 + rel] += _cg_bIn_1_1_1_reg * c[0][2][1];
            buf0[0 + rel] += _cg_bIn_1_1_1_reg * c[0][2][2];
          }
          {
            // New offset [1, 2, 2]
            bElem _cg_bIn_1_1_1_reg;
            {
              _cg_bIn_1_1_1_vecbuf = _cg_bIn0_1_1_vecbuf;
              _cg_bIn_1_1_1_reg = _cg_bIn_1_1_1_vecbuf;
            }
            buf0[2 + rel] += _cg_bIn_1_1_1_reg * c[1][2][0];
            buf0[1 + rel] += _cg_bIn_1_1_1_reg * c[1][2][1];
            buf0[0 + rel] += _cg_bIn_1_1_1_reg * c[1][2][2];
          }
          {
            // New offset [2, 2, 2]
            bElem _cg_bIn_1_1_1_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor17 * bIn.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor14 * bIn.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn0_1_1_vecbuf
              dev_shl(_cg_bIn0_1_1_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_bIn_1_1_1_reg = _cg_vectmp2;
            }
            buf0[2 + rel] += _cg_bIn_1_1_1_reg * c[2][2][0];
            buf0[1 + rel] += _cg_bIn_1_1_1_reg * c[2][2][1];
            buf0[0 + rel] += _cg_bIn_1_1_1_reg * c[2][2][2];
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [0, 0, 8]
        bElem _cg_bIn_1_1_1_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor12 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor9 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn_1_1_1_vecbuf
          dev_shl(_cg_bIn_1_1_1_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor10 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_bIn0_1_1_vecbuf
          dev_shl(_cg_bIn0_1_1_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp4;
        }
        buf0[7] += _cg_bIn_1_1_1_reg * c[0][0][1];
        buf0[6] += _cg_bIn_1_1_1_reg * c[0][0][2];
      }
      {
        // New offset [1, 0, 8]
        bElem _cg_bIn_1_1_1_reg;
        {
          _cg_bIn_1_1_1_vecbuf = _cg_bIn0_1_1_vecbuf;
          _cg_bIn_1_1_1_reg = _cg_bIn_1_1_1_vecbuf;
        }
        buf0[7] += _cg_bIn_1_1_1_reg * c[1][0][1];
        buf0[6] += _cg_bIn_1_1_1_reg * c[1][0][2];
      }
      {
        // New offset [2, 0, 8]
        bElem _cg_bIn_1_1_1_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor14 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor11 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn0_1_1_vecbuf
          dev_shl(_cg_bIn0_1_1_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp2;
        }
        buf0[7] += _cg_bIn_1_1_1_reg * c[2][0][1];
        buf0[6] += _cg_bIn_1_1_1_reg * c[2][0][2];
      }
      {
        // New offset [0, 1, 8]
        bElem _cg_bIn_1_1_1_reg;
        {
          _cg_bIn_1_1_1_vecbuf = bIn.dat[neighbor12 * bIn.step + 448 + hipThreadIdx_x];
          _cg_bIn0_1_1_vecbuf = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_1_1_1_reg * c[0][1][1];
        buf0[6] += _cg_bIn_1_1_1_reg * c[0][1][2];
      }
      {
        // New offset [1, 1, 8]
        bElem _cg_bIn_1_1_1_reg;
        {
          _cg_bIn_1_1_1_vecbuf = _cg_bIn0_1_1_vecbuf;
          _cg_bIn_1_1_1_reg = _cg_bIn_1_1_1_vecbuf;
        }
        buf0[7] += _cg_bIn_1_1_1_reg * c[1][1][1];
        buf0[6] += _cg_bIn_1_1_1_reg * c[1][1][2];
      }
      {
        // New offset [2, 1, 8]
        bElem _cg_bIn_1_1_1_reg;
        {
          _cg_bIn0_1_1_vecbuf = bIn.dat[neighbor14 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_1_1_1_reg * c[2][1][1];
        buf0[6] += _cg_bIn_1_1_1_reg * c[2][1][2];
      }
      {
        // New offset [0, 2, 8]
        bElem _cg_bIn_1_1_1_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor15 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor12 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn_1_1_1_vecbuf
          dev_shl(_cg_bIn_1_1_1_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor16 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_bIn0_1_1_vecbuf
          dev_shl(_cg_bIn0_1_1_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp4;
        }
        buf0[7] += _cg_bIn_1_1_1_reg * c[0][2][1];
        buf0[6] += _cg_bIn_1_1_1_reg * c[0][2][2];
      }
      {
        // New offset [1, 2, 8]
        bElem _cg_bIn_1_1_1_reg;
        {
          _cg_bIn_1_1_1_vecbuf = _cg_bIn0_1_1_vecbuf;
          _cg_bIn_1_1_1_reg = _cg_bIn_1_1_1_vecbuf;
        }
        buf0[7] += _cg_bIn_1_1_1_reg * c[1][2][1];
        buf0[6] += _cg_bIn_1_1_1_reg * c[1][2][2];
      }
      {
        // New offset [2, 2, 8]
        bElem _cg_bIn_1_1_1_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor17 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor14 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn0_1_1_vecbuf
          dev_shl(_cg_bIn0_1_1_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp2;
        }
        buf0[7] += _cg_bIn_1_1_1_reg * c[2][2][1];
        buf0[6] += _cg_bIn_1_1_1_reg * c[2][2][2];
      }
      {
        // New offset [0, 0, 9]
        bElem _cg_bIn_1_1_1_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor21 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor18 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn_1_1_1_vecbuf
          dev_shl(_cg_bIn_1_1_1_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor22 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor19 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_bIn0_1_1_vecbuf
          dev_shl(_cg_bIn0_1_1_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp4;
        }
        buf0[7] += _cg_bIn_1_1_1_reg * c[0][0][2];
      }
      {
        // New offset [1, 0, 9]
        bElem _cg_bIn_1_1_1_reg;
        {
          _cg_bIn_1_1_1_vecbuf = _cg_bIn0_1_1_vecbuf;
          _cg_bIn_1_1_1_reg = _cg_bIn_1_1_1_vecbuf;
        }
        buf0[7] += _cg_bIn_1_1_1_reg * c[1][0][2];
      }
      {
        // New offset [2, 0, 9]
        bElem _cg_bIn_1_1_1_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor23 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor20 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn0_1_1_vecbuf
          dev_shl(_cg_bIn0_1_1_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp2;
        }
        buf0[7] += _cg_bIn_1_1_1_reg * c[2][0][2];
      }
      {
        // New offset [0, 1, 9]
        bElem _cg_bIn_1_1_1_reg;
        {
          _cg_bIn_1_1_1_vecbuf = bIn.dat[neighbor21 * bIn.step + hipThreadIdx_x];
          _cg_bIn0_1_1_vecbuf = bIn.dat[neighbor22 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_1_1_1_reg * c[0][1][2];
      }
      {
        // New offset [1, 1, 9]
        bElem _cg_bIn_1_1_1_reg;
        {
          _cg_bIn_1_1_1_vecbuf = _cg_bIn0_1_1_vecbuf;
          _cg_bIn_1_1_1_reg = _cg_bIn_1_1_1_vecbuf;
        }
        buf0[7] += _cg_bIn_1_1_1_reg * c[1][1][2];
      }
      {
        // New offset [2, 1, 9]
        bElem _cg_bIn_1_1_1_reg;
        {
          _cg_bIn0_1_1_vecbuf = bIn.dat[neighbor23 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_1_1_1_reg * c[2][1][2];
      }
      {
        // New offset [0, 2, 9]
        bElem _cg_bIn_1_1_1_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor24 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor21 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn_1_1_1_vecbuf
          dev_shl(_cg_bIn_1_1_1_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor25 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor22 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_bIn0_1_1_vecbuf
          dev_shl(_cg_bIn0_1_1_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 7 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp4;
        }
        buf0[7] += _cg_bIn_1_1_1_reg * c[0][2][2];
      }
      {
        // New offset [1, 2, 9]
        bElem _cg_bIn_1_1_1_reg;
        {
          _cg_bIn_1_1_1_vecbuf = _cg_bIn0_1_1_vecbuf;
          _cg_bIn_1_1_1_reg = _cg_bIn_1_1_1_vecbuf;
        }
        buf0[7] += _cg_bIn_1_1_1_reg * c[1][2][2];
      }
      {
        // New offset [2, 2, 9]
        bElem _cg_bIn_1_1_1_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor26 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor23 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn0_1_1_vecbuf
          dev_shl(_cg_bIn0_1_1_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_1_1_vecbuf ,_cg_bIn0_1_1_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_1_1_vecbuf, _cg_bIn0_1_1_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_1_1_1_reg = _cg_vectmp2;
        }
        buf0[7] += _cg_bIn_1_1_1_reg * c[2][2][2];
      }
    }
    bElem *bOut_ref = &bOut.dat[neighbor13 * bOut.step];
    for (long sti = 0; sti < 8; ++sti)
    {
      bOut_ref[sti * 64 + hipThreadIdx_x] = buf0[sti];
    }
  }
}
# 57 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/f3d/intermediate_gen/f3d.cu" 2

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
# 1 "VSTile-f3d2.py-HIP-8x8x64" 1
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
        buf0[0] += 0;
        buf0[0] += bIn(i + hipThreadIdx_x + -2, j + -2, k + -2) * c[0][0][0];
        buf0[0] += bIn(i + hipThreadIdx_x + -1, j + -2, k + -2) * c[1][0][0];
        buf0[0] += bIn(i + hipThreadIdx_x, j + -2, k + -2) * c[2][0][0];
        buf0[0] += bIn(i + hipThreadIdx_x + 1, j + -2, k + -2) * c[3][0][0];
        buf0[0] += bIn(i + hipThreadIdx_x + 2, j + -2, k + -2) * c[4][0][0];
      }
      {
        buf0[1] += 0;
        buf0[1] += bIn(i + hipThreadIdx_x + -2, j + -1, k + -2) * c[0][0][0];
        buf0[1] += bIn(i + hipThreadIdx_x + -1, j + -1, k + -2) * c[1][0][0];
        buf0[1] += bIn(i + hipThreadIdx_x, j + -1, k + -2) * c[2][0][0];
        buf0[1] += bIn(i + hipThreadIdx_x + 1, j + -1, k + -2) * c[3][0][0];
        buf0[1] += bIn(i + hipThreadIdx_x + 2, j + -1, k + -2) * c[4][0][0];
        buf0[0] += bIn(i + hipThreadIdx_x + -2, j + -1, k + -2) * c[0][1][0];
        buf0[0] += bIn(i + hipThreadIdx_x + -1, j + -1, k + -2) * c[1][1][0];
        buf0[0] += bIn(i + hipThreadIdx_x, j + -1, k + -2) * c[2][1][0];
        buf0[0] += bIn(i + hipThreadIdx_x + 1, j + -1, k + -2) * c[3][1][0];
        buf0[0] += bIn(i + hipThreadIdx_x + 2, j + -1, k + -2) * c[4][1][0];
      }
      {
        buf0[2] += 0;
        buf0[2] += bIn(i + hipThreadIdx_x + -2, j, k + -2) * c[0][0][0];
        buf0[2] += bIn(i + hipThreadIdx_x + -1, j, k + -2) * c[1][0][0];
        buf0[2] += bIn(i + hipThreadIdx_x, j, k + -2) * c[2][0][0];
        buf0[2] += bIn(i + hipThreadIdx_x + 1, j, k + -2) * c[3][0][0];
        buf0[2] += bIn(i + hipThreadIdx_x + 2, j, k + -2) * c[4][0][0];
        buf0[1] += bIn(i + hipThreadIdx_x + -2, j, k + -2) * c[0][1][0];
        buf0[1] += bIn(i + hipThreadIdx_x + -1, j, k + -2) * c[1][1][0];
        buf0[1] += bIn(i + hipThreadIdx_x, j, k + -2) * c[2][1][0];
        buf0[1] += bIn(i + hipThreadIdx_x + 1, j, k + -2) * c[3][1][0];
        buf0[1] += bIn(i + hipThreadIdx_x + 2, j, k + -2) * c[4][1][0];
        buf0[0] += bIn(i + hipThreadIdx_x + -2, j, k + -2) * c[0][2][0];
        buf0[0] += bIn(i + hipThreadIdx_x + -1, j, k + -2) * c[1][2][0];
        buf0[0] += bIn(i + hipThreadIdx_x, j, k + -2) * c[2][2][0];
        buf0[0] += bIn(i + hipThreadIdx_x + 1, j, k + -2) * c[3][2][0];
        buf0[0] += bIn(i + hipThreadIdx_x + 2, j, k + -2) * c[4][2][0];
      }
      {
        buf0[3] += 0;
        buf0[3] += bIn(i + hipThreadIdx_x + -2, j + 1, k + -2) * c[0][0][0];
        buf0[3] += bIn(i + hipThreadIdx_x + -1, j + 1, k + -2) * c[1][0][0];
        buf0[3] += bIn(i + hipThreadIdx_x, j + 1, k + -2) * c[2][0][0];
        buf0[3] += bIn(i + hipThreadIdx_x + 1, j + 1, k + -2) * c[3][0][0];
        buf0[3] += bIn(i + hipThreadIdx_x + 2, j + 1, k + -2) * c[4][0][0];
        buf0[2] += bIn(i + hipThreadIdx_x + -2, j + 1, k + -2) * c[0][1][0];
        buf0[2] += bIn(i + hipThreadIdx_x + -1, j + 1, k + -2) * c[1][1][0];
        buf0[2] += bIn(i + hipThreadIdx_x, j + 1, k + -2) * c[2][1][0];
        buf0[2] += bIn(i + hipThreadIdx_x + 1, j + 1, k + -2) * c[3][1][0];
        buf0[2] += bIn(i + hipThreadIdx_x + 2, j + 1, k + -2) * c[4][1][0];
        buf0[1] += bIn(i + hipThreadIdx_x + -2, j + 1, k + -2) * c[0][2][0];
        buf0[1] += bIn(i + hipThreadIdx_x + -1, j + 1, k + -2) * c[1][2][0];
        buf0[1] += bIn(i + hipThreadIdx_x, j + 1, k + -2) * c[2][2][0];
        buf0[1] += bIn(i + hipThreadIdx_x + 1, j + 1, k + -2) * c[3][2][0];
        buf0[1] += bIn(i + hipThreadIdx_x + 2, j + 1, k + -2) * c[4][2][0];
        buf0[0] += bIn(i + hipThreadIdx_x + -2, j + 1, k + -2) * c[0][3][0];
        buf0[0] += bIn(i + hipThreadIdx_x + -1, j + 1, k + -2) * c[1][3][0];
        buf0[0] += bIn(i + hipThreadIdx_x, j + 1, k + -2) * c[2][3][0];
        buf0[0] += bIn(i + hipThreadIdx_x + 1, j + 1, k + -2) * c[3][3][0];
        buf0[0] += bIn(i + hipThreadIdx_x + 2, j + 1, k + -2) * c[4][3][0];
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 4; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[4 + rel] += 0;
            buf0[4 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + -2) * c[0][0][0];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + -2) * c[1][0][0];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + -2) * c[2][0][0];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + -2) * c[3][0][0];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + -2) * c[4][0][0];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + -2) * c[0][1][0];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + -2) * c[1][1][0];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + -2) * c[2][1][0];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + -2) * c[3][1][0];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + -2) * c[4][1][0];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + -2) * c[0][2][0];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + -2) * c[1][2][0];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + -2) * c[2][2][0];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + -2) * c[3][2][0];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + -2) * c[4][2][0];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + -2) * c[0][3][0];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + -2) * c[1][3][0];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + -2) * c[2][3][0];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + -2) * c[3][3][0];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + -2) * c[4][3][0];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + -2) * c[0][4][0];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + -2) * c[1][4][0];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + -2) * c[2][4][0];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + -2) * c[3][4][0];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + -2) * c[4][4][0];
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[7] += bIn(i + hipThreadIdx_x + -2, j + 6, k + -2) * c[0][1][0];
        buf0[7] += bIn(i + hipThreadIdx_x + -1, j + 6, k + -2) * c[1][1][0];
        buf0[7] += bIn(i + hipThreadIdx_x, j + 6, k + -2) * c[2][1][0];
        buf0[7] += bIn(i + hipThreadIdx_x + 1, j + 6, k + -2) * c[3][1][0];
        buf0[7] += bIn(i + hipThreadIdx_x + 2, j + 6, k + -2) * c[4][1][0];
        buf0[6] += bIn(i + hipThreadIdx_x + -2, j + 6, k + -2) * c[0][2][0];
        buf0[6] += bIn(i + hipThreadIdx_x + -1, j + 6, k + -2) * c[1][2][0];
        buf0[6] += bIn(i + hipThreadIdx_x, j + 6, k + -2) * c[2][2][0];
        buf0[6] += bIn(i + hipThreadIdx_x + 1, j + 6, k + -2) * c[3][2][0];
        buf0[6] += bIn(i + hipThreadIdx_x + 2, j + 6, k + -2) * c[4][2][0];
        buf0[5] += bIn(i + hipThreadIdx_x + -2, j + 6, k + -2) * c[0][3][0];
        buf0[5] += bIn(i + hipThreadIdx_x + -1, j + 6, k + -2) * c[1][3][0];
        buf0[5] += bIn(i + hipThreadIdx_x, j + 6, k + -2) * c[2][3][0];
        buf0[5] += bIn(i + hipThreadIdx_x + 1, j + 6, k + -2) * c[3][3][0];
        buf0[5] += bIn(i + hipThreadIdx_x + 2, j + 6, k + -2) * c[4][3][0];
        buf0[4] += bIn(i + hipThreadIdx_x + -2, j + 6, k + -2) * c[0][4][0];
        buf0[4] += bIn(i + hipThreadIdx_x + -1, j + 6, k + -2) * c[1][4][0];
        buf0[4] += bIn(i + hipThreadIdx_x, j + 6, k + -2) * c[2][4][0];
        buf0[4] += bIn(i + hipThreadIdx_x + 1, j + 6, k + -2) * c[3][4][0];
        buf0[4] += bIn(i + hipThreadIdx_x + 2, j + 6, k + -2) * c[4][4][0];
      }
      {
        buf0[7] += bIn(i + hipThreadIdx_x + -2, j + 7, k + -2) * c[0][2][0];
        buf0[7] += bIn(i + hipThreadIdx_x + -1, j + 7, k + -2) * c[1][2][0];
        buf0[7] += bIn(i + hipThreadIdx_x, j + 7, k + -2) * c[2][2][0];
        buf0[7] += bIn(i + hipThreadIdx_x + 1, j + 7, k + -2) * c[3][2][0];
        buf0[7] += bIn(i + hipThreadIdx_x + 2, j + 7, k + -2) * c[4][2][0];
        buf0[6] += bIn(i + hipThreadIdx_x + -2, j + 7, k + -2) * c[0][3][0];
        buf0[6] += bIn(i + hipThreadIdx_x + -1, j + 7, k + -2) * c[1][3][0];
        buf0[6] += bIn(i + hipThreadIdx_x, j + 7, k + -2) * c[2][3][0];
        buf0[6] += bIn(i + hipThreadIdx_x + 1, j + 7, k + -2) * c[3][3][0];
        buf0[6] += bIn(i + hipThreadIdx_x + 2, j + 7, k + -2) * c[4][3][0];
        buf0[5] += bIn(i + hipThreadIdx_x + -2, j + 7, k + -2) * c[0][4][0];
        buf0[5] += bIn(i + hipThreadIdx_x + -1, j + 7, k + -2) * c[1][4][0];
        buf0[5] += bIn(i + hipThreadIdx_x, j + 7, k + -2) * c[2][4][0];
        buf0[5] += bIn(i + hipThreadIdx_x + 1, j + 7, k + -2) * c[3][4][0];
        buf0[5] += bIn(i + hipThreadIdx_x + 2, j + 7, k + -2) * c[4][4][0];
      }
      {
        buf0[7] += bIn(i + hipThreadIdx_x + -2, j + 8, k + -2) * c[0][3][0];
        buf0[7] += bIn(i + hipThreadIdx_x + -1, j + 8, k + -2) * c[1][3][0];
        buf0[7] += bIn(i + hipThreadIdx_x, j + 8, k + -2) * c[2][3][0];
        buf0[7] += bIn(i + hipThreadIdx_x + 1, j + 8, k + -2) * c[3][3][0];
        buf0[7] += bIn(i + hipThreadIdx_x + 2, j + 8, k + -2) * c[4][3][0];
        buf0[6] += bIn(i + hipThreadIdx_x + -2, j + 8, k + -2) * c[0][4][0];
        buf0[6] += bIn(i + hipThreadIdx_x + -1, j + 8, k + -2) * c[1][4][0];
        buf0[6] += bIn(i + hipThreadIdx_x, j + 8, k + -2) * c[2][4][0];
        buf0[6] += bIn(i + hipThreadIdx_x + 1, j + 8, k + -2) * c[3][4][0];
        buf0[6] += bIn(i + hipThreadIdx_x + 2, j + 8, k + -2) * c[4][4][0];
      }
      {
        buf0[7] += bIn(i + hipThreadIdx_x + -2, j + 9, k + -2) * c[0][4][0];
        buf0[7] += bIn(i + hipThreadIdx_x + -1, j + 9, k + -2) * c[1][4][0];
        buf0[7] += bIn(i + hipThreadIdx_x, j + 9, k + -2) * c[2][4][0];
        buf0[7] += bIn(i + hipThreadIdx_x + 1, j + 9, k + -2) * c[3][4][0];
        buf0[7] += bIn(i + hipThreadIdx_x + 2, j + 9, k + -2) * c[4][4][0];
      }
      {
        buf0[8] += 0;
        buf0[8] += bIn(i + hipThreadIdx_x + -2, j + -2, k + -1) * c[0][0][0];
        buf0[8] += bIn(i + hipThreadIdx_x + -1, j + -2, k + -1) * c[1][0][0];
        buf0[8] += bIn(i + hipThreadIdx_x, j + -2, k + -1) * c[2][0][0];
        buf0[8] += bIn(i + hipThreadIdx_x + 1, j + -2, k + -1) * c[3][0][0];
        buf0[8] += bIn(i + hipThreadIdx_x + 2, j + -2, k + -1) * c[4][0][0];
        buf0[0] += bIn(i + hipThreadIdx_x + -2, j + -2, k + -1) * c[0][0][1];
        buf0[0] += bIn(i + hipThreadIdx_x + -1, j + -2, k + -1) * c[1][0][1];
        buf0[0] += bIn(i + hipThreadIdx_x, j + -2, k + -1) * c[2][0][1];
        buf0[0] += bIn(i + hipThreadIdx_x + 1, j + -2, k + -1) * c[3][0][1];
        buf0[0] += bIn(i + hipThreadIdx_x + 2, j + -2, k + -1) * c[4][0][1];
      }
      {
        buf0[9] += 0;
        buf0[9] += bIn(i + hipThreadIdx_x + -2, j + -1, k + -1) * c[0][0][0];
        buf0[9] += bIn(i + hipThreadIdx_x + -1, j + -1, k + -1) * c[1][0][0];
        buf0[9] += bIn(i + hipThreadIdx_x, j + -1, k + -1) * c[2][0][0];
        buf0[9] += bIn(i + hipThreadIdx_x + 1, j + -1, k + -1) * c[3][0][0];
        buf0[9] += bIn(i + hipThreadIdx_x + 2, j + -1, k + -1) * c[4][0][0];
        buf0[1] += bIn(i + hipThreadIdx_x + -2, j + -1, k + -1) * c[0][0][1];
        buf0[1] += bIn(i + hipThreadIdx_x + -1, j + -1, k + -1) * c[1][0][1];
        buf0[1] += bIn(i + hipThreadIdx_x, j + -1, k + -1) * c[2][0][1];
        buf0[1] += bIn(i + hipThreadIdx_x + 1, j + -1, k + -1) * c[3][0][1];
        buf0[1] += bIn(i + hipThreadIdx_x + 2, j + -1, k + -1) * c[4][0][1];
        buf0[8] += bIn(i + hipThreadIdx_x + -2, j + -1, k + -1) * c[0][1][0];
        buf0[8] += bIn(i + hipThreadIdx_x + -1, j + -1, k + -1) * c[1][1][0];
        buf0[8] += bIn(i + hipThreadIdx_x, j + -1, k + -1) * c[2][1][0];
        buf0[8] += bIn(i + hipThreadIdx_x + 1, j + -1, k + -1) * c[3][1][0];
        buf0[8] += bIn(i + hipThreadIdx_x + 2, j + -1, k + -1) * c[4][1][0];
        buf0[0] += bIn(i + hipThreadIdx_x + -2, j + -1, k + -1) * c[0][1][1];
        buf0[0] += bIn(i + hipThreadIdx_x + -1, j + -1, k + -1) * c[1][1][1];
        buf0[0] += bIn(i + hipThreadIdx_x, j + -1, k + -1) * c[2][1][1];
        buf0[0] += bIn(i + hipThreadIdx_x + 1, j + -1, k + -1) * c[3][1][1];
        buf0[0] += bIn(i + hipThreadIdx_x + 2, j + -1, k + -1) * c[4][1][1];
      }
      {
        buf0[10] += 0;
        buf0[10] += bIn(i + hipThreadIdx_x + -2, j, k + -1) * c[0][0][0];
        buf0[10] += bIn(i + hipThreadIdx_x + -1, j, k + -1) * c[1][0][0];
        buf0[10] += bIn(i + hipThreadIdx_x, j, k + -1) * c[2][0][0];
        buf0[10] += bIn(i + hipThreadIdx_x + 1, j, k + -1) * c[3][0][0];
        buf0[10] += bIn(i + hipThreadIdx_x + 2, j, k + -1) * c[4][0][0];
        buf0[2] += bIn(i + hipThreadIdx_x + -2, j, k + -1) * c[0][0][1];
        buf0[2] += bIn(i + hipThreadIdx_x + -1, j, k + -1) * c[1][0][1];
        buf0[2] += bIn(i + hipThreadIdx_x, j, k + -1) * c[2][0][1];
        buf0[2] += bIn(i + hipThreadIdx_x + 1, j, k + -1) * c[3][0][1];
        buf0[2] += bIn(i + hipThreadIdx_x + 2, j, k + -1) * c[4][0][1];
        buf0[9] += bIn(i + hipThreadIdx_x + -2, j, k + -1) * c[0][1][0];
        buf0[9] += bIn(i + hipThreadIdx_x + -1, j, k + -1) * c[1][1][0];
        buf0[9] += bIn(i + hipThreadIdx_x, j, k + -1) * c[2][1][0];
        buf0[9] += bIn(i + hipThreadIdx_x + 1, j, k + -1) * c[3][1][0];
        buf0[9] += bIn(i + hipThreadIdx_x + 2, j, k + -1) * c[4][1][0];
        buf0[1] += bIn(i + hipThreadIdx_x + -2, j, k + -1) * c[0][1][1];
        buf0[1] += bIn(i + hipThreadIdx_x + -1, j, k + -1) * c[1][1][1];
        buf0[1] += bIn(i + hipThreadIdx_x, j, k + -1) * c[2][1][1];
        buf0[1] += bIn(i + hipThreadIdx_x + 1, j, k + -1) * c[3][1][1];
        buf0[1] += bIn(i + hipThreadIdx_x + 2, j, k + -1) * c[4][1][1];
        buf0[8] += bIn(i + hipThreadIdx_x + -2, j, k + -1) * c[0][2][0];
        buf0[8] += bIn(i + hipThreadIdx_x + -1, j, k + -1) * c[1][2][0];
        buf0[8] += bIn(i + hipThreadIdx_x, j, k + -1) * c[2][2][0];
        buf0[8] += bIn(i + hipThreadIdx_x + 1, j, k + -1) * c[3][2][0];
        buf0[8] += bIn(i + hipThreadIdx_x + 2, j, k + -1) * c[4][2][0];
        buf0[0] += bIn(i + hipThreadIdx_x + -2, j, k + -1) * c[0][2][1];
        buf0[0] += bIn(i + hipThreadIdx_x + -1, j, k + -1) * c[1][2][1];
        buf0[0] += bIn(i + hipThreadIdx_x, j, k + -1) * c[2][2][1];
        buf0[0] += bIn(i + hipThreadIdx_x + 1, j, k + -1) * c[3][2][1];
        buf0[0] += bIn(i + hipThreadIdx_x + 2, j, k + -1) * c[4][2][1];
      }
      {
        buf0[11] += 0;
        buf0[11] += bIn(i + hipThreadIdx_x + -2, j + 1, k + -1) * c[0][0][0];
        buf0[11] += bIn(i + hipThreadIdx_x + -1, j + 1, k + -1) * c[1][0][0];
        buf0[11] += bIn(i + hipThreadIdx_x, j + 1, k + -1) * c[2][0][0];
        buf0[11] += bIn(i + hipThreadIdx_x + 1, j + 1, k + -1) * c[3][0][0];
        buf0[11] += bIn(i + hipThreadIdx_x + 2, j + 1, k + -1) * c[4][0][0];
        buf0[3] += bIn(i + hipThreadIdx_x + -2, j + 1, k + -1) * c[0][0][1];
        buf0[3] += bIn(i + hipThreadIdx_x + -1, j + 1, k + -1) * c[1][0][1];
        buf0[3] += bIn(i + hipThreadIdx_x, j + 1, k + -1) * c[2][0][1];
        buf0[3] += bIn(i + hipThreadIdx_x + 1, j + 1, k + -1) * c[3][0][1];
        buf0[3] += bIn(i + hipThreadIdx_x + 2, j + 1, k + -1) * c[4][0][1];
        buf0[10] += bIn(i + hipThreadIdx_x + -2, j + 1, k + -1) * c[0][1][0];
        buf0[10] += bIn(i + hipThreadIdx_x + -1, j + 1, k + -1) * c[1][1][0];
        buf0[10] += bIn(i + hipThreadIdx_x, j + 1, k + -1) * c[2][1][0];
        buf0[10] += bIn(i + hipThreadIdx_x + 1, j + 1, k + -1) * c[3][1][0];
        buf0[10] += bIn(i + hipThreadIdx_x + 2, j + 1, k + -1) * c[4][1][0];
        buf0[2] += bIn(i + hipThreadIdx_x + -2, j + 1, k + -1) * c[0][1][1];
        buf0[2] += bIn(i + hipThreadIdx_x + -1, j + 1, k + -1) * c[1][1][1];
        buf0[2] += bIn(i + hipThreadIdx_x, j + 1, k + -1) * c[2][1][1];
        buf0[2] += bIn(i + hipThreadIdx_x + 1, j + 1, k + -1) * c[3][1][1];
        buf0[2] += bIn(i + hipThreadIdx_x + 2, j + 1, k + -1) * c[4][1][1];
        buf0[9] += bIn(i + hipThreadIdx_x + -2, j + 1, k + -1) * c[0][2][0];
        buf0[9] += bIn(i + hipThreadIdx_x + -1, j + 1, k + -1) * c[1][2][0];
        buf0[9] += bIn(i + hipThreadIdx_x, j + 1, k + -1) * c[2][2][0];
        buf0[9] += bIn(i + hipThreadIdx_x + 1, j + 1, k + -1) * c[3][2][0];
        buf0[9] += bIn(i + hipThreadIdx_x + 2, j + 1, k + -1) * c[4][2][0];
        buf0[1] += bIn(i + hipThreadIdx_x + -2, j + 1, k + -1) * c[0][2][1];
        buf0[1] += bIn(i + hipThreadIdx_x + -1, j + 1, k + -1) * c[1][2][1];
        buf0[1] += bIn(i + hipThreadIdx_x, j + 1, k + -1) * c[2][2][1];
        buf0[1] += bIn(i + hipThreadIdx_x + 1, j + 1, k + -1) * c[3][2][1];
        buf0[1] += bIn(i + hipThreadIdx_x + 2, j + 1, k + -1) * c[4][2][1];
        buf0[8] += bIn(i + hipThreadIdx_x + -2, j + 1, k + -1) * c[0][3][0];
        buf0[8] += bIn(i + hipThreadIdx_x + -1, j + 1, k + -1) * c[1][3][0];
        buf0[8] += bIn(i + hipThreadIdx_x, j + 1, k + -1) * c[2][3][0];
        buf0[8] += bIn(i + hipThreadIdx_x + 1, j + 1, k + -1) * c[3][3][0];
        buf0[8] += bIn(i + hipThreadIdx_x + 2, j + 1, k + -1) * c[4][3][0];
        buf0[0] += bIn(i + hipThreadIdx_x + -2, j + 1, k + -1) * c[0][3][1];
        buf0[0] += bIn(i + hipThreadIdx_x + -1, j + 1, k + -1) * c[1][3][1];
        buf0[0] += bIn(i + hipThreadIdx_x, j + 1, k + -1) * c[2][3][1];
        buf0[0] += bIn(i + hipThreadIdx_x + 1, j + 1, k + -1) * c[3][3][1];
        buf0[0] += bIn(i + hipThreadIdx_x + 2, j + 1, k + -1) * c[4][3][1];
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 4; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[12 + rel] += 0;
            buf0[12 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + -1) * c[0][0][0];
            buf0[12 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + -1) * c[1][0][0];
            buf0[12 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + -1) * c[2][0][0];
            buf0[12 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + -1) * c[3][0][0];
            buf0[12 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + -1) * c[4][0][0];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + -1) * c[0][0][1];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + -1) * c[1][0][1];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + -1) * c[2][0][1];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + -1) * c[3][0][1];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + -1) * c[4][0][1];
            buf0[11 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + -1) * c[0][1][0];
            buf0[11 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + -1) * c[1][1][0];
            buf0[11 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + -1) * c[2][1][0];
            buf0[11 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + -1) * c[3][1][0];
            buf0[11 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + -1) * c[4][1][0];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + -1) * c[0][1][1];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + -1) * c[1][1][1];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + -1) * c[2][1][1];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + -1) * c[3][1][1];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + -1) * c[4][1][1];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + -1) * c[0][2][0];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + -1) * c[1][2][0];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + -1) * c[2][2][0];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + -1) * c[3][2][0];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + -1) * c[4][2][0];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + -1) * c[0][2][1];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + -1) * c[1][2][1];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + -1) * c[2][2][1];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + -1) * c[3][2][1];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + -1) * c[4][2][1];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + -1) * c[0][3][0];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + -1) * c[1][3][0];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + -1) * c[2][3][0];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + -1) * c[3][3][0];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + -1) * c[4][3][0];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + -1) * c[0][3][1];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + -1) * c[1][3][1];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + -1) * c[2][3][1];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + -1) * c[3][3][1];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + -1) * c[4][3][1];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + -1) * c[0][4][0];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + -1) * c[1][4][0];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + -1) * c[2][4][0];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + -1) * c[3][4][0];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + -1) * c[4][4][0];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + -1) * c[0][4][1];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + -1) * c[1][4][1];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + -1) * c[2][4][1];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + -1) * c[3][4][1];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + -1) * c[4][4][1];
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[15] += bIn(i + hipThreadIdx_x + -2, j + 6, k + -1) * c[0][1][0];
        buf0[15] += bIn(i + hipThreadIdx_x + -1, j + 6, k + -1) * c[1][1][0];
        buf0[15] += bIn(i + hipThreadIdx_x, j + 6, k + -1) * c[2][1][0];
        buf0[15] += bIn(i + hipThreadIdx_x + 1, j + 6, k + -1) * c[3][1][0];
        buf0[15] += bIn(i + hipThreadIdx_x + 2, j + 6, k + -1) * c[4][1][0];
        buf0[7] += bIn(i + hipThreadIdx_x + -2, j + 6, k + -1) * c[0][1][1];
        buf0[7] += bIn(i + hipThreadIdx_x + -1, j + 6, k + -1) * c[1][1][1];
        buf0[7] += bIn(i + hipThreadIdx_x, j + 6, k + -1) * c[2][1][1];
        buf0[7] += bIn(i + hipThreadIdx_x + 1, j + 6, k + -1) * c[3][1][1];
        buf0[7] += bIn(i + hipThreadIdx_x + 2, j + 6, k + -1) * c[4][1][1];
        buf0[14] += bIn(i + hipThreadIdx_x + -2, j + 6, k + -1) * c[0][2][0];
        buf0[14] += bIn(i + hipThreadIdx_x + -1, j + 6, k + -1) * c[1][2][0];
        buf0[14] += bIn(i + hipThreadIdx_x, j + 6, k + -1) * c[2][2][0];
        buf0[14] += bIn(i + hipThreadIdx_x + 1, j + 6, k + -1) * c[3][2][0];
        buf0[14] += bIn(i + hipThreadIdx_x + 2, j + 6, k + -1) * c[4][2][0];
        buf0[6] += bIn(i + hipThreadIdx_x + -2, j + 6, k + -1) * c[0][2][1];
        buf0[6] += bIn(i + hipThreadIdx_x + -1, j + 6, k + -1) * c[1][2][1];
        buf0[6] += bIn(i + hipThreadIdx_x, j + 6, k + -1) * c[2][2][1];
        buf0[6] += bIn(i + hipThreadIdx_x + 1, j + 6, k + -1) * c[3][2][1];
        buf0[6] += bIn(i + hipThreadIdx_x + 2, j + 6, k + -1) * c[4][2][1];
        buf0[13] += bIn(i + hipThreadIdx_x + -2, j + 6, k + -1) * c[0][3][0];
        buf0[13] += bIn(i + hipThreadIdx_x + -1, j + 6, k + -1) * c[1][3][0];
        buf0[13] += bIn(i + hipThreadIdx_x, j + 6, k + -1) * c[2][3][0];
        buf0[13] += bIn(i + hipThreadIdx_x + 1, j + 6, k + -1) * c[3][3][0];
        buf0[13] += bIn(i + hipThreadIdx_x + 2, j + 6, k + -1) * c[4][3][0];
        buf0[5] += bIn(i + hipThreadIdx_x + -2, j + 6, k + -1) * c[0][3][1];
        buf0[5] += bIn(i + hipThreadIdx_x + -1, j + 6, k + -1) * c[1][3][1];
        buf0[5] += bIn(i + hipThreadIdx_x, j + 6, k + -1) * c[2][3][1];
        buf0[5] += bIn(i + hipThreadIdx_x + 1, j + 6, k + -1) * c[3][3][1];
        buf0[5] += bIn(i + hipThreadIdx_x + 2, j + 6, k + -1) * c[4][3][1];
        buf0[12] += bIn(i + hipThreadIdx_x + -2, j + 6, k + -1) * c[0][4][0];
        buf0[12] += bIn(i + hipThreadIdx_x + -1, j + 6, k + -1) * c[1][4][0];
        buf0[12] += bIn(i + hipThreadIdx_x, j + 6, k + -1) * c[2][4][0];
        buf0[12] += bIn(i + hipThreadIdx_x + 1, j + 6, k + -1) * c[3][4][0];
        buf0[12] += bIn(i + hipThreadIdx_x + 2, j + 6, k + -1) * c[4][4][0];
        buf0[4] += bIn(i + hipThreadIdx_x + -2, j + 6, k + -1) * c[0][4][1];
        buf0[4] += bIn(i + hipThreadIdx_x + -1, j + 6, k + -1) * c[1][4][1];
        buf0[4] += bIn(i + hipThreadIdx_x, j + 6, k + -1) * c[2][4][1];
        buf0[4] += bIn(i + hipThreadIdx_x + 1, j + 6, k + -1) * c[3][4][1];
        buf0[4] += bIn(i + hipThreadIdx_x + 2, j + 6, k + -1) * c[4][4][1];
      }
      {
        buf0[15] += bIn(i + hipThreadIdx_x + -2, j + 7, k + -1) * c[0][2][0];
        buf0[15] += bIn(i + hipThreadIdx_x + -1, j + 7, k + -1) * c[1][2][0];
        buf0[15] += bIn(i + hipThreadIdx_x, j + 7, k + -1) * c[2][2][0];
        buf0[15] += bIn(i + hipThreadIdx_x + 1, j + 7, k + -1) * c[3][2][0];
        buf0[15] += bIn(i + hipThreadIdx_x + 2, j + 7, k + -1) * c[4][2][0];
        buf0[7] += bIn(i + hipThreadIdx_x + -2, j + 7, k + -1) * c[0][2][1];
        buf0[7] += bIn(i + hipThreadIdx_x + -1, j + 7, k + -1) * c[1][2][1];
        buf0[7] += bIn(i + hipThreadIdx_x, j + 7, k + -1) * c[2][2][1];
        buf0[7] += bIn(i + hipThreadIdx_x + 1, j + 7, k + -1) * c[3][2][1];
        buf0[7] += bIn(i + hipThreadIdx_x + 2, j + 7, k + -1) * c[4][2][1];
        buf0[14] += bIn(i + hipThreadIdx_x + -2, j + 7, k + -1) * c[0][3][0];
        buf0[14] += bIn(i + hipThreadIdx_x + -1, j + 7, k + -1) * c[1][3][0];
        buf0[14] += bIn(i + hipThreadIdx_x, j + 7, k + -1) * c[2][3][0];
        buf0[14] += bIn(i + hipThreadIdx_x + 1, j + 7, k + -1) * c[3][3][0];
        buf0[14] += bIn(i + hipThreadIdx_x + 2, j + 7, k + -1) * c[4][3][0];
        buf0[6] += bIn(i + hipThreadIdx_x + -2, j + 7, k + -1) * c[0][3][1];
        buf0[6] += bIn(i + hipThreadIdx_x + -1, j + 7, k + -1) * c[1][3][1];
        buf0[6] += bIn(i + hipThreadIdx_x, j + 7, k + -1) * c[2][3][1];
        buf0[6] += bIn(i + hipThreadIdx_x + 1, j + 7, k + -1) * c[3][3][1];
        buf0[6] += bIn(i + hipThreadIdx_x + 2, j + 7, k + -1) * c[4][3][1];
        buf0[13] += bIn(i + hipThreadIdx_x + -2, j + 7, k + -1) * c[0][4][0];
        buf0[13] += bIn(i + hipThreadIdx_x + -1, j + 7, k + -1) * c[1][4][0];
        buf0[13] += bIn(i + hipThreadIdx_x, j + 7, k + -1) * c[2][4][0];
        buf0[13] += bIn(i + hipThreadIdx_x + 1, j + 7, k + -1) * c[3][4][0];
        buf0[13] += bIn(i + hipThreadIdx_x + 2, j + 7, k + -1) * c[4][4][0];
        buf0[5] += bIn(i + hipThreadIdx_x + -2, j + 7, k + -1) * c[0][4][1];
        buf0[5] += bIn(i + hipThreadIdx_x + -1, j + 7, k + -1) * c[1][4][1];
        buf0[5] += bIn(i + hipThreadIdx_x, j + 7, k + -1) * c[2][4][1];
        buf0[5] += bIn(i + hipThreadIdx_x + 1, j + 7, k + -1) * c[3][4][1];
        buf0[5] += bIn(i + hipThreadIdx_x + 2, j + 7, k + -1) * c[4][4][1];
      }
      {
        buf0[15] += bIn(i + hipThreadIdx_x + -2, j + 8, k + -1) * c[0][3][0];
        buf0[15] += bIn(i + hipThreadIdx_x + -1, j + 8, k + -1) * c[1][3][0];
        buf0[15] += bIn(i + hipThreadIdx_x, j + 8, k + -1) * c[2][3][0];
        buf0[15] += bIn(i + hipThreadIdx_x + 1, j + 8, k + -1) * c[3][3][0];
        buf0[15] += bIn(i + hipThreadIdx_x + 2, j + 8, k + -1) * c[4][3][0];
        buf0[7] += bIn(i + hipThreadIdx_x + -2, j + 8, k + -1) * c[0][3][1];
        buf0[7] += bIn(i + hipThreadIdx_x + -1, j + 8, k + -1) * c[1][3][1];
        buf0[7] += bIn(i + hipThreadIdx_x, j + 8, k + -1) * c[2][3][1];
        buf0[7] += bIn(i + hipThreadIdx_x + 1, j + 8, k + -1) * c[3][3][1];
        buf0[7] += bIn(i + hipThreadIdx_x + 2, j + 8, k + -1) * c[4][3][1];
        buf0[14] += bIn(i + hipThreadIdx_x + -2, j + 8, k + -1) * c[0][4][0];
        buf0[14] += bIn(i + hipThreadIdx_x + -1, j + 8, k + -1) * c[1][4][0];
        buf0[14] += bIn(i + hipThreadIdx_x, j + 8, k + -1) * c[2][4][0];
        buf0[14] += bIn(i + hipThreadIdx_x + 1, j + 8, k + -1) * c[3][4][0];
        buf0[14] += bIn(i + hipThreadIdx_x + 2, j + 8, k + -1) * c[4][4][0];
        buf0[6] += bIn(i + hipThreadIdx_x + -2, j + 8, k + -1) * c[0][4][1];
        buf0[6] += bIn(i + hipThreadIdx_x + -1, j + 8, k + -1) * c[1][4][1];
        buf0[6] += bIn(i + hipThreadIdx_x, j + 8, k + -1) * c[2][4][1];
        buf0[6] += bIn(i + hipThreadIdx_x + 1, j + 8, k + -1) * c[3][4][1];
        buf0[6] += bIn(i + hipThreadIdx_x + 2, j + 8, k + -1) * c[4][4][1];
      }
      {
        buf0[15] += bIn(i + hipThreadIdx_x + -2, j + 9, k + -1) * c[0][4][0];
        buf0[15] += bIn(i + hipThreadIdx_x + -1, j + 9, k + -1) * c[1][4][0];
        buf0[15] += bIn(i + hipThreadIdx_x, j + 9, k + -1) * c[2][4][0];
        buf0[15] += bIn(i + hipThreadIdx_x + 1, j + 9, k + -1) * c[3][4][0];
        buf0[15] += bIn(i + hipThreadIdx_x + 2, j + 9, k + -1) * c[4][4][0];
        buf0[7] += bIn(i + hipThreadIdx_x + -2, j + 9, k + -1) * c[0][4][1];
        buf0[7] += bIn(i + hipThreadIdx_x + -1, j + 9, k + -1) * c[1][4][1];
        buf0[7] += bIn(i + hipThreadIdx_x, j + 9, k + -1) * c[2][4][1];
        buf0[7] += bIn(i + hipThreadIdx_x + 1, j + 9, k + -1) * c[3][4][1];
        buf0[7] += bIn(i + hipThreadIdx_x + 2, j + 9, k + -1) * c[4][4][1];
      }
      {
        buf0[16] += 0;
        buf0[16] += bIn(i + hipThreadIdx_x + -2, j + -2, k) * c[0][0][0];
        buf0[16] += bIn(i + hipThreadIdx_x + -1, j + -2, k) * c[1][0][0];
        buf0[16] += bIn(i + hipThreadIdx_x, j + -2, k) * c[2][0][0];
        buf0[16] += bIn(i + hipThreadIdx_x + 1, j + -2, k) * c[3][0][0];
        buf0[16] += bIn(i + hipThreadIdx_x + 2, j + -2, k) * c[4][0][0];
        buf0[8] += bIn(i + hipThreadIdx_x + -2, j + -2, k) * c[0][0][1];
        buf0[8] += bIn(i + hipThreadIdx_x + -1, j + -2, k) * c[1][0][1];
        buf0[8] += bIn(i + hipThreadIdx_x, j + -2, k) * c[2][0][1];
        buf0[8] += bIn(i + hipThreadIdx_x + 1, j + -2, k) * c[3][0][1];
        buf0[8] += bIn(i + hipThreadIdx_x + 2, j + -2, k) * c[4][0][1];
        buf0[0] += bIn(i + hipThreadIdx_x + -2, j + -2, k) * c[0][0][2];
        buf0[0] += bIn(i + hipThreadIdx_x + -1, j + -2, k) * c[1][0][2];
        buf0[0] += bIn(i + hipThreadIdx_x, j + -2, k) * c[2][0][2];
        buf0[0] += bIn(i + hipThreadIdx_x + 1, j + -2, k) * c[3][0][2];
        buf0[0] += bIn(i + hipThreadIdx_x + 2, j + -2, k) * c[4][0][2];
      }
      {
        buf0[17] += 0;
        buf0[17] += bIn(i + hipThreadIdx_x + -2, j + -1, k) * c[0][0][0];
        buf0[17] += bIn(i + hipThreadIdx_x + -1, j + -1, k) * c[1][0][0];
        buf0[17] += bIn(i + hipThreadIdx_x, j + -1, k) * c[2][0][0];
        buf0[17] += bIn(i + hipThreadIdx_x + 1, j + -1, k) * c[3][0][0];
        buf0[17] += bIn(i + hipThreadIdx_x + 2, j + -1, k) * c[4][0][0];
        buf0[9] += bIn(i + hipThreadIdx_x + -2, j + -1, k) * c[0][0][1];
        buf0[9] += bIn(i + hipThreadIdx_x + -1, j + -1, k) * c[1][0][1];
        buf0[9] += bIn(i + hipThreadIdx_x, j + -1, k) * c[2][0][1];
        buf0[9] += bIn(i + hipThreadIdx_x + 1, j + -1, k) * c[3][0][1];
        buf0[9] += bIn(i + hipThreadIdx_x + 2, j + -1, k) * c[4][0][1];
        buf0[1] += bIn(i + hipThreadIdx_x + -2, j + -1, k) * c[0][0][2];
        buf0[1] += bIn(i + hipThreadIdx_x + -1, j + -1, k) * c[1][0][2];
        buf0[1] += bIn(i + hipThreadIdx_x, j + -1, k) * c[2][0][2];
        buf0[1] += bIn(i + hipThreadIdx_x + 1, j + -1, k) * c[3][0][2];
        buf0[1] += bIn(i + hipThreadIdx_x + 2, j + -1, k) * c[4][0][2];
        buf0[16] += bIn(i + hipThreadIdx_x + -2, j + -1, k) * c[0][1][0];
        buf0[16] += bIn(i + hipThreadIdx_x + -1, j + -1, k) * c[1][1][0];
        buf0[16] += bIn(i + hipThreadIdx_x, j + -1, k) * c[2][1][0];
        buf0[16] += bIn(i + hipThreadIdx_x + 1, j + -1, k) * c[3][1][0];
        buf0[16] += bIn(i + hipThreadIdx_x + 2, j + -1, k) * c[4][1][0];
        buf0[8] += bIn(i + hipThreadIdx_x + -2, j + -1, k) * c[0][1][1];
        buf0[8] += bIn(i + hipThreadIdx_x + -1, j + -1, k) * c[1][1][1];
        buf0[8] += bIn(i + hipThreadIdx_x, j + -1, k) * c[2][1][1];
        buf0[8] += bIn(i + hipThreadIdx_x + 1, j + -1, k) * c[3][1][1];
        buf0[8] += bIn(i + hipThreadIdx_x + 2, j + -1, k) * c[4][1][1];
        buf0[0] += bIn(i + hipThreadIdx_x + -2, j + -1, k) * c[0][1][2];
        buf0[0] += bIn(i + hipThreadIdx_x + -1, j + -1, k) * c[1][1][2];
        buf0[0] += bIn(i + hipThreadIdx_x, j + -1, k) * c[2][1][2];
        buf0[0] += bIn(i + hipThreadIdx_x + 1, j + -1, k) * c[3][1][2];
        buf0[0] += bIn(i + hipThreadIdx_x + 2, j + -1, k) * c[4][1][2];
      }
      {
        buf0[18] += 0;
        buf0[18] += bIn(i + hipThreadIdx_x + -2, j, k) * c[0][0][0];
        buf0[18] += bIn(i + hipThreadIdx_x + -1, j, k) * c[1][0][0];
        buf0[18] += bIn(i + hipThreadIdx_x, j, k) * c[2][0][0];
        buf0[18] += bIn(i + hipThreadIdx_x + 1, j, k) * c[3][0][0];
        buf0[18] += bIn(i + hipThreadIdx_x + 2, j, k) * c[4][0][0];
        buf0[10] += bIn(i + hipThreadIdx_x + -2, j, k) * c[0][0][1];
        buf0[10] += bIn(i + hipThreadIdx_x + -1, j, k) * c[1][0][1];
        buf0[10] += bIn(i + hipThreadIdx_x, j, k) * c[2][0][1];
        buf0[10] += bIn(i + hipThreadIdx_x + 1, j, k) * c[3][0][1];
        buf0[10] += bIn(i + hipThreadIdx_x + 2, j, k) * c[4][0][1];
        buf0[2] += bIn(i + hipThreadIdx_x + -2, j, k) * c[0][0][2];
        buf0[2] += bIn(i + hipThreadIdx_x + -1, j, k) * c[1][0][2];
        buf0[2] += bIn(i + hipThreadIdx_x, j, k) * c[2][0][2];
        buf0[2] += bIn(i + hipThreadIdx_x + 1, j, k) * c[3][0][2];
        buf0[2] += bIn(i + hipThreadIdx_x + 2, j, k) * c[4][0][2];
        buf0[17] += bIn(i + hipThreadIdx_x + -2, j, k) * c[0][1][0];
        buf0[17] += bIn(i + hipThreadIdx_x + -1, j, k) * c[1][1][0];
        buf0[17] += bIn(i + hipThreadIdx_x, j, k) * c[2][1][0];
        buf0[17] += bIn(i + hipThreadIdx_x + 1, j, k) * c[3][1][0];
        buf0[17] += bIn(i + hipThreadIdx_x + 2, j, k) * c[4][1][0];
        buf0[9] += bIn(i + hipThreadIdx_x + -2, j, k) * c[0][1][1];
        buf0[9] += bIn(i + hipThreadIdx_x + -1, j, k) * c[1][1][1];
        buf0[9] += bIn(i + hipThreadIdx_x, j, k) * c[2][1][1];
        buf0[9] += bIn(i + hipThreadIdx_x + 1, j, k) * c[3][1][1];
        buf0[9] += bIn(i + hipThreadIdx_x + 2, j, k) * c[4][1][1];
        buf0[1] += bIn(i + hipThreadIdx_x + -2, j, k) * c[0][1][2];
        buf0[1] += bIn(i + hipThreadIdx_x + -1, j, k) * c[1][1][2];
        buf0[1] += bIn(i + hipThreadIdx_x, j, k) * c[2][1][2];
        buf0[1] += bIn(i + hipThreadIdx_x + 1, j, k) * c[3][1][2];
        buf0[1] += bIn(i + hipThreadIdx_x + 2, j, k) * c[4][1][2];
        buf0[16] += bIn(i + hipThreadIdx_x + -2, j, k) * c[0][2][0];
        buf0[16] += bIn(i + hipThreadIdx_x + -1, j, k) * c[1][2][0];
        buf0[16] += bIn(i + hipThreadIdx_x, j, k) * c[2][2][0];
        buf0[16] += bIn(i + hipThreadIdx_x + 1, j, k) * c[3][2][0];
        buf0[16] += bIn(i + hipThreadIdx_x + 2, j, k) * c[4][2][0];
        buf0[8] += bIn(i + hipThreadIdx_x + -2, j, k) * c[0][2][1];
        buf0[8] += bIn(i + hipThreadIdx_x + -1, j, k) * c[1][2][1];
        buf0[8] += bIn(i + hipThreadIdx_x, j, k) * c[2][2][1];
        buf0[8] += bIn(i + hipThreadIdx_x + 1, j, k) * c[3][2][1];
        buf0[8] += bIn(i + hipThreadIdx_x + 2, j, k) * c[4][2][1];
        buf0[0] += bIn(i + hipThreadIdx_x + -2, j, k) * c[0][2][2];
        buf0[0] += bIn(i + hipThreadIdx_x + -1, j, k) * c[1][2][2];
        buf0[0] += bIn(i + hipThreadIdx_x, j, k) * c[2][2][2];
        buf0[0] += bIn(i + hipThreadIdx_x + 1, j, k) * c[3][2][2];
        buf0[0] += bIn(i + hipThreadIdx_x + 2, j, k) * c[4][2][2];
      }
      {
        buf0[19] += 0;
        buf0[19] += bIn(i + hipThreadIdx_x + -2, j + 1, k) * c[0][0][0];
        buf0[19] += bIn(i + hipThreadIdx_x + -1, j + 1, k) * c[1][0][0];
        buf0[19] += bIn(i + hipThreadIdx_x, j + 1, k) * c[2][0][0];
        buf0[19] += bIn(i + hipThreadIdx_x + 1, j + 1, k) * c[3][0][0];
        buf0[19] += bIn(i + hipThreadIdx_x + 2, j + 1, k) * c[4][0][0];
        buf0[11] += bIn(i + hipThreadIdx_x + -2, j + 1, k) * c[0][0][1];
        buf0[11] += bIn(i + hipThreadIdx_x + -1, j + 1, k) * c[1][0][1];
        buf0[11] += bIn(i + hipThreadIdx_x, j + 1, k) * c[2][0][1];
        buf0[11] += bIn(i + hipThreadIdx_x + 1, j + 1, k) * c[3][0][1];
        buf0[11] += bIn(i + hipThreadIdx_x + 2, j + 1, k) * c[4][0][1];
        buf0[3] += bIn(i + hipThreadIdx_x + -2, j + 1, k) * c[0][0][2];
        buf0[3] += bIn(i + hipThreadIdx_x + -1, j + 1, k) * c[1][0][2];
        buf0[3] += bIn(i + hipThreadIdx_x, j + 1, k) * c[2][0][2];
        buf0[3] += bIn(i + hipThreadIdx_x + 1, j + 1, k) * c[3][0][2];
        buf0[3] += bIn(i + hipThreadIdx_x + 2, j + 1, k) * c[4][0][2];
        buf0[18] += bIn(i + hipThreadIdx_x + -2, j + 1, k) * c[0][1][0];
        buf0[18] += bIn(i + hipThreadIdx_x + -1, j + 1, k) * c[1][1][0];
        buf0[18] += bIn(i + hipThreadIdx_x, j + 1, k) * c[2][1][0];
        buf0[18] += bIn(i + hipThreadIdx_x + 1, j + 1, k) * c[3][1][0];
        buf0[18] += bIn(i + hipThreadIdx_x + 2, j + 1, k) * c[4][1][0];
        buf0[10] += bIn(i + hipThreadIdx_x + -2, j + 1, k) * c[0][1][1];
        buf0[10] += bIn(i + hipThreadIdx_x + -1, j + 1, k) * c[1][1][1];
        buf0[10] += bIn(i + hipThreadIdx_x, j + 1, k) * c[2][1][1];
        buf0[10] += bIn(i + hipThreadIdx_x + 1, j + 1, k) * c[3][1][1];
        buf0[10] += bIn(i + hipThreadIdx_x + 2, j + 1, k) * c[4][1][1];
        buf0[2] += bIn(i + hipThreadIdx_x + -2, j + 1, k) * c[0][1][2];
        buf0[2] += bIn(i + hipThreadIdx_x + -1, j + 1, k) * c[1][1][2];
        buf0[2] += bIn(i + hipThreadIdx_x, j + 1, k) * c[2][1][2];
        buf0[2] += bIn(i + hipThreadIdx_x + 1, j + 1, k) * c[3][1][2];
        buf0[2] += bIn(i + hipThreadIdx_x + 2, j + 1, k) * c[4][1][2];
        buf0[17] += bIn(i + hipThreadIdx_x + -2, j + 1, k) * c[0][2][0];
        buf0[17] += bIn(i + hipThreadIdx_x + -1, j + 1, k) * c[1][2][0];
        buf0[17] += bIn(i + hipThreadIdx_x, j + 1, k) * c[2][2][0];
        buf0[17] += bIn(i + hipThreadIdx_x + 1, j + 1, k) * c[3][2][0];
        buf0[17] += bIn(i + hipThreadIdx_x + 2, j + 1, k) * c[4][2][0];
        buf0[9] += bIn(i + hipThreadIdx_x + -2, j + 1, k) * c[0][2][1];
        buf0[9] += bIn(i + hipThreadIdx_x + -1, j + 1, k) * c[1][2][1];
        buf0[9] += bIn(i + hipThreadIdx_x, j + 1, k) * c[2][2][1];
        buf0[9] += bIn(i + hipThreadIdx_x + 1, j + 1, k) * c[3][2][1];
        buf0[9] += bIn(i + hipThreadIdx_x + 2, j + 1, k) * c[4][2][1];
        buf0[1] += bIn(i + hipThreadIdx_x + -2, j + 1, k) * c[0][2][2];
        buf0[1] += bIn(i + hipThreadIdx_x + -1, j + 1, k) * c[1][2][2];
        buf0[1] += bIn(i + hipThreadIdx_x, j + 1, k) * c[2][2][2];
        buf0[1] += bIn(i + hipThreadIdx_x + 1, j + 1, k) * c[3][2][2];
        buf0[1] += bIn(i + hipThreadIdx_x + 2, j + 1, k) * c[4][2][2];
        buf0[16] += bIn(i + hipThreadIdx_x + -2, j + 1, k) * c[0][3][0];
        buf0[16] += bIn(i + hipThreadIdx_x + -1, j + 1, k) * c[1][3][0];
        buf0[16] += bIn(i + hipThreadIdx_x, j + 1, k) * c[2][3][0];
        buf0[16] += bIn(i + hipThreadIdx_x + 1, j + 1, k) * c[3][3][0];
        buf0[16] += bIn(i + hipThreadIdx_x + 2, j + 1, k) * c[4][3][0];
        buf0[8] += bIn(i + hipThreadIdx_x + -2, j + 1, k) * c[0][3][1];
        buf0[8] += bIn(i + hipThreadIdx_x + -1, j + 1, k) * c[1][3][1];
        buf0[8] += bIn(i + hipThreadIdx_x, j + 1, k) * c[2][3][1];
        buf0[8] += bIn(i + hipThreadIdx_x + 1, j + 1, k) * c[3][3][1];
        buf0[8] += bIn(i + hipThreadIdx_x + 2, j + 1, k) * c[4][3][1];
        buf0[0] += bIn(i + hipThreadIdx_x + -2, j + 1, k) * c[0][3][2];
        buf0[0] += bIn(i + hipThreadIdx_x + -1, j + 1, k) * c[1][3][2];
        buf0[0] += bIn(i + hipThreadIdx_x, j + 1, k) * c[2][3][2];
        buf0[0] += bIn(i + hipThreadIdx_x + 1, j + 1, k) * c[3][3][2];
        buf0[0] += bIn(i + hipThreadIdx_x + 2, j + 1, k) * c[4][3][2];
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 4; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[20 + rel] += 0;
            buf0[20 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k) * c[0][0][0];
            buf0[20 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k) * c[1][0][0];
            buf0[20 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) * c[2][0][0];
            buf0[20 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k) * c[3][0][0];
            buf0[20 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k) * c[4][0][0];
            buf0[12 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k) * c[0][0][1];
            buf0[12 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k) * c[1][0][1];
            buf0[12 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) * c[2][0][1];
            buf0[12 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k) * c[3][0][1];
            buf0[12 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k) * c[4][0][1];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k) * c[0][0][2];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k) * c[1][0][2];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) * c[2][0][2];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k) * c[3][0][2];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k) * c[4][0][2];
            buf0[19 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k) * c[0][1][0];
            buf0[19 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k) * c[1][1][0];
            buf0[19 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) * c[2][1][0];
            buf0[19 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k) * c[3][1][0];
            buf0[19 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k) * c[4][1][0];
            buf0[11 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k) * c[0][1][1];
            buf0[11 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k) * c[1][1][1];
            buf0[11 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) * c[2][1][1];
            buf0[11 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k) * c[3][1][1];
            buf0[11 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k) * c[4][1][1];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k) * c[0][1][2];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k) * c[1][1][2];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) * c[2][1][2];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k) * c[3][1][2];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k) * c[4][1][2];
            buf0[18 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k) * c[0][2][0];
            buf0[18 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k) * c[1][2][0];
            buf0[18 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) * c[2][2][0];
            buf0[18 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k) * c[3][2][0];
            buf0[18 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k) * c[4][2][0];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k) * c[0][2][1];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k) * c[1][2][1];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) * c[2][2][1];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k) * c[3][2][1];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k) * c[4][2][1];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k) * c[0][2][2];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k) * c[1][2][2];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) * c[2][2][2];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k) * c[3][2][2];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k) * c[4][2][2];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k) * c[0][3][0];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k) * c[1][3][0];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) * c[2][3][0];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k) * c[3][3][0];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k) * c[4][3][0];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k) * c[0][3][1];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k) * c[1][3][1];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) * c[2][3][1];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k) * c[3][3][1];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k) * c[4][3][1];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k) * c[0][3][2];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k) * c[1][3][2];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) * c[2][3][2];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k) * c[3][3][2];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k) * c[4][3][2];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k) * c[0][4][0];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k) * c[1][4][0];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) * c[2][4][0];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k) * c[3][4][0];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k) * c[4][4][0];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k) * c[0][4][1];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k) * c[1][4][1];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) * c[2][4][1];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k) * c[3][4][1];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k) * c[4][4][1];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k) * c[0][4][2];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k) * c[1][4][2];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k) * c[2][4][2];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k) * c[3][4][2];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k) * c[4][4][2];
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[23] += bIn(i + hipThreadIdx_x + -2, j + 6, k) * c[0][1][0];
        buf0[23] += bIn(i + hipThreadIdx_x + -1, j + 6, k) * c[1][1][0];
        buf0[23] += bIn(i + hipThreadIdx_x, j + 6, k) * c[2][1][0];
        buf0[23] += bIn(i + hipThreadIdx_x + 1, j + 6, k) * c[3][1][0];
        buf0[23] += bIn(i + hipThreadIdx_x + 2, j + 6, k) * c[4][1][0];
        buf0[15] += bIn(i + hipThreadIdx_x + -2, j + 6, k) * c[0][1][1];
        buf0[15] += bIn(i + hipThreadIdx_x + -1, j + 6, k) * c[1][1][1];
        buf0[15] += bIn(i + hipThreadIdx_x, j + 6, k) * c[2][1][1];
        buf0[15] += bIn(i + hipThreadIdx_x + 1, j + 6, k) * c[3][1][1];
        buf0[15] += bIn(i + hipThreadIdx_x + 2, j + 6, k) * c[4][1][1];
        buf0[7] += bIn(i + hipThreadIdx_x + -2, j + 6, k) * c[0][1][2];
        buf0[7] += bIn(i + hipThreadIdx_x + -1, j + 6, k) * c[1][1][2];
        buf0[7] += bIn(i + hipThreadIdx_x, j + 6, k) * c[2][1][2];
        buf0[7] += bIn(i + hipThreadIdx_x + 1, j + 6, k) * c[3][1][2];
        buf0[7] += bIn(i + hipThreadIdx_x + 2, j + 6, k) * c[4][1][2];
        buf0[22] += bIn(i + hipThreadIdx_x + -2, j + 6, k) * c[0][2][0];
        buf0[22] += bIn(i + hipThreadIdx_x + -1, j + 6, k) * c[1][2][0];
        buf0[22] += bIn(i + hipThreadIdx_x, j + 6, k) * c[2][2][0];
        buf0[22] += bIn(i + hipThreadIdx_x + 1, j + 6, k) * c[3][2][0];
        buf0[22] += bIn(i + hipThreadIdx_x + 2, j + 6, k) * c[4][2][0];
        buf0[14] += bIn(i + hipThreadIdx_x + -2, j + 6, k) * c[0][2][1];
        buf0[14] += bIn(i + hipThreadIdx_x + -1, j + 6, k) * c[1][2][1];
        buf0[14] += bIn(i + hipThreadIdx_x, j + 6, k) * c[2][2][1];
        buf0[14] += bIn(i + hipThreadIdx_x + 1, j + 6, k) * c[3][2][1];
        buf0[14] += bIn(i + hipThreadIdx_x + 2, j + 6, k) * c[4][2][1];
        buf0[6] += bIn(i + hipThreadIdx_x + -2, j + 6, k) * c[0][2][2];
        buf0[6] += bIn(i + hipThreadIdx_x + -1, j + 6, k) * c[1][2][2];
        buf0[6] += bIn(i + hipThreadIdx_x, j + 6, k) * c[2][2][2];
        buf0[6] += bIn(i + hipThreadIdx_x + 1, j + 6, k) * c[3][2][2];
        buf0[6] += bIn(i + hipThreadIdx_x + 2, j + 6, k) * c[4][2][2];
        buf0[21] += bIn(i + hipThreadIdx_x + -2, j + 6, k) * c[0][3][0];
        buf0[21] += bIn(i + hipThreadIdx_x + -1, j + 6, k) * c[1][3][0];
        buf0[21] += bIn(i + hipThreadIdx_x, j + 6, k) * c[2][3][0];
        buf0[21] += bIn(i + hipThreadIdx_x + 1, j + 6, k) * c[3][3][0];
        buf0[21] += bIn(i + hipThreadIdx_x + 2, j + 6, k) * c[4][3][0];
        buf0[13] += bIn(i + hipThreadIdx_x + -2, j + 6, k) * c[0][3][1];
        buf0[13] += bIn(i + hipThreadIdx_x + -1, j + 6, k) * c[1][3][1];
        buf0[13] += bIn(i + hipThreadIdx_x, j + 6, k) * c[2][3][1];
        buf0[13] += bIn(i + hipThreadIdx_x + 1, j + 6, k) * c[3][3][1];
        buf0[13] += bIn(i + hipThreadIdx_x + 2, j + 6, k) * c[4][3][1];
        buf0[5] += bIn(i + hipThreadIdx_x + -2, j + 6, k) * c[0][3][2];
        buf0[5] += bIn(i + hipThreadIdx_x + -1, j + 6, k) * c[1][3][2];
        buf0[5] += bIn(i + hipThreadIdx_x, j + 6, k) * c[2][3][2];
        buf0[5] += bIn(i + hipThreadIdx_x + 1, j + 6, k) * c[3][3][2];
        buf0[5] += bIn(i + hipThreadIdx_x + 2, j + 6, k) * c[4][3][2];
        buf0[20] += bIn(i + hipThreadIdx_x + -2, j + 6, k) * c[0][4][0];
        buf0[20] += bIn(i + hipThreadIdx_x + -1, j + 6, k) * c[1][4][0];
        buf0[20] += bIn(i + hipThreadIdx_x, j + 6, k) * c[2][4][0];
        buf0[20] += bIn(i + hipThreadIdx_x + 1, j + 6, k) * c[3][4][0];
        buf0[20] += bIn(i + hipThreadIdx_x + 2, j + 6, k) * c[4][4][0];
        buf0[12] += bIn(i + hipThreadIdx_x + -2, j + 6, k) * c[0][4][1];
        buf0[12] += bIn(i + hipThreadIdx_x + -1, j + 6, k) * c[1][4][1];
        buf0[12] += bIn(i + hipThreadIdx_x, j + 6, k) * c[2][4][1];
        buf0[12] += bIn(i + hipThreadIdx_x + 1, j + 6, k) * c[3][4][1];
        buf0[12] += bIn(i + hipThreadIdx_x + 2, j + 6, k) * c[4][4][1];
        buf0[4] += bIn(i + hipThreadIdx_x + -2, j + 6, k) * c[0][4][2];
        buf0[4] += bIn(i + hipThreadIdx_x + -1, j + 6, k) * c[1][4][2];
        buf0[4] += bIn(i + hipThreadIdx_x, j + 6, k) * c[2][4][2];
        buf0[4] += bIn(i + hipThreadIdx_x + 1, j + 6, k) * c[3][4][2];
        buf0[4] += bIn(i + hipThreadIdx_x + 2, j + 6, k) * c[4][4][2];
      }
      {
        buf0[23] += bIn(i + hipThreadIdx_x + -2, j + 7, k) * c[0][2][0];
        buf0[23] += bIn(i + hipThreadIdx_x + -1, j + 7, k) * c[1][2][0];
        buf0[23] += bIn(i + hipThreadIdx_x, j + 7, k) * c[2][2][0];
        buf0[23] += bIn(i + hipThreadIdx_x + 1, j + 7, k) * c[3][2][0];
        buf0[23] += bIn(i + hipThreadIdx_x + 2, j + 7, k) * c[4][2][0];
        buf0[15] += bIn(i + hipThreadIdx_x + -2, j + 7, k) * c[0][2][1];
        buf0[15] += bIn(i + hipThreadIdx_x + -1, j + 7, k) * c[1][2][1];
        buf0[15] += bIn(i + hipThreadIdx_x, j + 7, k) * c[2][2][1];
        buf0[15] += bIn(i + hipThreadIdx_x + 1, j + 7, k) * c[3][2][1];
        buf0[15] += bIn(i + hipThreadIdx_x + 2, j + 7, k) * c[4][2][1];
        buf0[7] += bIn(i + hipThreadIdx_x + -2, j + 7, k) * c[0][2][2];
        buf0[7] += bIn(i + hipThreadIdx_x + -1, j + 7, k) * c[1][2][2];
        buf0[7] += bIn(i + hipThreadIdx_x, j + 7, k) * c[2][2][2];
        buf0[7] += bIn(i + hipThreadIdx_x + 1, j + 7, k) * c[3][2][2];
        buf0[7] += bIn(i + hipThreadIdx_x + 2, j + 7, k) * c[4][2][2];
        buf0[22] += bIn(i + hipThreadIdx_x + -2, j + 7, k) * c[0][3][0];
        buf0[22] += bIn(i + hipThreadIdx_x + -1, j + 7, k) * c[1][3][0];
        buf0[22] += bIn(i + hipThreadIdx_x, j + 7, k) * c[2][3][0];
        buf0[22] += bIn(i + hipThreadIdx_x + 1, j + 7, k) * c[3][3][0];
        buf0[22] += bIn(i + hipThreadIdx_x + 2, j + 7, k) * c[4][3][0];
        buf0[14] += bIn(i + hipThreadIdx_x + -2, j + 7, k) * c[0][3][1];
        buf0[14] += bIn(i + hipThreadIdx_x + -1, j + 7, k) * c[1][3][1];
        buf0[14] += bIn(i + hipThreadIdx_x, j + 7, k) * c[2][3][1];
        buf0[14] += bIn(i + hipThreadIdx_x + 1, j + 7, k) * c[3][3][1];
        buf0[14] += bIn(i + hipThreadIdx_x + 2, j + 7, k) * c[4][3][1];
        buf0[6] += bIn(i + hipThreadIdx_x + -2, j + 7, k) * c[0][3][2];
        buf0[6] += bIn(i + hipThreadIdx_x + -1, j + 7, k) * c[1][3][2];
        buf0[6] += bIn(i + hipThreadIdx_x, j + 7, k) * c[2][3][2];
        buf0[6] += bIn(i + hipThreadIdx_x + 1, j + 7, k) * c[3][3][2];
        buf0[6] += bIn(i + hipThreadIdx_x + 2, j + 7, k) * c[4][3][2];
        buf0[21] += bIn(i + hipThreadIdx_x + -2, j + 7, k) * c[0][4][0];
        buf0[21] += bIn(i + hipThreadIdx_x + -1, j + 7, k) * c[1][4][0];
        buf0[21] += bIn(i + hipThreadIdx_x, j + 7, k) * c[2][4][0];
        buf0[21] += bIn(i + hipThreadIdx_x + 1, j + 7, k) * c[3][4][0];
        buf0[21] += bIn(i + hipThreadIdx_x + 2, j + 7, k) * c[4][4][0];
        buf0[13] += bIn(i + hipThreadIdx_x + -2, j + 7, k) * c[0][4][1];
        buf0[13] += bIn(i + hipThreadIdx_x + -1, j + 7, k) * c[1][4][1];
        buf0[13] += bIn(i + hipThreadIdx_x, j + 7, k) * c[2][4][1];
        buf0[13] += bIn(i + hipThreadIdx_x + 1, j + 7, k) * c[3][4][1];
        buf0[13] += bIn(i + hipThreadIdx_x + 2, j + 7, k) * c[4][4][1];
        buf0[5] += bIn(i + hipThreadIdx_x + -2, j + 7, k) * c[0][4][2];
        buf0[5] += bIn(i + hipThreadIdx_x + -1, j + 7, k) * c[1][4][2];
        buf0[5] += bIn(i + hipThreadIdx_x, j + 7, k) * c[2][4][2];
        buf0[5] += bIn(i + hipThreadIdx_x + 1, j + 7, k) * c[3][4][2];
        buf0[5] += bIn(i + hipThreadIdx_x + 2, j + 7, k) * c[4][4][2];
      }
      {
        buf0[23] += bIn(i + hipThreadIdx_x + -2, j + 8, k) * c[0][3][0];
        buf0[23] += bIn(i + hipThreadIdx_x + -1, j + 8, k) * c[1][3][0];
        buf0[23] += bIn(i + hipThreadIdx_x, j + 8, k) * c[2][3][0];
        buf0[23] += bIn(i + hipThreadIdx_x + 1, j + 8, k) * c[3][3][0];
        buf0[23] += bIn(i + hipThreadIdx_x + 2, j + 8, k) * c[4][3][0];
        buf0[15] += bIn(i + hipThreadIdx_x + -2, j + 8, k) * c[0][3][1];
        buf0[15] += bIn(i + hipThreadIdx_x + -1, j + 8, k) * c[1][3][1];
        buf0[15] += bIn(i + hipThreadIdx_x, j + 8, k) * c[2][3][1];
        buf0[15] += bIn(i + hipThreadIdx_x + 1, j + 8, k) * c[3][3][1];
        buf0[15] += bIn(i + hipThreadIdx_x + 2, j + 8, k) * c[4][3][1];
        buf0[7] += bIn(i + hipThreadIdx_x + -2, j + 8, k) * c[0][3][2];
        buf0[7] += bIn(i + hipThreadIdx_x + -1, j + 8, k) * c[1][3][2];
        buf0[7] += bIn(i + hipThreadIdx_x, j + 8, k) * c[2][3][2];
        buf0[7] += bIn(i + hipThreadIdx_x + 1, j + 8, k) * c[3][3][2];
        buf0[7] += bIn(i + hipThreadIdx_x + 2, j + 8, k) * c[4][3][2];
        buf0[22] += bIn(i + hipThreadIdx_x + -2, j + 8, k) * c[0][4][0];
        buf0[22] += bIn(i + hipThreadIdx_x + -1, j + 8, k) * c[1][4][0];
        buf0[22] += bIn(i + hipThreadIdx_x, j + 8, k) * c[2][4][0];
        buf0[22] += bIn(i + hipThreadIdx_x + 1, j + 8, k) * c[3][4][0];
        buf0[22] += bIn(i + hipThreadIdx_x + 2, j + 8, k) * c[4][4][0];
        buf0[14] += bIn(i + hipThreadIdx_x + -2, j + 8, k) * c[0][4][1];
        buf0[14] += bIn(i + hipThreadIdx_x + -1, j + 8, k) * c[1][4][1];
        buf0[14] += bIn(i + hipThreadIdx_x, j + 8, k) * c[2][4][1];
        buf0[14] += bIn(i + hipThreadIdx_x + 1, j + 8, k) * c[3][4][1];
        buf0[14] += bIn(i + hipThreadIdx_x + 2, j + 8, k) * c[4][4][1];
        buf0[6] += bIn(i + hipThreadIdx_x + -2, j + 8, k) * c[0][4][2];
        buf0[6] += bIn(i + hipThreadIdx_x + -1, j + 8, k) * c[1][4][2];
        buf0[6] += bIn(i + hipThreadIdx_x, j + 8, k) * c[2][4][2];
        buf0[6] += bIn(i + hipThreadIdx_x + 1, j + 8, k) * c[3][4][2];
        buf0[6] += bIn(i + hipThreadIdx_x + 2, j + 8, k) * c[4][4][2];
      }
      {
        buf0[23] += bIn(i + hipThreadIdx_x + -2, j + 9, k) * c[0][4][0];
        buf0[23] += bIn(i + hipThreadIdx_x + -1, j + 9, k) * c[1][4][0];
        buf0[23] += bIn(i + hipThreadIdx_x, j + 9, k) * c[2][4][0];
        buf0[23] += bIn(i + hipThreadIdx_x + 1, j + 9, k) * c[3][4][0];
        buf0[23] += bIn(i + hipThreadIdx_x + 2, j + 9, k) * c[4][4][0];
        buf0[15] += bIn(i + hipThreadIdx_x + -2, j + 9, k) * c[0][4][1];
        buf0[15] += bIn(i + hipThreadIdx_x + -1, j + 9, k) * c[1][4][1];
        buf0[15] += bIn(i + hipThreadIdx_x, j + 9, k) * c[2][4][1];
        buf0[15] += bIn(i + hipThreadIdx_x + 1, j + 9, k) * c[3][4][1];
        buf0[15] += bIn(i + hipThreadIdx_x + 2, j + 9, k) * c[4][4][1];
        buf0[7] += bIn(i + hipThreadIdx_x + -2, j + 9, k) * c[0][4][2];
        buf0[7] += bIn(i + hipThreadIdx_x + -1, j + 9, k) * c[1][4][2];
        buf0[7] += bIn(i + hipThreadIdx_x, j + 9, k) * c[2][4][2];
        buf0[7] += bIn(i + hipThreadIdx_x + 1, j + 9, k) * c[3][4][2];
        buf0[7] += bIn(i + hipThreadIdx_x + 2, j + 9, k) * c[4][4][2];
      }
      {
        buf0[24] += 0;
        buf0[24] += bIn(i + hipThreadIdx_x + -2, j + -2, k + 1) * c[0][0][0];
        buf0[24] += bIn(i + hipThreadIdx_x + -1, j + -2, k + 1) * c[1][0][0];
        buf0[24] += bIn(i + hipThreadIdx_x, j + -2, k + 1) * c[2][0][0];
        buf0[24] += bIn(i + hipThreadIdx_x + 1, j + -2, k + 1) * c[3][0][0];
        buf0[24] += bIn(i + hipThreadIdx_x + 2, j + -2, k + 1) * c[4][0][0];
        buf0[16] += bIn(i + hipThreadIdx_x + -2, j + -2, k + 1) * c[0][0][1];
        buf0[16] += bIn(i + hipThreadIdx_x + -1, j + -2, k + 1) * c[1][0][1];
        buf0[16] += bIn(i + hipThreadIdx_x, j + -2, k + 1) * c[2][0][1];
        buf0[16] += bIn(i + hipThreadIdx_x + 1, j + -2, k + 1) * c[3][0][1];
        buf0[16] += bIn(i + hipThreadIdx_x + 2, j + -2, k + 1) * c[4][0][1];
        buf0[8] += bIn(i + hipThreadIdx_x + -2, j + -2, k + 1) * c[0][0][2];
        buf0[8] += bIn(i + hipThreadIdx_x + -1, j + -2, k + 1) * c[1][0][2];
        buf0[8] += bIn(i + hipThreadIdx_x, j + -2, k + 1) * c[2][0][2];
        buf0[8] += bIn(i + hipThreadIdx_x + 1, j + -2, k + 1) * c[3][0][2];
        buf0[8] += bIn(i + hipThreadIdx_x + 2, j + -2, k + 1) * c[4][0][2];
        buf0[0] += bIn(i + hipThreadIdx_x + -2, j + -2, k + 1) * c[0][0][3];
        buf0[0] += bIn(i + hipThreadIdx_x + -1, j + -2, k + 1) * c[1][0][3];
        buf0[0] += bIn(i + hipThreadIdx_x, j + -2, k + 1) * c[2][0][3];
        buf0[0] += bIn(i + hipThreadIdx_x + 1, j + -2, k + 1) * c[3][0][3];
        buf0[0] += bIn(i + hipThreadIdx_x + 2, j + -2, k + 1) * c[4][0][3];
      }
      {
        buf0[25] += 0;
        buf0[25] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 1) * c[0][0][0];
        buf0[25] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 1) * c[1][0][0];
        buf0[25] += bIn(i + hipThreadIdx_x, j + -1, k + 1) * c[2][0][0];
        buf0[25] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 1) * c[3][0][0];
        buf0[25] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 1) * c[4][0][0];
        buf0[17] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 1) * c[0][0][1];
        buf0[17] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 1) * c[1][0][1];
        buf0[17] += bIn(i + hipThreadIdx_x, j + -1, k + 1) * c[2][0][1];
        buf0[17] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 1) * c[3][0][1];
        buf0[17] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 1) * c[4][0][1];
        buf0[9] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 1) * c[0][0][2];
        buf0[9] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 1) * c[1][0][2];
        buf0[9] += bIn(i + hipThreadIdx_x, j + -1, k + 1) * c[2][0][2];
        buf0[9] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 1) * c[3][0][2];
        buf0[9] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 1) * c[4][0][2];
        buf0[1] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 1) * c[0][0][3];
        buf0[1] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 1) * c[1][0][3];
        buf0[1] += bIn(i + hipThreadIdx_x, j + -1, k + 1) * c[2][0][3];
        buf0[1] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 1) * c[3][0][3];
        buf0[1] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 1) * c[4][0][3];
        buf0[24] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 1) * c[0][1][0];
        buf0[24] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 1) * c[1][1][0];
        buf0[24] += bIn(i + hipThreadIdx_x, j + -1, k + 1) * c[2][1][0];
        buf0[24] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 1) * c[3][1][0];
        buf0[24] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 1) * c[4][1][0];
        buf0[16] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 1) * c[0][1][1];
        buf0[16] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 1) * c[1][1][1];
        buf0[16] += bIn(i + hipThreadIdx_x, j + -1, k + 1) * c[2][1][1];
        buf0[16] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 1) * c[3][1][1];
        buf0[16] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 1) * c[4][1][1];
        buf0[8] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 1) * c[0][1][2];
        buf0[8] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 1) * c[1][1][2];
        buf0[8] += bIn(i + hipThreadIdx_x, j + -1, k + 1) * c[2][1][2];
        buf0[8] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 1) * c[3][1][2];
        buf0[8] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 1) * c[4][1][2];
        buf0[0] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 1) * c[0][1][3];
        buf0[0] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 1) * c[1][1][3];
        buf0[0] += bIn(i + hipThreadIdx_x, j + -1, k + 1) * c[2][1][3];
        buf0[0] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 1) * c[3][1][3];
        buf0[0] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 1) * c[4][1][3];
      }
      {
        buf0[26] += 0;
        buf0[26] += bIn(i + hipThreadIdx_x + -2, j, k + 1) * c[0][0][0];
        buf0[26] += bIn(i + hipThreadIdx_x + -1, j, k + 1) * c[1][0][0];
        buf0[26] += bIn(i + hipThreadIdx_x, j, k + 1) * c[2][0][0];
        buf0[26] += bIn(i + hipThreadIdx_x + 1, j, k + 1) * c[3][0][0];
        buf0[26] += bIn(i + hipThreadIdx_x + 2, j, k + 1) * c[4][0][0];
        buf0[18] += bIn(i + hipThreadIdx_x + -2, j, k + 1) * c[0][0][1];
        buf0[18] += bIn(i + hipThreadIdx_x + -1, j, k + 1) * c[1][0][1];
        buf0[18] += bIn(i + hipThreadIdx_x, j, k + 1) * c[2][0][1];
        buf0[18] += bIn(i + hipThreadIdx_x + 1, j, k + 1) * c[3][0][1];
        buf0[18] += bIn(i + hipThreadIdx_x + 2, j, k + 1) * c[4][0][1];
        buf0[10] += bIn(i + hipThreadIdx_x + -2, j, k + 1) * c[0][0][2];
        buf0[10] += bIn(i + hipThreadIdx_x + -1, j, k + 1) * c[1][0][2];
        buf0[10] += bIn(i + hipThreadIdx_x, j, k + 1) * c[2][0][2];
        buf0[10] += bIn(i + hipThreadIdx_x + 1, j, k + 1) * c[3][0][2];
        buf0[10] += bIn(i + hipThreadIdx_x + 2, j, k + 1) * c[4][0][2];
        buf0[2] += bIn(i + hipThreadIdx_x + -2, j, k + 1) * c[0][0][3];
        buf0[2] += bIn(i + hipThreadIdx_x + -1, j, k + 1) * c[1][0][3];
        buf0[2] += bIn(i + hipThreadIdx_x, j, k + 1) * c[2][0][3];
        buf0[2] += bIn(i + hipThreadIdx_x + 1, j, k + 1) * c[3][0][3];
        buf0[2] += bIn(i + hipThreadIdx_x + 2, j, k + 1) * c[4][0][3];
        buf0[25] += bIn(i + hipThreadIdx_x + -2, j, k + 1) * c[0][1][0];
        buf0[25] += bIn(i + hipThreadIdx_x + -1, j, k + 1) * c[1][1][0];
        buf0[25] += bIn(i + hipThreadIdx_x, j, k + 1) * c[2][1][0];
        buf0[25] += bIn(i + hipThreadIdx_x + 1, j, k + 1) * c[3][1][0];
        buf0[25] += bIn(i + hipThreadIdx_x + 2, j, k + 1) * c[4][1][0];
        buf0[17] += bIn(i + hipThreadIdx_x + -2, j, k + 1) * c[0][1][1];
        buf0[17] += bIn(i + hipThreadIdx_x + -1, j, k + 1) * c[1][1][1];
        buf0[17] += bIn(i + hipThreadIdx_x, j, k + 1) * c[2][1][1];
        buf0[17] += bIn(i + hipThreadIdx_x + 1, j, k + 1) * c[3][1][1];
        buf0[17] += bIn(i + hipThreadIdx_x + 2, j, k + 1) * c[4][1][1];
        buf0[9] += bIn(i + hipThreadIdx_x + -2, j, k + 1) * c[0][1][2];
        buf0[9] += bIn(i + hipThreadIdx_x + -1, j, k + 1) * c[1][1][2];
        buf0[9] += bIn(i + hipThreadIdx_x, j, k + 1) * c[2][1][2];
        buf0[9] += bIn(i + hipThreadIdx_x + 1, j, k + 1) * c[3][1][2];
        buf0[9] += bIn(i + hipThreadIdx_x + 2, j, k + 1) * c[4][1][2];
        buf0[1] += bIn(i + hipThreadIdx_x + -2, j, k + 1) * c[0][1][3];
        buf0[1] += bIn(i + hipThreadIdx_x + -1, j, k + 1) * c[1][1][3];
        buf0[1] += bIn(i + hipThreadIdx_x, j, k + 1) * c[2][1][3];
        buf0[1] += bIn(i + hipThreadIdx_x + 1, j, k + 1) * c[3][1][3];
        buf0[1] += bIn(i + hipThreadIdx_x + 2, j, k + 1) * c[4][1][3];
        buf0[24] += bIn(i + hipThreadIdx_x + -2, j, k + 1) * c[0][2][0];
        buf0[24] += bIn(i + hipThreadIdx_x + -1, j, k + 1) * c[1][2][0];
        buf0[24] += bIn(i + hipThreadIdx_x, j, k + 1) * c[2][2][0];
        buf0[24] += bIn(i + hipThreadIdx_x + 1, j, k + 1) * c[3][2][0];
        buf0[24] += bIn(i + hipThreadIdx_x + 2, j, k + 1) * c[4][2][0];
        buf0[16] += bIn(i + hipThreadIdx_x + -2, j, k + 1) * c[0][2][1];
        buf0[16] += bIn(i + hipThreadIdx_x + -1, j, k + 1) * c[1][2][1];
        buf0[16] += bIn(i + hipThreadIdx_x, j, k + 1) * c[2][2][1];
        buf0[16] += bIn(i + hipThreadIdx_x + 1, j, k + 1) * c[3][2][1];
        buf0[16] += bIn(i + hipThreadIdx_x + 2, j, k + 1) * c[4][2][1];
        buf0[8] += bIn(i + hipThreadIdx_x + -2, j, k + 1) * c[0][2][2];
        buf0[8] += bIn(i + hipThreadIdx_x + -1, j, k + 1) * c[1][2][2];
        buf0[8] += bIn(i + hipThreadIdx_x, j, k + 1) * c[2][2][2];
        buf0[8] += bIn(i + hipThreadIdx_x + 1, j, k + 1) * c[3][2][2];
        buf0[8] += bIn(i + hipThreadIdx_x + 2, j, k + 1) * c[4][2][2];
        buf0[0] += bIn(i + hipThreadIdx_x + -2, j, k + 1) * c[0][2][3];
        buf0[0] += bIn(i + hipThreadIdx_x + -1, j, k + 1) * c[1][2][3];
        buf0[0] += bIn(i + hipThreadIdx_x, j, k + 1) * c[2][2][3];
        buf0[0] += bIn(i + hipThreadIdx_x + 1, j, k + 1) * c[3][2][3];
        buf0[0] += bIn(i + hipThreadIdx_x + 2, j, k + 1) * c[4][2][3];
      }
      {
        buf0[27] += 0;
        buf0[27] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 1) * c[0][0][0];
        buf0[27] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 1) * c[1][0][0];
        buf0[27] += bIn(i + hipThreadIdx_x, j + 1, k + 1) * c[2][0][0];
        buf0[27] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 1) * c[3][0][0];
        buf0[27] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 1) * c[4][0][0];
        buf0[19] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 1) * c[0][0][1];
        buf0[19] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 1) * c[1][0][1];
        buf0[19] += bIn(i + hipThreadIdx_x, j + 1, k + 1) * c[2][0][1];
        buf0[19] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 1) * c[3][0][1];
        buf0[19] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 1) * c[4][0][1];
        buf0[11] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 1) * c[0][0][2];
        buf0[11] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 1) * c[1][0][2];
        buf0[11] += bIn(i + hipThreadIdx_x, j + 1, k + 1) * c[2][0][2];
        buf0[11] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 1) * c[3][0][2];
        buf0[11] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 1) * c[4][0][2];
        buf0[3] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 1) * c[0][0][3];
        buf0[3] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 1) * c[1][0][3];
        buf0[3] += bIn(i + hipThreadIdx_x, j + 1, k + 1) * c[2][0][3];
        buf0[3] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 1) * c[3][0][3];
        buf0[3] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 1) * c[4][0][3];
        buf0[26] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 1) * c[0][1][0];
        buf0[26] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 1) * c[1][1][0];
        buf0[26] += bIn(i + hipThreadIdx_x, j + 1, k + 1) * c[2][1][0];
        buf0[26] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 1) * c[3][1][0];
        buf0[26] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 1) * c[4][1][0];
        buf0[18] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 1) * c[0][1][1];
        buf0[18] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 1) * c[1][1][1];
        buf0[18] += bIn(i + hipThreadIdx_x, j + 1, k + 1) * c[2][1][1];
        buf0[18] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 1) * c[3][1][1];
        buf0[18] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 1) * c[4][1][1];
        buf0[10] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 1) * c[0][1][2];
        buf0[10] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 1) * c[1][1][2];
        buf0[10] += bIn(i + hipThreadIdx_x, j + 1, k + 1) * c[2][1][2];
        buf0[10] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 1) * c[3][1][2];
        buf0[10] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 1) * c[4][1][2];
        buf0[2] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 1) * c[0][1][3];
        buf0[2] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 1) * c[1][1][3];
        buf0[2] += bIn(i + hipThreadIdx_x, j + 1, k + 1) * c[2][1][3];
        buf0[2] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 1) * c[3][1][3];
        buf0[2] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 1) * c[4][1][3];
        buf0[25] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 1) * c[0][2][0];
        buf0[25] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 1) * c[1][2][0];
        buf0[25] += bIn(i + hipThreadIdx_x, j + 1, k + 1) * c[2][2][0];
        buf0[25] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 1) * c[3][2][0];
        buf0[25] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 1) * c[4][2][0];
        buf0[17] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 1) * c[0][2][1];
        buf0[17] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 1) * c[1][2][1];
        buf0[17] += bIn(i + hipThreadIdx_x, j + 1, k + 1) * c[2][2][1];
        buf0[17] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 1) * c[3][2][1];
        buf0[17] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 1) * c[4][2][1];
        buf0[9] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 1) * c[0][2][2];
        buf0[9] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 1) * c[1][2][2];
        buf0[9] += bIn(i + hipThreadIdx_x, j + 1, k + 1) * c[2][2][2];
        buf0[9] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 1) * c[3][2][2];
        buf0[9] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 1) * c[4][2][2];
        buf0[1] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 1) * c[0][2][3];
        buf0[1] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 1) * c[1][2][3];
        buf0[1] += bIn(i + hipThreadIdx_x, j + 1, k + 1) * c[2][2][3];
        buf0[1] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 1) * c[3][2][3];
        buf0[1] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 1) * c[4][2][3];
        buf0[24] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 1) * c[0][3][0];
        buf0[24] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 1) * c[1][3][0];
        buf0[24] += bIn(i + hipThreadIdx_x, j + 1, k + 1) * c[2][3][0];
        buf0[24] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 1) * c[3][3][0];
        buf0[24] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 1) * c[4][3][0];
        buf0[16] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 1) * c[0][3][1];
        buf0[16] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 1) * c[1][3][1];
        buf0[16] += bIn(i + hipThreadIdx_x, j + 1, k + 1) * c[2][3][1];
        buf0[16] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 1) * c[3][3][1];
        buf0[16] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 1) * c[4][3][1];
        buf0[8] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 1) * c[0][3][2];
        buf0[8] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 1) * c[1][3][2];
        buf0[8] += bIn(i + hipThreadIdx_x, j + 1, k + 1) * c[2][3][2];
        buf0[8] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 1) * c[3][3][2];
        buf0[8] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 1) * c[4][3][2];
        buf0[0] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 1) * c[0][3][3];
        buf0[0] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 1) * c[1][3][3];
        buf0[0] += bIn(i + hipThreadIdx_x, j + 1, k + 1) * c[2][3][3];
        buf0[0] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 1) * c[3][3][3];
        buf0[0] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 1) * c[4][3][3];
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 4; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[28 + rel] += 0;
            buf0[28 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1) * c[0][0][0];
            buf0[28 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1) * c[1][0][0];
            buf0[28 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1) * c[2][0][0];
            buf0[28 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1) * c[3][0][0];
            buf0[28 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1) * c[4][0][0];
            buf0[20 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1) * c[0][0][1];
            buf0[20 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1) * c[1][0][1];
            buf0[20 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1) * c[2][0][1];
            buf0[20 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1) * c[3][0][1];
            buf0[20 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1) * c[4][0][1];
            buf0[12 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1) * c[0][0][2];
            buf0[12 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1) * c[1][0][2];
            buf0[12 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1) * c[2][0][2];
            buf0[12 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1) * c[3][0][2];
            buf0[12 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1) * c[4][0][2];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1) * c[0][0][3];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1) * c[1][0][3];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1) * c[2][0][3];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1) * c[3][0][3];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1) * c[4][0][3];
            buf0[27 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1) * c[0][1][0];
            buf0[27 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1) * c[1][1][0];
            buf0[27 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1) * c[2][1][0];
            buf0[27 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1) * c[3][1][0];
            buf0[27 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1) * c[4][1][0];
            buf0[19 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1) * c[0][1][1];
            buf0[19 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1) * c[1][1][1];
            buf0[19 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1) * c[2][1][1];
            buf0[19 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1) * c[3][1][1];
            buf0[19 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1) * c[4][1][1];
            buf0[11 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1) * c[0][1][2];
            buf0[11 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1) * c[1][1][2];
            buf0[11 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1) * c[2][1][2];
            buf0[11 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1) * c[3][1][2];
            buf0[11 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1) * c[4][1][2];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1) * c[0][1][3];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1) * c[1][1][3];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1) * c[2][1][3];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1) * c[3][1][3];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1) * c[4][1][3];
            buf0[26 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1) * c[0][2][0];
            buf0[26 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1) * c[1][2][0];
            buf0[26 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1) * c[2][2][0];
            buf0[26 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1) * c[3][2][0];
            buf0[26 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1) * c[4][2][0];
            buf0[18 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1) * c[0][2][1];
            buf0[18 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1) * c[1][2][1];
            buf0[18 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1) * c[2][2][1];
            buf0[18 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1) * c[3][2][1];
            buf0[18 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1) * c[4][2][1];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1) * c[0][2][2];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1) * c[1][2][2];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1) * c[2][2][2];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1) * c[3][2][2];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1) * c[4][2][2];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1) * c[0][2][3];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1) * c[1][2][3];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1) * c[2][2][3];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1) * c[3][2][3];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1) * c[4][2][3];
            buf0[25 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1) * c[0][3][0];
            buf0[25 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1) * c[1][3][0];
            buf0[25 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1) * c[2][3][0];
            buf0[25 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1) * c[3][3][0];
            buf0[25 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1) * c[4][3][0];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1) * c[0][3][1];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1) * c[1][3][1];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1) * c[2][3][1];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1) * c[3][3][1];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1) * c[4][3][1];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1) * c[0][3][2];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1) * c[1][3][2];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1) * c[2][3][2];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1) * c[3][3][2];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1) * c[4][3][2];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1) * c[0][3][3];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1) * c[1][3][3];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1) * c[2][3][3];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1) * c[3][3][3];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1) * c[4][3][3];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1) * c[0][4][0];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1) * c[1][4][0];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1) * c[2][4][0];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1) * c[3][4][0];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1) * c[4][4][0];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1) * c[0][4][1];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1) * c[1][4][1];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1) * c[2][4][1];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1) * c[3][4][1];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1) * c[4][4][1];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1) * c[0][4][2];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1) * c[1][4][2];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1) * c[2][4][2];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1) * c[3][4][2];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1) * c[4][4][2];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1) * c[0][4][3];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1) * c[1][4][3];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1) * c[2][4][3];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1) * c[3][4][3];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1) * c[4][4][3];
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[31] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 1) * c[0][1][0];
        buf0[31] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 1) * c[1][1][0];
        buf0[31] += bIn(i + hipThreadIdx_x, j + 6, k + 1) * c[2][1][0];
        buf0[31] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 1) * c[3][1][0];
        buf0[31] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 1) * c[4][1][0];
        buf0[23] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 1) * c[0][1][1];
        buf0[23] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 1) * c[1][1][1];
        buf0[23] += bIn(i + hipThreadIdx_x, j + 6, k + 1) * c[2][1][1];
        buf0[23] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 1) * c[3][1][1];
        buf0[23] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 1) * c[4][1][1];
        buf0[15] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 1) * c[0][1][2];
        buf0[15] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 1) * c[1][1][2];
        buf0[15] += bIn(i + hipThreadIdx_x, j + 6, k + 1) * c[2][1][2];
        buf0[15] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 1) * c[3][1][2];
        buf0[15] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 1) * c[4][1][2];
        buf0[7] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 1) * c[0][1][3];
        buf0[7] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 1) * c[1][1][3];
        buf0[7] += bIn(i + hipThreadIdx_x, j + 6, k + 1) * c[2][1][3];
        buf0[7] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 1) * c[3][1][3];
        buf0[7] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 1) * c[4][1][3];
        buf0[30] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 1) * c[0][2][0];
        buf0[30] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 1) * c[1][2][0];
        buf0[30] += bIn(i + hipThreadIdx_x, j + 6, k + 1) * c[2][2][0];
        buf0[30] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 1) * c[3][2][0];
        buf0[30] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 1) * c[4][2][0];
        buf0[22] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 1) * c[0][2][1];
        buf0[22] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 1) * c[1][2][1];
        buf0[22] += bIn(i + hipThreadIdx_x, j + 6, k + 1) * c[2][2][1];
        buf0[22] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 1) * c[3][2][1];
        buf0[22] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 1) * c[4][2][1];
        buf0[14] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 1) * c[0][2][2];
        buf0[14] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 1) * c[1][2][2];
        buf0[14] += bIn(i + hipThreadIdx_x, j + 6, k + 1) * c[2][2][2];
        buf0[14] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 1) * c[3][2][2];
        buf0[14] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 1) * c[4][2][2];
        buf0[6] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 1) * c[0][2][3];
        buf0[6] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 1) * c[1][2][3];
        buf0[6] += bIn(i + hipThreadIdx_x, j + 6, k + 1) * c[2][2][3];
        buf0[6] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 1) * c[3][2][3];
        buf0[6] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 1) * c[4][2][3];
        buf0[29] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 1) * c[0][3][0];
        buf0[29] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 1) * c[1][3][0];
        buf0[29] += bIn(i + hipThreadIdx_x, j + 6, k + 1) * c[2][3][0];
        buf0[29] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 1) * c[3][3][0];
        buf0[29] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 1) * c[4][3][0];
        buf0[21] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 1) * c[0][3][1];
        buf0[21] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 1) * c[1][3][1];
        buf0[21] += bIn(i + hipThreadIdx_x, j + 6, k + 1) * c[2][3][1];
        buf0[21] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 1) * c[3][3][1];
        buf0[21] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 1) * c[4][3][1];
        buf0[13] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 1) * c[0][3][2];
        buf0[13] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 1) * c[1][3][2];
        buf0[13] += bIn(i + hipThreadIdx_x, j + 6, k + 1) * c[2][3][2];
        buf0[13] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 1) * c[3][3][2];
        buf0[13] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 1) * c[4][3][2];
        buf0[5] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 1) * c[0][3][3];
        buf0[5] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 1) * c[1][3][3];
        buf0[5] += bIn(i + hipThreadIdx_x, j + 6, k + 1) * c[2][3][3];
        buf0[5] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 1) * c[3][3][3];
        buf0[5] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 1) * c[4][3][3];
        buf0[28] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 1) * c[0][4][0];
        buf0[28] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 1) * c[1][4][0];
        buf0[28] += bIn(i + hipThreadIdx_x, j + 6, k + 1) * c[2][4][0];
        buf0[28] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 1) * c[3][4][0];
        buf0[28] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 1) * c[4][4][0];
        buf0[20] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 1) * c[0][4][1];
        buf0[20] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 1) * c[1][4][1];
        buf0[20] += bIn(i + hipThreadIdx_x, j + 6, k + 1) * c[2][4][1];
        buf0[20] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 1) * c[3][4][1];
        buf0[20] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 1) * c[4][4][1];
        buf0[12] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 1) * c[0][4][2];
        buf0[12] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 1) * c[1][4][2];
        buf0[12] += bIn(i + hipThreadIdx_x, j + 6, k + 1) * c[2][4][2];
        buf0[12] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 1) * c[3][4][2];
        buf0[12] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 1) * c[4][4][2];
        buf0[4] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 1) * c[0][4][3];
        buf0[4] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 1) * c[1][4][3];
        buf0[4] += bIn(i + hipThreadIdx_x, j + 6, k + 1) * c[2][4][3];
        buf0[4] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 1) * c[3][4][3];
        buf0[4] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 1) * c[4][4][3];
      }
      {
        buf0[31] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 1) * c[0][2][0];
        buf0[31] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 1) * c[1][2][0];
        buf0[31] += bIn(i + hipThreadIdx_x, j + 7, k + 1) * c[2][2][0];
        buf0[31] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 1) * c[3][2][0];
        buf0[31] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 1) * c[4][2][0];
        buf0[23] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 1) * c[0][2][1];
        buf0[23] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 1) * c[1][2][1];
        buf0[23] += bIn(i + hipThreadIdx_x, j + 7, k + 1) * c[2][2][1];
        buf0[23] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 1) * c[3][2][1];
        buf0[23] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 1) * c[4][2][1];
        buf0[15] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 1) * c[0][2][2];
        buf0[15] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 1) * c[1][2][2];
        buf0[15] += bIn(i + hipThreadIdx_x, j + 7, k + 1) * c[2][2][2];
        buf0[15] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 1) * c[3][2][2];
        buf0[15] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 1) * c[4][2][2];
        buf0[7] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 1) * c[0][2][3];
        buf0[7] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 1) * c[1][2][3];
        buf0[7] += bIn(i + hipThreadIdx_x, j + 7, k + 1) * c[2][2][3];
        buf0[7] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 1) * c[3][2][3];
        buf0[7] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 1) * c[4][2][3];
        buf0[30] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 1) * c[0][3][0];
        buf0[30] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 1) * c[1][3][0];
        buf0[30] += bIn(i + hipThreadIdx_x, j + 7, k + 1) * c[2][3][0];
        buf0[30] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 1) * c[3][3][0];
        buf0[30] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 1) * c[4][3][0];
        buf0[22] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 1) * c[0][3][1];
        buf0[22] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 1) * c[1][3][1];
        buf0[22] += bIn(i + hipThreadIdx_x, j + 7, k + 1) * c[2][3][1];
        buf0[22] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 1) * c[3][3][1];
        buf0[22] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 1) * c[4][3][1];
        buf0[14] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 1) * c[0][3][2];
        buf0[14] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 1) * c[1][3][2];
        buf0[14] += bIn(i + hipThreadIdx_x, j + 7, k + 1) * c[2][3][2];
        buf0[14] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 1) * c[3][3][2];
        buf0[14] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 1) * c[4][3][2];
        buf0[6] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 1) * c[0][3][3];
        buf0[6] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 1) * c[1][3][3];
        buf0[6] += bIn(i + hipThreadIdx_x, j + 7, k + 1) * c[2][3][3];
        buf0[6] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 1) * c[3][3][3];
        buf0[6] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 1) * c[4][3][3];
        buf0[29] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 1) * c[0][4][0];
        buf0[29] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 1) * c[1][4][0];
        buf0[29] += bIn(i + hipThreadIdx_x, j + 7, k + 1) * c[2][4][0];
        buf0[29] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 1) * c[3][4][0];
        buf0[29] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 1) * c[4][4][0];
        buf0[21] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 1) * c[0][4][1];
        buf0[21] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 1) * c[1][4][1];
        buf0[21] += bIn(i + hipThreadIdx_x, j + 7, k + 1) * c[2][4][1];
        buf0[21] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 1) * c[3][4][1];
        buf0[21] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 1) * c[4][4][1];
        buf0[13] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 1) * c[0][4][2];
        buf0[13] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 1) * c[1][4][2];
        buf0[13] += bIn(i + hipThreadIdx_x, j + 7, k + 1) * c[2][4][2];
        buf0[13] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 1) * c[3][4][2];
        buf0[13] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 1) * c[4][4][2];
        buf0[5] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 1) * c[0][4][3];
        buf0[5] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 1) * c[1][4][3];
        buf0[5] += bIn(i + hipThreadIdx_x, j + 7, k + 1) * c[2][4][3];
        buf0[5] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 1) * c[3][4][3];
        buf0[5] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 1) * c[4][4][3];
      }
      {
        buf0[31] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 1) * c[0][3][0];
        buf0[31] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 1) * c[1][3][0];
        buf0[31] += bIn(i + hipThreadIdx_x, j + 8, k + 1) * c[2][3][0];
        buf0[31] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 1) * c[3][3][0];
        buf0[31] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 1) * c[4][3][0];
        buf0[23] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 1) * c[0][3][1];
        buf0[23] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 1) * c[1][3][1];
        buf0[23] += bIn(i + hipThreadIdx_x, j + 8, k + 1) * c[2][3][1];
        buf0[23] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 1) * c[3][3][1];
        buf0[23] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 1) * c[4][3][1];
        buf0[15] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 1) * c[0][3][2];
        buf0[15] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 1) * c[1][3][2];
        buf0[15] += bIn(i + hipThreadIdx_x, j + 8, k + 1) * c[2][3][2];
        buf0[15] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 1) * c[3][3][2];
        buf0[15] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 1) * c[4][3][2];
        buf0[7] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 1) * c[0][3][3];
        buf0[7] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 1) * c[1][3][3];
        buf0[7] += bIn(i + hipThreadIdx_x, j + 8, k + 1) * c[2][3][3];
        buf0[7] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 1) * c[3][3][3];
        buf0[7] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 1) * c[4][3][3];
        buf0[30] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 1) * c[0][4][0];
        buf0[30] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 1) * c[1][4][0];
        buf0[30] += bIn(i + hipThreadIdx_x, j + 8, k + 1) * c[2][4][0];
        buf0[30] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 1) * c[3][4][0];
        buf0[30] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 1) * c[4][4][0];
        buf0[22] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 1) * c[0][4][1];
        buf0[22] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 1) * c[1][4][1];
        buf0[22] += bIn(i + hipThreadIdx_x, j + 8, k + 1) * c[2][4][1];
        buf0[22] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 1) * c[3][4][1];
        buf0[22] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 1) * c[4][4][1];
        buf0[14] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 1) * c[0][4][2];
        buf0[14] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 1) * c[1][4][2];
        buf0[14] += bIn(i + hipThreadIdx_x, j + 8, k + 1) * c[2][4][2];
        buf0[14] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 1) * c[3][4][2];
        buf0[14] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 1) * c[4][4][2];
        buf0[6] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 1) * c[0][4][3];
        buf0[6] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 1) * c[1][4][3];
        buf0[6] += bIn(i + hipThreadIdx_x, j + 8, k + 1) * c[2][4][3];
        buf0[6] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 1) * c[3][4][3];
        buf0[6] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 1) * c[4][4][3];
      }
      {
        buf0[31] += bIn(i + hipThreadIdx_x + -2, j + 9, k + 1) * c[0][4][0];
        buf0[31] += bIn(i + hipThreadIdx_x + -1, j + 9, k + 1) * c[1][4][0];
        buf0[31] += bIn(i + hipThreadIdx_x, j + 9, k + 1) * c[2][4][0];
        buf0[31] += bIn(i + hipThreadIdx_x + 1, j + 9, k + 1) * c[3][4][0];
        buf0[31] += bIn(i + hipThreadIdx_x + 2, j + 9, k + 1) * c[4][4][0];
        buf0[23] += bIn(i + hipThreadIdx_x + -2, j + 9, k + 1) * c[0][4][1];
        buf0[23] += bIn(i + hipThreadIdx_x + -1, j + 9, k + 1) * c[1][4][1];
        buf0[23] += bIn(i + hipThreadIdx_x, j + 9, k + 1) * c[2][4][1];
        buf0[23] += bIn(i + hipThreadIdx_x + 1, j + 9, k + 1) * c[3][4][1];
        buf0[23] += bIn(i + hipThreadIdx_x + 2, j + 9, k + 1) * c[4][4][1];
        buf0[15] += bIn(i + hipThreadIdx_x + -2, j + 9, k + 1) * c[0][4][2];
        buf0[15] += bIn(i + hipThreadIdx_x + -1, j + 9, k + 1) * c[1][4][2];
        buf0[15] += bIn(i + hipThreadIdx_x, j + 9, k + 1) * c[2][4][2];
        buf0[15] += bIn(i + hipThreadIdx_x + 1, j + 9, k + 1) * c[3][4][2];
        buf0[15] += bIn(i + hipThreadIdx_x + 2, j + 9, k + 1) * c[4][4][2];
        buf0[7] += bIn(i + hipThreadIdx_x + -2, j + 9, k + 1) * c[0][4][3];
        buf0[7] += bIn(i + hipThreadIdx_x + -1, j + 9, k + 1) * c[1][4][3];
        buf0[7] += bIn(i + hipThreadIdx_x, j + 9, k + 1) * c[2][4][3];
        buf0[7] += bIn(i + hipThreadIdx_x + 1, j + 9, k + 1) * c[3][4][3];
        buf0[7] += bIn(i + hipThreadIdx_x + 2, j + 9, k + 1) * c[4][4][3];
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 4; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            buf0[32 + rel] += 0;
            buf0[32 + rel] += bIn(i + hipThreadIdx_x + -2, j + -2, k + _cg_idx2 + 2) * c[0][0][0];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x + -1, j + -2, k + _cg_idx2 + 2) * c[1][0][0];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x, j + -2, k + _cg_idx2 + 2) * c[2][0][0];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x + 1, j + -2, k + _cg_idx2 + 2) * c[3][0][0];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x + 2, j + -2, k + _cg_idx2 + 2) * c[4][0][0];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x + -2, j + -2, k + _cg_idx2 + 2) * c[0][0][1];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x + -1, j + -2, k + _cg_idx2 + 2) * c[1][0][1];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x, j + -2, k + _cg_idx2 + 2) * c[2][0][1];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x + 1, j + -2, k + _cg_idx2 + 2) * c[3][0][1];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x + 2, j + -2, k + _cg_idx2 + 2) * c[4][0][1];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + -2, j + -2, k + _cg_idx2 + 2) * c[0][0][2];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + -1, j + -2, k + _cg_idx2 + 2) * c[1][0][2];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x, j + -2, k + _cg_idx2 + 2) * c[2][0][2];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + 1, j + -2, k + _cg_idx2 + 2) * c[3][0][2];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + 2, j + -2, k + _cg_idx2 + 2) * c[4][0][2];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + -2, j + -2, k + _cg_idx2 + 2) * c[0][0][3];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + -1, j + -2, k + _cg_idx2 + 2) * c[1][0][3];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x, j + -2, k + _cg_idx2 + 2) * c[2][0][3];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + 1, j + -2, k + _cg_idx2 + 2) * c[3][0][3];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + 2, j + -2, k + _cg_idx2 + 2) * c[4][0][3];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + -2, j + -2, k + _cg_idx2 + 2) * c[0][0][4];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + -1, j + -2, k + _cg_idx2 + 2) * c[1][0][4];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x, j + -2, k + _cg_idx2 + 2) * c[2][0][4];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + 1, j + -2, k + _cg_idx2 + 2) * c[3][0][4];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + 2, j + -2, k + _cg_idx2 + 2) * c[4][0][4];
          }
          {
            buf0[33 + rel] += 0;
            buf0[33 + rel] += bIn(i + hipThreadIdx_x + -2, j + -1, k + _cg_idx2 + 2) * c[0][0][0];
            buf0[33 + rel] += bIn(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 2) * c[1][0][0];
            buf0[33 + rel] += bIn(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2) * c[2][0][0];
            buf0[33 + rel] += bIn(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 2) * c[3][0][0];
            buf0[33 + rel] += bIn(i + hipThreadIdx_x + 2, j + -1, k + _cg_idx2 + 2) * c[4][0][0];
            buf0[25 + rel] += bIn(i + hipThreadIdx_x + -2, j + -1, k + _cg_idx2 + 2) * c[0][0][1];
            buf0[25 + rel] += bIn(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 2) * c[1][0][1];
            buf0[25 + rel] += bIn(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2) * c[2][0][1];
            buf0[25 + rel] += bIn(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 2) * c[3][0][1];
            buf0[25 + rel] += bIn(i + hipThreadIdx_x + 2, j + -1, k + _cg_idx2 + 2) * c[4][0][1];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + -2, j + -1, k + _cg_idx2 + 2) * c[0][0][2];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 2) * c[1][0][2];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2) * c[2][0][2];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 2) * c[3][0][2];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + 2, j + -1, k + _cg_idx2 + 2) * c[4][0][2];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + -2, j + -1, k + _cg_idx2 + 2) * c[0][0][3];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 2) * c[1][0][3];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2) * c[2][0][3];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 2) * c[3][0][3];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + 2, j + -1, k + _cg_idx2 + 2) * c[4][0][3];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + -2, j + -1, k + _cg_idx2 + 2) * c[0][0][4];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 2) * c[1][0][4];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2) * c[2][0][4];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 2) * c[3][0][4];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + 2, j + -1, k + _cg_idx2 + 2) * c[4][0][4];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x + -2, j + -1, k + _cg_idx2 + 2) * c[0][1][0];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 2) * c[1][1][0];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2) * c[2][1][0];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 2) * c[3][1][0];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x + 2, j + -1, k + _cg_idx2 + 2) * c[4][1][0];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x + -2, j + -1, k + _cg_idx2 + 2) * c[0][1][1];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 2) * c[1][1][1];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2) * c[2][1][1];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 2) * c[3][1][1];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x + 2, j + -1, k + _cg_idx2 + 2) * c[4][1][1];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + -2, j + -1, k + _cg_idx2 + 2) * c[0][1][2];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 2) * c[1][1][2];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2) * c[2][1][2];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 2) * c[3][1][2];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + 2, j + -1, k + _cg_idx2 + 2) * c[4][1][2];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + -2, j + -1, k + _cg_idx2 + 2) * c[0][1][3];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 2) * c[1][1][3];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2) * c[2][1][3];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 2) * c[3][1][3];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + 2, j + -1, k + _cg_idx2 + 2) * c[4][1][3];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + -2, j + -1, k + _cg_idx2 + 2) * c[0][1][4];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + -1, j + -1, k + _cg_idx2 + 2) * c[1][1][4];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2) * c[2][1][4];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + 1, j + -1, k + _cg_idx2 + 2) * c[3][1][4];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + 2, j + -1, k + _cg_idx2 + 2) * c[4][1][4];
          }
          {
            buf0[34 + rel] += 0;
            buf0[34 + rel] += bIn(i + hipThreadIdx_x + -2, j, k + _cg_idx2 + 2) * c[0][0][0];
            buf0[34 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2) * c[1][0][0];
            buf0[34 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2) * c[2][0][0];
            buf0[34 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2) * c[3][0][0];
            buf0[34 + rel] += bIn(i + hipThreadIdx_x + 2, j, k + _cg_idx2 + 2) * c[4][0][0];
            buf0[26 + rel] += bIn(i + hipThreadIdx_x + -2, j, k + _cg_idx2 + 2) * c[0][0][1];
            buf0[26 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2) * c[1][0][1];
            buf0[26 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2) * c[2][0][1];
            buf0[26 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2) * c[3][0][1];
            buf0[26 + rel] += bIn(i + hipThreadIdx_x + 2, j, k + _cg_idx2 + 2) * c[4][0][1];
            buf0[18 + rel] += bIn(i + hipThreadIdx_x + -2, j, k + _cg_idx2 + 2) * c[0][0][2];
            buf0[18 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2) * c[1][0][2];
            buf0[18 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2) * c[2][0][2];
            buf0[18 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2) * c[3][0][2];
            buf0[18 + rel] += bIn(i + hipThreadIdx_x + 2, j, k + _cg_idx2 + 2) * c[4][0][2];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + -2, j, k + _cg_idx2 + 2) * c[0][0][3];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2) * c[1][0][3];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2) * c[2][0][3];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2) * c[3][0][3];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + 2, j, k + _cg_idx2 + 2) * c[4][0][3];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + -2, j, k + _cg_idx2 + 2) * c[0][0][4];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2) * c[1][0][4];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2) * c[2][0][4];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2) * c[3][0][4];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + 2, j, k + _cg_idx2 + 2) * c[4][0][4];
            buf0[33 + rel] += bIn(i + hipThreadIdx_x + -2, j, k + _cg_idx2 + 2) * c[0][1][0];
            buf0[33 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2) * c[1][1][0];
            buf0[33 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2) * c[2][1][0];
            buf0[33 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2) * c[3][1][0];
            buf0[33 + rel] += bIn(i + hipThreadIdx_x + 2, j, k + _cg_idx2 + 2) * c[4][1][0];
            buf0[25 + rel] += bIn(i + hipThreadIdx_x + -2, j, k + _cg_idx2 + 2) * c[0][1][1];
            buf0[25 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2) * c[1][1][1];
            buf0[25 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2) * c[2][1][1];
            buf0[25 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2) * c[3][1][1];
            buf0[25 + rel] += bIn(i + hipThreadIdx_x + 2, j, k + _cg_idx2 + 2) * c[4][1][1];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + -2, j, k + _cg_idx2 + 2) * c[0][1][2];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2) * c[1][1][2];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2) * c[2][1][2];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2) * c[3][1][2];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + 2, j, k + _cg_idx2 + 2) * c[4][1][2];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + -2, j, k + _cg_idx2 + 2) * c[0][1][3];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2) * c[1][1][3];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2) * c[2][1][3];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2) * c[3][1][3];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + 2, j, k + _cg_idx2 + 2) * c[4][1][3];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + -2, j, k + _cg_idx2 + 2) * c[0][1][4];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2) * c[1][1][4];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2) * c[2][1][4];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2) * c[3][1][4];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + 2, j, k + _cg_idx2 + 2) * c[4][1][4];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x + -2, j, k + _cg_idx2 + 2) * c[0][2][0];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2) * c[1][2][0];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2) * c[2][2][0];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2) * c[3][2][0];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x + 2, j, k + _cg_idx2 + 2) * c[4][2][0];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x + -2, j, k + _cg_idx2 + 2) * c[0][2][1];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2) * c[1][2][1];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2) * c[2][2][1];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2) * c[3][2][1];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x + 2, j, k + _cg_idx2 + 2) * c[4][2][1];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + -2, j, k + _cg_idx2 + 2) * c[0][2][2];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2) * c[1][2][2];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2) * c[2][2][2];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2) * c[3][2][2];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + 2, j, k + _cg_idx2 + 2) * c[4][2][2];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + -2, j, k + _cg_idx2 + 2) * c[0][2][3];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2) * c[1][2][3];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2) * c[2][2][3];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2) * c[3][2][3];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + 2, j, k + _cg_idx2 + 2) * c[4][2][3];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + -2, j, k + _cg_idx2 + 2) * c[0][2][4];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2) * c[1][2][4];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2) * c[2][2][4];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2) * c[3][2][4];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + 2, j, k + _cg_idx2 + 2) * c[4][2][4];
          }
          {
            buf0[35 + rel] += 0;
            buf0[35 + rel] += bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2) * c[0][0][0];
            buf0[35 + rel] += bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2) * c[1][0][0];
            buf0[35 + rel] += bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2) * c[2][0][0];
            buf0[35 + rel] += bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2) * c[3][0][0];
            buf0[35 + rel] += bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2) * c[4][0][0];
            buf0[27 + rel] += bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2) * c[0][0][1];
            buf0[27 + rel] += bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2) * c[1][0][1];
            buf0[27 + rel] += bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2) * c[2][0][1];
            buf0[27 + rel] += bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2) * c[3][0][1];
            buf0[27 + rel] += bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2) * c[4][0][1];
            buf0[19 + rel] += bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2) * c[0][0][2];
            buf0[19 + rel] += bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2) * c[1][0][2];
            buf0[19 + rel] += bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2) * c[2][0][2];
            buf0[19 + rel] += bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2) * c[3][0][2];
            buf0[19 + rel] += bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2) * c[4][0][2];
            buf0[11 + rel] += bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2) * c[0][0][3];
            buf0[11 + rel] += bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2) * c[1][0][3];
            buf0[11 + rel] += bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2) * c[2][0][3];
            buf0[11 + rel] += bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2) * c[3][0][3];
            buf0[11 + rel] += bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2) * c[4][0][3];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2) * c[0][0][4];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2) * c[1][0][4];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2) * c[2][0][4];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2) * c[3][0][4];
            buf0[3 + rel] += bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2) * c[4][0][4];
            buf0[34 + rel] += bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2) * c[0][1][0];
            buf0[34 + rel] += bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2) * c[1][1][0];
            buf0[34 + rel] += bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2) * c[2][1][0];
            buf0[34 + rel] += bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2) * c[3][1][0];
            buf0[34 + rel] += bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2) * c[4][1][0];
            buf0[26 + rel] += bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2) * c[0][1][1];
            buf0[26 + rel] += bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2) * c[1][1][1];
            buf0[26 + rel] += bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2) * c[2][1][1];
            buf0[26 + rel] += bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2) * c[3][1][1];
            buf0[26 + rel] += bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2) * c[4][1][1];
            buf0[18 + rel] += bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2) * c[0][1][2];
            buf0[18 + rel] += bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2) * c[1][1][2];
            buf0[18 + rel] += bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2) * c[2][1][2];
            buf0[18 + rel] += bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2) * c[3][1][2];
            buf0[18 + rel] += bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2) * c[4][1][2];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2) * c[0][1][3];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2) * c[1][1][3];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2) * c[2][1][3];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2) * c[3][1][3];
            buf0[10 + rel] += bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2) * c[4][1][3];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2) * c[0][1][4];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2) * c[1][1][4];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2) * c[2][1][4];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2) * c[3][1][4];
            buf0[2 + rel] += bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2) * c[4][1][4];
            buf0[33 + rel] += bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2) * c[0][2][0];
            buf0[33 + rel] += bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2) * c[1][2][0];
            buf0[33 + rel] += bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2) * c[2][2][0];
            buf0[33 + rel] += bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2) * c[3][2][0];
            buf0[33 + rel] += bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2) * c[4][2][0];
            buf0[25 + rel] += bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2) * c[0][2][1];
            buf0[25 + rel] += bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2) * c[1][2][1];
            buf0[25 + rel] += bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2) * c[2][2][1];
            buf0[25 + rel] += bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2) * c[3][2][1];
            buf0[25 + rel] += bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2) * c[4][2][1];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2) * c[0][2][2];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2) * c[1][2][2];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2) * c[2][2][2];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2) * c[3][2][2];
            buf0[17 + rel] += bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2) * c[4][2][2];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2) * c[0][2][3];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2) * c[1][2][3];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2) * c[2][2][3];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2) * c[3][2][3];
            buf0[9 + rel] += bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2) * c[4][2][3];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2) * c[0][2][4];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2) * c[1][2][4];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2) * c[2][2][4];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2) * c[3][2][4];
            buf0[1 + rel] += bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2) * c[4][2][4];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2) * c[0][3][0];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2) * c[1][3][0];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2) * c[2][3][0];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2) * c[3][3][0];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2) * c[4][3][0];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2) * c[0][3][1];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2) * c[1][3][1];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2) * c[2][3][1];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2) * c[3][3][1];
            buf0[24 + rel] += bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2) * c[4][3][1];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2) * c[0][3][2];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2) * c[1][3][2];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2) * c[2][3][2];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2) * c[3][3][2];
            buf0[16 + rel] += bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2) * c[4][3][2];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2) * c[0][3][3];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2) * c[1][3][3];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2) * c[2][3][3];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2) * c[3][3][3];
            buf0[8 + rel] += bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2) * c[4][3][3];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2) * c[0][3][4];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2) * c[1][3][4];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2) * c[2][3][4];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2) * c[3][3][4];
            buf0[0 + rel] += bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2) * c[4][3][4];
          }
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 4; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf0[36 + rel] += 0;
                buf0[36 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][0][0];
                buf0[36 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][0][0];
                buf0[36 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][0][0];
                buf0[36 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][0][0];
                buf0[36 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][0][0];
                buf0[28 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][0][1];
                buf0[28 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][0][1];
                buf0[28 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][0][1];
                buf0[28 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][0][1];
                buf0[28 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][0][1];
                buf0[20 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][0][2];
                buf0[20 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][0][2];
                buf0[20 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][0][2];
                buf0[20 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][0][2];
                buf0[20 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][0][2];
                buf0[12 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][0][3];
                buf0[12 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][0][3];
                buf0[12 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][0][3];
                buf0[12 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][0][3];
                buf0[12 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][0][3];
                buf0[4 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][0][4];
                buf0[4 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][0][4];
                buf0[4 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][0][4];
                buf0[4 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][0][4];
                buf0[4 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][0][4];
                buf0[35 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][1][0];
                buf0[35 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][1][0];
                buf0[35 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][1][0];
                buf0[35 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][1][0];
                buf0[35 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][1][0];
                buf0[27 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][1][1];
                buf0[27 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][1][1];
                buf0[27 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][1][1];
                buf0[27 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][1][1];
                buf0[27 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][1][1];
                buf0[19 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][1][2];
                buf0[19 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][1][2];
                buf0[19 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][1][2];
                buf0[19 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][1][2];
                buf0[19 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][1][2];
                buf0[11 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][1][3];
                buf0[11 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][1][3];
                buf0[11 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][1][3];
                buf0[11 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][1][3];
                buf0[11 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][1][3];
                buf0[3 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][1][4];
                buf0[3 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][1][4];
                buf0[3 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][1][4];
                buf0[3 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][1][4];
                buf0[3 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][1][4];
                buf0[34 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][2][0];
                buf0[34 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][2][0];
                buf0[34 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][2][0];
                buf0[34 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][2][0];
                buf0[34 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][2][0];
                buf0[26 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][2][1];
                buf0[26 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][2][1];
                buf0[26 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][2][1];
                buf0[26 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][2][1];
                buf0[26 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][2][1];
                buf0[18 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][2][2];
                buf0[18 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][2][2];
                buf0[18 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][2][2];
                buf0[18 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][2][2];
                buf0[18 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][2][2];
                buf0[10 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][2][3];
                buf0[10 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][2][3];
                buf0[10 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][2][3];
                buf0[10 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][2][3];
                buf0[10 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][2][3];
                buf0[2 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][2][4];
                buf0[2 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][2][4];
                buf0[2 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][2][4];
                buf0[2 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][2][4];
                buf0[2 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][2][4];
                buf0[33 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][3][0];
                buf0[33 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][3][0];
                buf0[33 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][3][0];
                buf0[33 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][3][0];
                buf0[33 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][3][0];
                buf0[25 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][3][1];
                buf0[25 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][3][1];
                buf0[25 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][3][1];
                buf0[25 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][3][1];
                buf0[25 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][3][1];
                buf0[17 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][3][2];
                buf0[17 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][3][2];
                buf0[17 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][3][2];
                buf0[17 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][3][2];
                buf0[17 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][3][2];
                buf0[9 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][3][3];
                buf0[9 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][3][3];
                buf0[9 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][3][3];
                buf0[9 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][3][3];
                buf0[9 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][3][3];
                buf0[1 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][3][4];
                buf0[1 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][3][4];
                buf0[1 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][3][4];
                buf0[1 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][3][4];
                buf0[1 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][3][4];
                buf0[32 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][4][0];
                buf0[32 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][4][0];
                buf0[32 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][4][0];
                buf0[32 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][4][0];
                buf0[32 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][4][0];
                buf0[24 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][4][1];
                buf0[24 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][4][1];
                buf0[24 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][4][1];
                buf0[24 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][4][1];
                buf0[24 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][4][1];
                buf0[16 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][4][2];
                buf0[16 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][4][2];
                buf0[16 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][4][2];
                buf0[16 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][4][2];
                buf0[16 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][4][2];
                buf0[8 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][4][3];
                buf0[8 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][4][3];
                buf0[8 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][4][3];
                buf0[8 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][4][3];
                buf0[8 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][4][3];
                buf0[0 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[0][4][4];
                buf0[0 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[1][4][4];
                buf0[0 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[2][4][4];
                buf0[0 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[3][4][4];
                buf0[0 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2) * c[4][4][4];
              }
              _cg_rel1 += 1;
            }
          }
          {
            buf0[39 + rel] += bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2) * c[0][1][0];
            buf0[39 + rel] += bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2) * c[1][1][0];
            buf0[39 + rel] += bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2) * c[2][1][0];
            buf0[39 + rel] += bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2) * c[3][1][0];
            buf0[39 + rel] += bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2) * c[4][1][0];
            buf0[31 + rel] += bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2) * c[0][1][1];
            buf0[31 + rel] += bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2) * c[1][1][1];
            buf0[31 + rel] += bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2) * c[2][1][1];
            buf0[31 + rel] += bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2) * c[3][1][1];
            buf0[31 + rel] += bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2) * c[4][1][1];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2) * c[0][1][2];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2) * c[1][1][2];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2) * c[2][1][2];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2) * c[3][1][2];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2) * c[4][1][2];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2) * c[0][1][3];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2) * c[1][1][3];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2) * c[2][1][3];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2) * c[3][1][3];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2) * c[4][1][3];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2) * c[0][1][4];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2) * c[1][1][4];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2) * c[2][1][4];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2) * c[3][1][4];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2) * c[4][1][4];
            buf0[38 + rel] += bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2) * c[0][2][0];
            buf0[38 + rel] += bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2) * c[1][2][0];
            buf0[38 + rel] += bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2) * c[2][2][0];
            buf0[38 + rel] += bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2) * c[3][2][0];
            buf0[38 + rel] += bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2) * c[4][2][0];
            buf0[30 + rel] += bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2) * c[0][2][1];
            buf0[30 + rel] += bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2) * c[1][2][1];
            buf0[30 + rel] += bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2) * c[2][2][1];
            buf0[30 + rel] += bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2) * c[3][2][1];
            buf0[30 + rel] += bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2) * c[4][2][1];
            buf0[22 + rel] += bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2) * c[0][2][2];
            buf0[22 + rel] += bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2) * c[1][2][2];
            buf0[22 + rel] += bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2) * c[2][2][2];
            buf0[22 + rel] += bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2) * c[3][2][2];
            buf0[22 + rel] += bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2) * c[4][2][2];
            buf0[14 + rel] += bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2) * c[0][2][3];
            buf0[14 + rel] += bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2) * c[1][2][3];
            buf0[14 + rel] += bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2) * c[2][2][3];
            buf0[14 + rel] += bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2) * c[3][2][3];
            buf0[14 + rel] += bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2) * c[4][2][3];
            buf0[6 + rel] += bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2) * c[0][2][4];
            buf0[6 + rel] += bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2) * c[1][2][4];
            buf0[6 + rel] += bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2) * c[2][2][4];
            buf0[6 + rel] += bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2) * c[3][2][4];
            buf0[6 + rel] += bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2) * c[4][2][4];
            buf0[37 + rel] += bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2) * c[0][3][0];
            buf0[37 + rel] += bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2) * c[1][3][0];
            buf0[37 + rel] += bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2) * c[2][3][0];
            buf0[37 + rel] += bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2) * c[3][3][0];
            buf0[37 + rel] += bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2) * c[4][3][0];
            buf0[29 + rel] += bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2) * c[0][3][1];
            buf0[29 + rel] += bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2) * c[1][3][1];
            buf0[29 + rel] += bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2) * c[2][3][1];
            buf0[29 + rel] += bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2) * c[3][3][1];
            buf0[29 + rel] += bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2) * c[4][3][1];
            buf0[21 + rel] += bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2) * c[0][3][2];
            buf0[21 + rel] += bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2) * c[1][3][2];
            buf0[21 + rel] += bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2) * c[2][3][2];
            buf0[21 + rel] += bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2) * c[3][3][2];
            buf0[21 + rel] += bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2) * c[4][3][2];
            buf0[13 + rel] += bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2) * c[0][3][3];
            buf0[13 + rel] += bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2) * c[1][3][3];
            buf0[13 + rel] += bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2) * c[2][3][3];
            buf0[13 + rel] += bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2) * c[3][3][3];
            buf0[13 + rel] += bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2) * c[4][3][3];
            buf0[5 + rel] += bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2) * c[0][3][4];
            buf0[5 + rel] += bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2) * c[1][3][4];
            buf0[5 + rel] += bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2) * c[2][3][4];
            buf0[5 + rel] += bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2) * c[3][3][4];
            buf0[5 + rel] += bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2) * c[4][3][4];
            buf0[36 + rel] += bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2) * c[0][4][0];
            buf0[36 + rel] += bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2) * c[1][4][0];
            buf0[36 + rel] += bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2) * c[2][4][0];
            buf0[36 + rel] += bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2) * c[3][4][0];
            buf0[36 + rel] += bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2) * c[4][4][0];
            buf0[28 + rel] += bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2) * c[0][4][1];
            buf0[28 + rel] += bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2) * c[1][4][1];
            buf0[28 + rel] += bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2) * c[2][4][1];
            buf0[28 + rel] += bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2) * c[3][4][1];
            buf0[28 + rel] += bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2) * c[4][4][1];
            buf0[20 + rel] += bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2) * c[0][4][2];
            buf0[20 + rel] += bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2) * c[1][4][2];
            buf0[20 + rel] += bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2) * c[2][4][2];
            buf0[20 + rel] += bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2) * c[3][4][2];
            buf0[20 + rel] += bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2) * c[4][4][2];
            buf0[12 + rel] += bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2) * c[0][4][3];
            buf0[12 + rel] += bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2) * c[1][4][3];
            buf0[12 + rel] += bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2) * c[2][4][3];
            buf0[12 + rel] += bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2) * c[3][4][3];
            buf0[12 + rel] += bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2) * c[4][4][3];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2) * c[0][4][4];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2) * c[1][4][4];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2) * c[2][4][4];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2) * c[3][4][4];
            buf0[4 + rel] += bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2) * c[4][4][4];
          }
          {
            buf0[39 + rel] += bIn(i + hipThreadIdx_x + -2, j + 7, k + _cg_idx2 + 2) * c[0][2][0];
            buf0[39 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2) * c[1][2][0];
            buf0[39 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2) * c[2][2][0];
            buf0[39 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2) * c[3][2][0];
            buf0[39 + rel] += bIn(i + hipThreadIdx_x + 2, j + 7, k + _cg_idx2 + 2) * c[4][2][0];
            buf0[31 + rel] += bIn(i + hipThreadIdx_x + -2, j + 7, k + _cg_idx2 + 2) * c[0][2][1];
            buf0[31 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2) * c[1][2][1];
            buf0[31 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2) * c[2][2][1];
            buf0[31 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2) * c[3][2][1];
            buf0[31 + rel] += bIn(i + hipThreadIdx_x + 2, j + 7, k + _cg_idx2 + 2) * c[4][2][1];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x + -2, j + 7, k + _cg_idx2 + 2) * c[0][2][2];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2) * c[1][2][2];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2) * c[2][2][2];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2) * c[3][2][2];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x + 2, j + 7, k + _cg_idx2 + 2) * c[4][2][2];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x + -2, j + 7, k + _cg_idx2 + 2) * c[0][2][3];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2) * c[1][2][3];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2) * c[2][2][3];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2) * c[3][2][3];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x + 2, j + 7, k + _cg_idx2 + 2) * c[4][2][3];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x + -2, j + 7, k + _cg_idx2 + 2) * c[0][2][4];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2) * c[1][2][4];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2) * c[2][2][4];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2) * c[3][2][4];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x + 2, j + 7, k + _cg_idx2 + 2) * c[4][2][4];
            buf0[38 + rel] += bIn(i + hipThreadIdx_x + -2, j + 7, k + _cg_idx2 + 2) * c[0][3][0];
            buf0[38 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2) * c[1][3][0];
            buf0[38 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2) * c[2][3][0];
            buf0[38 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2) * c[3][3][0];
            buf0[38 + rel] += bIn(i + hipThreadIdx_x + 2, j + 7, k + _cg_idx2 + 2) * c[4][3][0];
            buf0[30 + rel] += bIn(i + hipThreadIdx_x + -2, j + 7, k + _cg_idx2 + 2) * c[0][3][1];
            buf0[30 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2) * c[1][3][1];
            buf0[30 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2) * c[2][3][1];
            buf0[30 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2) * c[3][3][1];
            buf0[30 + rel] += bIn(i + hipThreadIdx_x + 2, j + 7, k + _cg_idx2 + 2) * c[4][3][1];
            buf0[22 + rel] += bIn(i + hipThreadIdx_x + -2, j + 7, k + _cg_idx2 + 2) * c[0][3][2];
            buf0[22 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2) * c[1][3][2];
            buf0[22 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2) * c[2][3][2];
            buf0[22 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2) * c[3][3][2];
            buf0[22 + rel] += bIn(i + hipThreadIdx_x + 2, j + 7, k + _cg_idx2 + 2) * c[4][3][2];
            buf0[14 + rel] += bIn(i + hipThreadIdx_x + -2, j + 7, k + _cg_idx2 + 2) * c[0][3][3];
            buf0[14 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2) * c[1][3][3];
            buf0[14 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2) * c[2][3][3];
            buf0[14 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2) * c[3][3][3];
            buf0[14 + rel] += bIn(i + hipThreadIdx_x + 2, j + 7, k + _cg_idx2 + 2) * c[4][3][3];
            buf0[6 + rel] += bIn(i + hipThreadIdx_x + -2, j + 7, k + _cg_idx2 + 2) * c[0][3][4];
            buf0[6 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2) * c[1][3][4];
            buf0[6 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2) * c[2][3][4];
            buf0[6 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2) * c[3][3][4];
            buf0[6 + rel] += bIn(i + hipThreadIdx_x + 2, j + 7, k + _cg_idx2 + 2) * c[4][3][4];
            buf0[37 + rel] += bIn(i + hipThreadIdx_x + -2, j + 7, k + _cg_idx2 + 2) * c[0][4][0];
            buf0[37 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2) * c[1][4][0];
            buf0[37 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2) * c[2][4][0];
            buf0[37 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2) * c[3][4][0];
            buf0[37 + rel] += bIn(i + hipThreadIdx_x + 2, j + 7, k + _cg_idx2 + 2) * c[4][4][0];
            buf0[29 + rel] += bIn(i + hipThreadIdx_x + -2, j + 7, k + _cg_idx2 + 2) * c[0][4][1];
            buf0[29 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2) * c[1][4][1];
            buf0[29 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2) * c[2][4][1];
            buf0[29 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2) * c[3][4][1];
            buf0[29 + rel] += bIn(i + hipThreadIdx_x + 2, j + 7, k + _cg_idx2 + 2) * c[4][4][1];
            buf0[21 + rel] += bIn(i + hipThreadIdx_x + -2, j + 7, k + _cg_idx2 + 2) * c[0][4][2];
            buf0[21 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2) * c[1][4][2];
            buf0[21 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2) * c[2][4][2];
            buf0[21 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2) * c[3][4][2];
            buf0[21 + rel] += bIn(i + hipThreadIdx_x + 2, j + 7, k + _cg_idx2 + 2) * c[4][4][2];
            buf0[13 + rel] += bIn(i + hipThreadIdx_x + -2, j + 7, k + _cg_idx2 + 2) * c[0][4][3];
            buf0[13 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2) * c[1][4][3];
            buf0[13 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2) * c[2][4][3];
            buf0[13 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2) * c[3][4][3];
            buf0[13 + rel] += bIn(i + hipThreadIdx_x + 2, j + 7, k + _cg_idx2 + 2) * c[4][4][3];
            buf0[5 + rel] += bIn(i + hipThreadIdx_x + -2, j + 7, k + _cg_idx2 + 2) * c[0][4][4];
            buf0[5 + rel] += bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2) * c[1][4][4];
            buf0[5 + rel] += bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2) * c[2][4][4];
            buf0[5 + rel] += bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2) * c[3][4][4];
            buf0[5 + rel] += bIn(i + hipThreadIdx_x + 2, j + 7, k + _cg_idx2 + 2) * c[4][4][4];
          }
          {
            buf0[39 + rel] += bIn(i + hipThreadIdx_x + -2, j + 8, k + _cg_idx2 + 2) * c[0][3][0];
            buf0[39 + rel] += bIn(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 2) * c[1][3][0];
            buf0[39 + rel] += bIn(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2) * c[2][3][0];
            buf0[39 + rel] += bIn(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 2) * c[3][3][0];
            buf0[39 + rel] += bIn(i + hipThreadIdx_x + 2, j + 8, k + _cg_idx2 + 2) * c[4][3][0];
            buf0[31 + rel] += bIn(i + hipThreadIdx_x + -2, j + 8, k + _cg_idx2 + 2) * c[0][3][1];
            buf0[31 + rel] += bIn(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 2) * c[1][3][1];
            buf0[31 + rel] += bIn(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2) * c[2][3][1];
            buf0[31 + rel] += bIn(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 2) * c[3][3][1];
            buf0[31 + rel] += bIn(i + hipThreadIdx_x + 2, j + 8, k + _cg_idx2 + 2) * c[4][3][1];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x + -2, j + 8, k + _cg_idx2 + 2) * c[0][3][2];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 2) * c[1][3][2];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2) * c[2][3][2];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 2) * c[3][3][2];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x + 2, j + 8, k + _cg_idx2 + 2) * c[4][3][2];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x + -2, j + 8, k + _cg_idx2 + 2) * c[0][3][3];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 2) * c[1][3][3];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2) * c[2][3][3];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 2) * c[3][3][3];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x + 2, j + 8, k + _cg_idx2 + 2) * c[4][3][3];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x + -2, j + 8, k + _cg_idx2 + 2) * c[0][3][4];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 2) * c[1][3][4];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2) * c[2][3][4];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 2) * c[3][3][4];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x + 2, j + 8, k + _cg_idx2 + 2) * c[4][3][4];
            buf0[38 + rel] += bIn(i + hipThreadIdx_x + -2, j + 8, k + _cg_idx2 + 2) * c[0][4][0];
            buf0[38 + rel] += bIn(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 2) * c[1][4][0];
            buf0[38 + rel] += bIn(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2) * c[2][4][0];
            buf0[38 + rel] += bIn(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 2) * c[3][4][0];
            buf0[38 + rel] += bIn(i + hipThreadIdx_x + 2, j + 8, k + _cg_idx2 + 2) * c[4][4][0];
            buf0[30 + rel] += bIn(i + hipThreadIdx_x + -2, j + 8, k + _cg_idx2 + 2) * c[0][4][1];
            buf0[30 + rel] += bIn(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 2) * c[1][4][1];
            buf0[30 + rel] += bIn(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2) * c[2][4][1];
            buf0[30 + rel] += bIn(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 2) * c[3][4][1];
            buf0[30 + rel] += bIn(i + hipThreadIdx_x + 2, j + 8, k + _cg_idx2 + 2) * c[4][4][1];
            buf0[22 + rel] += bIn(i + hipThreadIdx_x + -2, j + 8, k + _cg_idx2 + 2) * c[0][4][2];
            buf0[22 + rel] += bIn(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 2) * c[1][4][2];
            buf0[22 + rel] += bIn(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2) * c[2][4][2];
            buf0[22 + rel] += bIn(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 2) * c[3][4][2];
            buf0[22 + rel] += bIn(i + hipThreadIdx_x + 2, j + 8, k + _cg_idx2 + 2) * c[4][4][2];
            buf0[14 + rel] += bIn(i + hipThreadIdx_x + -2, j + 8, k + _cg_idx2 + 2) * c[0][4][3];
            buf0[14 + rel] += bIn(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 2) * c[1][4][3];
            buf0[14 + rel] += bIn(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2) * c[2][4][3];
            buf0[14 + rel] += bIn(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 2) * c[3][4][3];
            buf0[14 + rel] += bIn(i + hipThreadIdx_x + 2, j + 8, k + _cg_idx2 + 2) * c[4][4][3];
            buf0[6 + rel] += bIn(i + hipThreadIdx_x + -2, j + 8, k + _cg_idx2 + 2) * c[0][4][4];
            buf0[6 + rel] += bIn(i + hipThreadIdx_x + -1, j + 8, k + _cg_idx2 + 2) * c[1][4][4];
            buf0[6 + rel] += bIn(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2) * c[2][4][4];
            buf0[6 + rel] += bIn(i + hipThreadIdx_x + 1, j + 8, k + _cg_idx2 + 2) * c[3][4][4];
            buf0[6 + rel] += bIn(i + hipThreadIdx_x + 2, j + 8, k + _cg_idx2 + 2) * c[4][4][4];
          }
          {
            buf0[39 + rel] += bIn(i + hipThreadIdx_x + -2, j + 9, k + _cg_idx2 + 2) * c[0][4][0];
            buf0[39 + rel] += bIn(i + hipThreadIdx_x + -1, j + 9, k + _cg_idx2 + 2) * c[1][4][0];
            buf0[39 + rel] += bIn(i + hipThreadIdx_x, j + 9, k + _cg_idx2 + 2) * c[2][4][0];
            buf0[39 + rel] += bIn(i + hipThreadIdx_x + 1, j + 9, k + _cg_idx2 + 2) * c[3][4][0];
            buf0[39 + rel] += bIn(i + hipThreadIdx_x + 2, j + 9, k + _cg_idx2 + 2) * c[4][4][0];
            buf0[31 + rel] += bIn(i + hipThreadIdx_x + -2, j + 9, k + _cg_idx2 + 2) * c[0][4][1];
            buf0[31 + rel] += bIn(i + hipThreadIdx_x + -1, j + 9, k + _cg_idx2 + 2) * c[1][4][1];
            buf0[31 + rel] += bIn(i + hipThreadIdx_x, j + 9, k + _cg_idx2 + 2) * c[2][4][1];
            buf0[31 + rel] += bIn(i + hipThreadIdx_x + 1, j + 9, k + _cg_idx2 + 2) * c[3][4][1];
            buf0[31 + rel] += bIn(i + hipThreadIdx_x + 2, j + 9, k + _cg_idx2 + 2) * c[4][4][1];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x + -2, j + 9, k + _cg_idx2 + 2) * c[0][4][2];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x + -1, j + 9, k + _cg_idx2 + 2) * c[1][4][2];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x, j + 9, k + _cg_idx2 + 2) * c[2][4][2];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x + 1, j + 9, k + _cg_idx2 + 2) * c[3][4][2];
            buf0[23 + rel] += bIn(i + hipThreadIdx_x + 2, j + 9, k + _cg_idx2 + 2) * c[4][4][2];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x + -2, j + 9, k + _cg_idx2 + 2) * c[0][4][3];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x + -1, j + 9, k + _cg_idx2 + 2) * c[1][4][3];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x, j + 9, k + _cg_idx2 + 2) * c[2][4][3];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x + 1, j + 9, k + _cg_idx2 + 2) * c[3][4][3];
            buf0[15 + rel] += bIn(i + hipThreadIdx_x + 2, j + 9, k + _cg_idx2 + 2) * c[4][4][3];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x + -2, j + 9, k + _cg_idx2 + 2) * c[0][4][4];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x + -1, j + 9, k + _cg_idx2 + 2) * c[1][4][4];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x, j + 9, k + _cg_idx2 + 2) * c[2][4][4];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x + 1, j + 9, k + _cg_idx2 + 2) * c[3][4][4];
            buf0[7 + rel] += bIn(i + hipThreadIdx_x + 2, j + 9, k + _cg_idx2 + 2) * c[4][4][4];
          }
          _cg_rel2 += 8;
        }
      }
      {
        buf0[56] += bIn(i + hipThreadIdx_x + -2, j + -2, k + 6) * c[0][0][1];
        buf0[56] += bIn(i + hipThreadIdx_x + -1, j + -2, k + 6) * c[1][0][1];
        buf0[56] += bIn(i + hipThreadIdx_x, j + -2, k + 6) * c[2][0][1];
        buf0[56] += bIn(i + hipThreadIdx_x + 1, j + -2, k + 6) * c[3][0][1];
        buf0[56] += bIn(i + hipThreadIdx_x + 2, j + -2, k + 6) * c[4][0][1];
        buf0[48] += bIn(i + hipThreadIdx_x + -2, j + -2, k + 6) * c[0][0][2];
        buf0[48] += bIn(i + hipThreadIdx_x + -1, j + -2, k + 6) * c[1][0][2];
        buf0[48] += bIn(i + hipThreadIdx_x, j + -2, k + 6) * c[2][0][2];
        buf0[48] += bIn(i + hipThreadIdx_x + 1, j + -2, k + 6) * c[3][0][2];
        buf0[48] += bIn(i + hipThreadIdx_x + 2, j + -2, k + 6) * c[4][0][2];
        buf0[40] += bIn(i + hipThreadIdx_x + -2, j + -2, k + 6) * c[0][0][3];
        buf0[40] += bIn(i + hipThreadIdx_x + -1, j + -2, k + 6) * c[1][0][3];
        buf0[40] += bIn(i + hipThreadIdx_x, j + -2, k + 6) * c[2][0][3];
        buf0[40] += bIn(i + hipThreadIdx_x + 1, j + -2, k + 6) * c[3][0][3];
        buf0[40] += bIn(i + hipThreadIdx_x + 2, j + -2, k + 6) * c[4][0][3];
        buf0[32] += bIn(i + hipThreadIdx_x + -2, j + -2, k + 6) * c[0][0][4];
        buf0[32] += bIn(i + hipThreadIdx_x + -1, j + -2, k + 6) * c[1][0][4];
        buf0[32] += bIn(i + hipThreadIdx_x, j + -2, k + 6) * c[2][0][4];
        buf0[32] += bIn(i + hipThreadIdx_x + 1, j + -2, k + 6) * c[3][0][4];
        buf0[32] += bIn(i + hipThreadIdx_x + 2, j + -2, k + 6) * c[4][0][4];
      }
      {
        buf0[57] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 6) * c[0][0][1];
        buf0[57] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 6) * c[1][0][1];
        buf0[57] += bIn(i + hipThreadIdx_x, j + -1, k + 6) * c[2][0][1];
        buf0[57] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 6) * c[3][0][1];
        buf0[57] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 6) * c[4][0][1];
        buf0[49] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 6) * c[0][0][2];
        buf0[49] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 6) * c[1][0][2];
        buf0[49] += bIn(i + hipThreadIdx_x, j + -1, k + 6) * c[2][0][2];
        buf0[49] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 6) * c[3][0][2];
        buf0[49] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 6) * c[4][0][2];
        buf0[41] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 6) * c[0][0][3];
        buf0[41] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 6) * c[1][0][3];
        buf0[41] += bIn(i + hipThreadIdx_x, j + -1, k + 6) * c[2][0][3];
        buf0[41] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 6) * c[3][0][3];
        buf0[41] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 6) * c[4][0][3];
        buf0[33] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 6) * c[0][0][4];
        buf0[33] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 6) * c[1][0][4];
        buf0[33] += bIn(i + hipThreadIdx_x, j + -1, k + 6) * c[2][0][4];
        buf0[33] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 6) * c[3][0][4];
        buf0[33] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 6) * c[4][0][4];
        buf0[56] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 6) * c[0][1][1];
        buf0[56] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 6) * c[1][1][1];
        buf0[56] += bIn(i + hipThreadIdx_x, j + -1, k + 6) * c[2][1][1];
        buf0[56] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 6) * c[3][1][1];
        buf0[56] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 6) * c[4][1][1];
        buf0[48] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 6) * c[0][1][2];
        buf0[48] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 6) * c[1][1][2];
        buf0[48] += bIn(i + hipThreadIdx_x, j + -1, k + 6) * c[2][1][2];
        buf0[48] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 6) * c[3][1][2];
        buf0[48] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 6) * c[4][1][2];
        buf0[40] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 6) * c[0][1][3];
        buf0[40] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 6) * c[1][1][3];
        buf0[40] += bIn(i + hipThreadIdx_x, j + -1, k + 6) * c[2][1][3];
        buf0[40] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 6) * c[3][1][3];
        buf0[40] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 6) * c[4][1][3];
        buf0[32] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 6) * c[0][1][4];
        buf0[32] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 6) * c[1][1][4];
        buf0[32] += bIn(i + hipThreadIdx_x, j + -1, k + 6) * c[2][1][4];
        buf0[32] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 6) * c[3][1][4];
        buf0[32] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 6) * c[4][1][4];
      }
      {
        buf0[58] += bIn(i + hipThreadIdx_x + -2, j, k + 6) * c[0][0][1];
        buf0[58] += bIn(i + hipThreadIdx_x + -1, j, k + 6) * c[1][0][1];
        buf0[58] += bIn(i + hipThreadIdx_x, j, k + 6) * c[2][0][1];
        buf0[58] += bIn(i + hipThreadIdx_x + 1, j, k + 6) * c[3][0][1];
        buf0[58] += bIn(i + hipThreadIdx_x + 2, j, k + 6) * c[4][0][1];
        buf0[50] += bIn(i + hipThreadIdx_x + -2, j, k + 6) * c[0][0][2];
        buf0[50] += bIn(i + hipThreadIdx_x + -1, j, k + 6) * c[1][0][2];
        buf0[50] += bIn(i + hipThreadIdx_x, j, k + 6) * c[2][0][2];
        buf0[50] += bIn(i + hipThreadIdx_x + 1, j, k + 6) * c[3][0][2];
        buf0[50] += bIn(i + hipThreadIdx_x + 2, j, k + 6) * c[4][0][2];
        buf0[42] += bIn(i + hipThreadIdx_x + -2, j, k + 6) * c[0][0][3];
        buf0[42] += bIn(i + hipThreadIdx_x + -1, j, k + 6) * c[1][0][3];
        buf0[42] += bIn(i + hipThreadIdx_x, j, k + 6) * c[2][0][3];
        buf0[42] += bIn(i + hipThreadIdx_x + 1, j, k + 6) * c[3][0][3];
        buf0[42] += bIn(i + hipThreadIdx_x + 2, j, k + 6) * c[4][0][3];
        buf0[34] += bIn(i + hipThreadIdx_x + -2, j, k + 6) * c[0][0][4];
        buf0[34] += bIn(i + hipThreadIdx_x + -1, j, k + 6) * c[1][0][4];
        buf0[34] += bIn(i + hipThreadIdx_x, j, k + 6) * c[2][0][4];
        buf0[34] += bIn(i + hipThreadIdx_x + 1, j, k + 6) * c[3][0][4];
        buf0[34] += bIn(i + hipThreadIdx_x + 2, j, k + 6) * c[4][0][4];
        buf0[57] += bIn(i + hipThreadIdx_x + -2, j, k + 6) * c[0][1][1];
        buf0[57] += bIn(i + hipThreadIdx_x + -1, j, k + 6) * c[1][1][1];
        buf0[57] += bIn(i + hipThreadIdx_x, j, k + 6) * c[2][1][1];
        buf0[57] += bIn(i + hipThreadIdx_x + 1, j, k + 6) * c[3][1][1];
        buf0[57] += bIn(i + hipThreadIdx_x + 2, j, k + 6) * c[4][1][1];
        buf0[49] += bIn(i + hipThreadIdx_x + -2, j, k + 6) * c[0][1][2];
        buf0[49] += bIn(i + hipThreadIdx_x + -1, j, k + 6) * c[1][1][2];
        buf0[49] += bIn(i + hipThreadIdx_x, j, k + 6) * c[2][1][2];
        buf0[49] += bIn(i + hipThreadIdx_x + 1, j, k + 6) * c[3][1][2];
        buf0[49] += bIn(i + hipThreadIdx_x + 2, j, k + 6) * c[4][1][2];
        buf0[41] += bIn(i + hipThreadIdx_x + -2, j, k + 6) * c[0][1][3];
        buf0[41] += bIn(i + hipThreadIdx_x + -1, j, k + 6) * c[1][1][3];
        buf0[41] += bIn(i + hipThreadIdx_x, j, k + 6) * c[2][1][3];
        buf0[41] += bIn(i + hipThreadIdx_x + 1, j, k + 6) * c[3][1][3];
        buf0[41] += bIn(i + hipThreadIdx_x + 2, j, k + 6) * c[4][1][3];
        buf0[33] += bIn(i + hipThreadIdx_x + -2, j, k + 6) * c[0][1][4];
        buf0[33] += bIn(i + hipThreadIdx_x + -1, j, k + 6) * c[1][1][4];
        buf0[33] += bIn(i + hipThreadIdx_x, j, k + 6) * c[2][1][4];
        buf0[33] += bIn(i + hipThreadIdx_x + 1, j, k + 6) * c[3][1][4];
        buf0[33] += bIn(i + hipThreadIdx_x + 2, j, k + 6) * c[4][1][4];
        buf0[56] += bIn(i + hipThreadIdx_x + -2, j, k + 6) * c[0][2][1];
        buf0[56] += bIn(i + hipThreadIdx_x + -1, j, k + 6) * c[1][2][1];
        buf0[56] += bIn(i + hipThreadIdx_x, j, k + 6) * c[2][2][1];
        buf0[56] += bIn(i + hipThreadIdx_x + 1, j, k + 6) * c[3][2][1];
        buf0[56] += bIn(i + hipThreadIdx_x + 2, j, k + 6) * c[4][2][1];
        buf0[48] += bIn(i + hipThreadIdx_x + -2, j, k + 6) * c[0][2][2];
        buf0[48] += bIn(i + hipThreadIdx_x + -1, j, k + 6) * c[1][2][2];
        buf0[48] += bIn(i + hipThreadIdx_x, j, k + 6) * c[2][2][2];
        buf0[48] += bIn(i + hipThreadIdx_x + 1, j, k + 6) * c[3][2][2];
        buf0[48] += bIn(i + hipThreadIdx_x + 2, j, k + 6) * c[4][2][2];
        buf0[40] += bIn(i + hipThreadIdx_x + -2, j, k + 6) * c[0][2][3];
        buf0[40] += bIn(i + hipThreadIdx_x + -1, j, k + 6) * c[1][2][3];
        buf0[40] += bIn(i + hipThreadIdx_x, j, k + 6) * c[2][2][3];
        buf0[40] += bIn(i + hipThreadIdx_x + 1, j, k + 6) * c[3][2][3];
        buf0[40] += bIn(i + hipThreadIdx_x + 2, j, k + 6) * c[4][2][3];
        buf0[32] += bIn(i + hipThreadIdx_x + -2, j, k + 6) * c[0][2][4];
        buf0[32] += bIn(i + hipThreadIdx_x + -1, j, k + 6) * c[1][2][4];
        buf0[32] += bIn(i + hipThreadIdx_x, j, k + 6) * c[2][2][4];
        buf0[32] += bIn(i + hipThreadIdx_x + 1, j, k + 6) * c[3][2][4];
        buf0[32] += bIn(i + hipThreadIdx_x + 2, j, k + 6) * c[4][2][4];
      }
      {
        buf0[59] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 6) * c[0][0][1];
        buf0[59] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 6) * c[1][0][1];
        buf0[59] += bIn(i + hipThreadIdx_x, j + 1, k + 6) * c[2][0][1];
        buf0[59] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 6) * c[3][0][1];
        buf0[59] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 6) * c[4][0][1];
        buf0[51] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 6) * c[0][0][2];
        buf0[51] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 6) * c[1][0][2];
        buf0[51] += bIn(i + hipThreadIdx_x, j + 1, k + 6) * c[2][0][2];
        buf0[51] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 6) * c[3][0][2];
        buf0[51] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 6) * c[4][0][2];
        buf0[43] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 6) * c[0][0][3];
        buf0[43] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 6) * c[1][0][3];
        buf0[43] += bIn(i + hipThreadIdx_x, j + 1, k + 6) * c[2][0][3];
        buf0[43] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 6) * c[3][0][3];
        buf0[43] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 6) * c[4][0][3];
        buf0[35] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 6) * c[0][0][4];
        buf0[35] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 6) * c[1][0][4];
        buf0[35] += bIn(i + hipThreadIdx_x, j + 1, k + 6) * c[2][0][4];
        buf0[35] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 6) * c[3][0][4];
        buf0[35] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 6) * c[4][0][4];
        buf0[58] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 6) * c[0][1][1];
        buf0[58] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 6) * c[1][1][1];
        buf0[58] += bIn(i + hipThreadIdx_x, j + 1, k + 6) * c[2][1][1];
        buf0[58] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 6) * c[3][1][1];
        buf0[58] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 6) * c[4][1][1];
        buf0[50] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 6) * c[0][1][2];
        buf0[50] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 6) * c[1][1][2];
        buf0[50] += bIn(i + hipThreadIdx_x, j + 1, k + 6) * c[2][1][2];
        buf0[50] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 6) * c[3][1][2];
        buf0[50] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 6) * c[4][1][2];
        buf0[42] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 6) * c[0][1][3];
        buf0[42] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 6) * c[1][1][3];
        buf0[42] += bIn(i + hipThreadIdx_x, j + 1, k + 6) * c[2][1][3];
        buf0[42] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 6) * c[3][1][3];
        buf0[42] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 6) * c[4][1][3];
        buf0[34] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 6) * c[0][1][4];
        buf0[34] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 6) * c[1][1][4];
        buf0[34] += bIn(i + hipThreadIdx_x, j + 1, k + 6) * c[2][1][4];
        buf0[34] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 6) * c[3][1][4];
        buf0[34] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 6) * c[4][1][4];
        buf0[57] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 6) * c[0][2][1];
        buf0[57] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 6) * c[1][2][1];
        buf0[57] += bIn(i + hipThreadIdx_x, j + 1, k + 6) * c[2][2][1];
        buf0[57] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 6) * c[3][2][1];
        buf0[57] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 6) * c[4][2][1];
        buf0[49] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 6) * c[0][2][2];
        buf0[49] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 6) * c[1][2][2];
        buf0[49] += bIn(i + hipThreadIdx_x, j + 1, k + 6) * c[2][2][2];
        buf0[49] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 6) * c[3][2][2];
        buf0[49] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 6) * c[4][2][2];
        buf0[41] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 6) * c[0][2][3];
        buf0[41] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 6) * c[1][2][3];
        buf0[41] += bIn(i + hipThreadIdx_x, j + 1, k + 6) * c[2][2][3];
        buf0[41] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 6) * c[3][2][3];
        buf0[41] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 6) * c[4][2][3];
        buf0[33] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 6) * c[0][2][4];
        buf0[33] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 6) * c[1][2][4];
        buf0[33] += bIn(i + hipThreadIdx_x, j + 1, k + 6) * c[2][2][4];
        buf0[33] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 6) * c[3][2][4];
        buf0[33] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 6) * c[4][2][4];
        buf0[56] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 6) * c[0][3][1];
        buf0[56] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 6) * c[1][3][1];
        buf0[56] += bIn(i + hipThreadIdx_x, j + 1, k + 6) * c[2][3][1];
        buf0[56] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 6) * c[3][3][1];
        buf0[56] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 6) * c[4][3][1];
        buf0[48] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 6) * c[0][3][2];
        buf0[48] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 6) * c[1][3][2];
        buf0[48] += bIn(i + hipThreadIdx_x, j + 1, k + 6) * c[2][3][2];
        buf0[48] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 6) * c[3][3][2];
        buf0[48] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 6) * c[4][3][2];
        buf0[40] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 6) * c[0][3][3];
        buf0[40] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 6) * c[1][3][3];
        buf0[40] += bIn(i + hipThreadIdx_x, j + 1, k + 6) * c[2][3][3];
        buf0[40] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 6) * c[3][3][3];
        buf0[40] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 6) * c[4][3][3];
        buf0[32] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 6) * c[0][3][4];
        buf0[32] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 6) * c[1][3][4];
        buf0[32] += bIn(i + hipThreadIdx_x, j + 1, k + 6) * c[2][3][4];
        buf0[32] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 6) * c[3][3][4];
        buf0[32] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 6) * c[4][3][4];
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 4; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[60 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6) * c[0][0][1];
            buf0[60 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6) * c[1][0][1];
            buf0[60 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6) * c[2][0][1];
            buf0[60 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6) * c[3][0][1];
            buf0[60 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6) * c[4][0][1];
            buf0[52 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6) * c[0][0][2];
            buf0[52 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6) * c[1][0][2];
            buf0[52 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6) * c[2][0][2];
            buf0[52 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6) * c[3][0][2];
            buf0[52 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6) * c[4][0][2];
            buf0[44 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6) * c[0][0][3];
            buf0[44 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6) * c[1][0][3];
            buf0[44 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6) * c[2][0][3];
            buf0[44 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6) * c[3][0][3];
            buf0[44 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6) * c[4][0][3];
            buf0[36 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6) * c[0][0][4];
            buf0[36 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6) * c[1][0][4];
            buf0[36 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6) * c[2][0][4];
            buf0[36 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6) * c[3][0][4];
            buf0[36 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6) * c[4][0][4];
            buf0[59 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6) * c[0][1][1];
            buf0[59 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6) * c[1][1][1];
            buf0[59 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6) * c[2][1][1];
            buf0[59 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6) * c[3][1][1];
            buf0[59 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6) * c[4][1][1];
            buf0[51 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6) * c[0][1][2];
            buf0[51 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6) * c[1][1][2];
            buf0[51 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6) * c[2][1][2];
            buf0[51 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6) * c[3][1][2];
            buf0[51 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6) * c[4][1][2];
            buf0[43 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6) * c[0][1][3];
            buf0[43 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6) * c[1][1][3];
            buf0[43 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6) * c[2][1][3];
            buf0[43 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6) * c[3][1][3];
            buf0[43 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6) * c[4][1][3];
            buf0[35 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6) * c[0][1][4];
            buf0[35 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6) * c[1][1][4];
            buf0[35 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6) * c[2][1][4];
            buf0[35 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6) * c[3][1][4];
            buf0[35 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6) * c[4][1][4];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6) * c[0][2][1];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6) * c[1][2][1];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6) * c[2][2][1];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6) * c[3][2][1];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6) * c[4][2][1];
            buf0[50 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6) * c[0][2][2];
            buf0[50 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6) * c[1][2][2];
            buf0[50 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6) * c[2][2][2];
            buf0[50 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6) * c[3][2][2];
            buf0[50 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6) * c[4][2][2];
            buf0[42 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6) * c[0][2][3];
            buf0[42 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6) * c[1][2][3];
            buf0[42 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6) * c[2][2][3];
            buf0[42 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6) * c[3][2][3];
            buf0[42 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6) * c[4][2][3];
            buf0[34 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6) * c[0][2][4];
            buf0[34 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6) * c[1][2][4];
            buf0[34 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6) * c[2][2][4];
            buf0[34 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6) * c[3][2][4];
            buf0[34 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6) * c[4][2][4];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6) * c[0][3][1];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6) * c[1][3][1];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6) * c[2][3][1];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6) * c[3][3][1];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6) * c[4][3][1];
            buf0[49 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6) * c[0][3][2];
            buf0[49 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6) * c[1][3][2];
            buf0[49 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6) * c[2][3][2];
            buf0[49 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6) * c[3][3][2];
            buf0[49 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6) * c[4][3][2];
            buf0[41 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6) * c[0][3][3];
            buf0[41 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6) * c[1][3][3];
            buf0[41 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6) * c[2][3][3];
            buf0[41 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6) * c[3][3][3];
            buf0[41 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6) * c[4][3][3];
            buf0[33 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6) * c[0][3][4];
            buf0[33 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6) * c[1][3][4];
            buf0[33 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6) * c[2][3][4];
            buf0[33 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6) * c[3][3][4];
            buf0[33 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6) * c[4][3][4];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6) * c[0][4][1];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6) * c[1][4][1];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6) * c[2][4][1];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6) * c[3][4][1];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6) * c[4][4][1];
            buf0[48 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6) * c[0][4][2];
            buf0[48 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6) * c[1][4][2];
            buf0[48 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6) * c[2][4][2];
            buf0[48 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6) * c[3][4][2];
            buf0[48 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6) * c[4][4][2];
            buf0[40 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6) * c[0][4][3];
            buf0[40 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6) * c[1][4][3];
            buf0[40 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6) * c[2][4][3];
            buf0[40 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6) * c[3][4][3];
            buf0[40 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6) * c[4][4][3];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6) * c[0][4][4];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6) * c[1][4][4];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6) * c[2][4][4];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6) * c[3][4][4];
            buf0[32 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6) * c[4][4][4];
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[63] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 6) * c[0][1][1];
        buf0[63] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 6) * c[1][1][1];
        buf0[63] += bIn(i + hipThreadIdx_x, j + 6, k + 6) * c[2][1][1];
        buf0[63] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 6) * c[3][1][1];
        buf0[63] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 6) * c[4][1][1];
        buf0[55] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 6) * c[0][1][2];
        buf0[55] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 6) * c[1][1][2];
        buf0[55] += bIn(i + hipThreadIdx_x, j + 6, k + 6) * c[2][1][2];
        buf0[55] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 6) * c[3][1][2];
        buf0[55] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 6) * c[4][1][2];
        buf0[47] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 6) * c[0][1][3];
        buf0[47] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 6) * c[1][1][3];
        buf0[47] += bIn(i + hipThreadIdx_x, j + 6, k + 6) * c[2][1][3];
        buf0[47] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 6) * c[3][1][3];
        buf0[47] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 6) * c[4][1][3];
        buf0[39] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 6) * c[0][1][4];
        buf0[39] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 6) * c[1][1][4];
        buf0[39] += bIn(i + hipThreadIdx_x, j + 6, k + 6) * c[2][1][4];
        buf0[39] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 6) * c[3][1][4];
        buf0[39] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 6) * c[4][1][4];
        buf0[62] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 6) * c[0][2][1];
        buf0[62] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 6) * c[1][2][1];
        buf0[62] += bIn(i + hipThreadIdx_x, j + 6, k + 6) * c[2][2][1];
        buf0[62] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 6) * c[3][2][1];
        buf0[62] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 6) * c[4][2][1];
        buf0[54] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 6) * c[0][2][2];
        buf0[54] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 6) * c[1][2][2];
        buf0[54] += bIn(i + hipThreadIdx_x, j + 6, k + 6) * c[2][2][2];
        buf0[54] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 6) * c[3][2][2];
        buf0[54] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 6) * c[4][2][2];
        buf0[46] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 6) * c[0][2][3];
        buf0[46] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 6) * c[1][2][3];
        buf0[46] += bIn(i + hipThreadIdx_x, j + 6, k + 6) * c[2][2][3];
        buf0[46] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 6) * c[3][2][3];
        buf0[46] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 6) * c[4][2][3];
        buf0[38] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 6) * c[0][2][4];
        buf0[38] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 6) * c[1][2][4];
        buf0[38] += bIn(i + hipThreadIdx_x, j + 6, k + 6) * c[2][2][4];
        buf0[38] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 6) * c[3][2][4];
        buf0[38] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 6) * c[4][2][4];
        buf0[61] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 6) * c[0][3][1];
        buf0[61] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 6) * c[1][3][1];
        buf0[61] += bIn(i + hipThreadIdx_x, j + 6, k + 6) * c[2][3][1];
        buf0[61] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 6) * c[3][3][1];
        buf0[61] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 6) * c[4][3][1];
        buf0[53] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 6) * c[0][3][2];
        buf0[53] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 6) * c[1][3][2];
        buf0[53] += bIn(i + hipThreadIdx_x, j + 6, k + 6) * c[2][3][2];
        buf0[53] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 6) * c[3][3][2];
        buf0[53] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 6) * c[4][3][2];
        buf0[45] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 6) * c[0][3][3];
        buf0[45] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 6) * c[1][3][3];
        buf0[45] += bIn(i + hipThreadIdx_x, j + 6, k + 6) * c[2][3][3];
        buf0[45] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 6) * c[3][3][3];
        buf0[45] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 6) * c[4][3][3];
        buf0[37] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 6) * c[0][3][4];
        buf0[37] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 6) * c[1][3][4];
        buf0[37] += bIn(i + hipThreadIdx_x, j + 6, k + 6) * c[2][3][4];
        buf0[37] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 6) * c[3][3][4];
        buf0[37] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 6) * c[4][3][4];
        buf0[60] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 6) * c[0][4][1];
        buf0[60] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 6) * c[1][4][1];
        buf0[60] += bIn(i + hipThreadIdx_x, j + 6, k + 6) * c[2][4][1];
        buf0[60] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 6) * c[3][4][1];
        buf0[60] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 6) * c[4][4][1];
        buf0[52] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 6) * c[0][4][2];
        buf0[52] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 6) * c[1][4][2];
        buf0[52] += bIn(i + hipThreadIdx_x, j + 6, k + 6) * c[2][4][2];
        buf0[52] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 6) * c[3][4][2];
        buf0[52] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 6) * c[4][4][2];
        buf0[44] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 6) * c[0][4][3];
        buf0[44] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 6) * c[1][4][3];
        buf0[44] += bIn(i + hipThreadIdx_x, j + 6, k + 6) * c[2][4][3];
        buf0[44] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 6) * c[3][4][3];
        buf0[44] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 6) * c[4][4][3];
        buf0[36] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 6) * c[0][4][4];
        buf0[36] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 6) * c[1][4][4];
        buf0[36] += bIn(i + hipThreadIdx_x, j + 6, k + 6) * c[2][4][4];
        buf0[36] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 6) * c[3][4][4];
        buf0[36] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 6) * c[4][4][4];
      }
      {
        buf0[63] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 6) * c[0][2][1];
        buf0[63] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 6) * c[1][2][1];
        buf0[63] += bIn(i + hipThreadIdx_x, j + 7, k + 6) * c[2][2][1];
        buf0[63] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 6) * c[3][2][1];
        buf0[63] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 6) * c[4][2][1];
        buf0[55] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 6) * c[0][2][2];
        buf0[55] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 6) * c[1][2][2];
        buf0[55] += bIn(i + hipThreadIdx_x, j + 7, k + 6) * c[2][2][2];
        buf0[55] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 6) * c[3][2][2];
        buf0[55] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 6) * c[4][2][2];
        buf0[47] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 6) * c[0][2][3];
        buf0[47] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 6) * c[1][2][3];
        buf0[47] += bIn(i + hipThreadIdx_x, j + 7, k + 6) * c[2][2][3];
        buf0[47] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 6) * c[3][2][3];
        buf0[47] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 6) * c[4][2][3];
        buf0[39] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 6) * c[0][2][4];
        buf0[39] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 6) * c[1][2][4];
        buf0[39] += bIn(i + hipThreadIdx_x, j + 7, k + 6) * c[2][2][4];
        buf0[39] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 6) * c[3][2][4];
        buf0[39] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 6) * c[4][2][4];
        buf0[62] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 6) * c[0][3][1];
        buf0[62] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 6) * c[1][3][1];
        buf0[62] += bIn(i + hipThreadIdx_x, j + 7, k + 6) * c[2][3][1];
        buf0[62] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 6) * c[3][3][1];
        buf0[62] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 6) * c[4][3][1];
        buf0[54] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 6) * c[0][3][2];
        buf0[54] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 6) * c[1][3][2];
        buf0[54] += bIn(i + hipThreadIdx_x, j + 7, k + 6) * c[2][3][2];
        buf0[54] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 6) * c[3][3][2];
        buf0[54] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 6) * c[4][3][2];
        buf0[46] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 6) * c[0][3][3];
        buf0[46] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 6) * c[1][3][3];
        buf0[46] += bIn(i + hipThreadIdx_x, j + 7, k + 6) * c[2][3][3];
        buf0[46] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 6) * c[3][3][3];
        buf0[46] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 6) * c[4][3][3];
        buf0[38] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 6) * c[0][3][4];
        buf0[38] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 6) * c[1][3][4];
        buf0[38] += bIn(i + hipThreadIdx_x, j + 7, k + 6) * c[2][3][4];
        buf0[38] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 6) * c[3][3][4];
        buf0[38] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 6) * c[4][3][4];
        buf0[61] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 6) * c[0][4][1];
        buf0[61] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 6) * c[1][4][1];
        buf0[61] += bIn(i + hipThreadIdx_x, j + 7, k + 6) * c[2][4][1];
        buf0[61] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 6) * c[3][4][1];
        buf0[61] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 6) * c[4][4][1];
        buf0[53] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 6) * c[0][4][2];
        buf0[53] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 6) * c[1][4][2];
        buf0[53] += bIn(i + hipThreadIdx_x, j + 7, k + 6) * c[2][4][2];
        buf0[53] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 6) * c[3][4][2];
        buf0[53] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 6) * c[4][4][2];
        buf0[45] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 6) * c[0][4][3];
        buf0[45] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 6) * c[1][4][3];
        buf0[45] += bIn(i + hipThreadIdx_x, j + 7, k + 6) * c[2][4][3];
        buf0[45] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 6) * c[3][4][3];
        buf0[45] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 6) * c[4][4][3];
        buf0[37] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 6) * c[0][4][4];
        buf0[37] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 6) * c[1][4][4];
        buf0[37] += bIn(i + hipThreadIdx_x, j + 7, k + 6) * c[2][4][4];
        buf0[37] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 6) * c[3][4][4];
        buf0[37] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 6) * c[4][4][4];
      }
      {
        buf0[63] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 6) * c[0][3][1];
        buf0[63] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 6) * c[1][3][1];
        buf0[63] += bIn(i + hipThreadIdx_x, j + 8, k + 6) * c[2][3][1];
        buf0[63] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 6) * c[3][3][1];
        buf0[63] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 6) * c[4][3][1];
        buf0[55] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 6) * c[0][3][2];
        buf0[55] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 6) * c[1][3][2];
        buf0[55] += bIn(i + hipThreadIdx_x, j + 8, k + 6) * c[2][3][2];
        buf0[55] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 6) * c[3][3][2];
        buf0[55] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 6) * c[4][3][2];
        buf0[47] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 6) * c[0][3][3];
        buf0[47] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 6) * c[1][3][3];
        buf0[47] += bIn(i + hipThreadIdx_x, j + 8, k + 6) * c[2][3][3];
        buf0[47] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 6) * c[3][3][3];
        buf0[47] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 6) * c[4][3][3];
        buf0[39] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 6) * c[0][3][4];
        buf0[39] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 6) * c[1][3][4];
        buf0[39] += bIn(i + hipThreadIdx_x, j + 8, k + 6) * c[2][3][4];
        buf0[39] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 6) * c[3][3][4];
        buf0[39] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 6) * c[4][3][4];
        buf0[62] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 6) * c[0][4][1];
        buf0[62] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 6) * c[1][4][1];
        buf0[62] += bIn(i + hipThreadIdx_x, j + 8, k + 6) * c[2][4][1];
        buf0[62] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 6) * c[3][4][1];
        buf0[62] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 6) * c[4][4][1];
        buf0[54] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 6) * c[0][4][2];
        buf0[54] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 6) * c[1][4][2];
        buf0[54] += bIn(i + hipThreadIdx_x, j + 8, k + 6) * c[2][4][2];
        buf0[54] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 6) * c[3][4][2];
        buf0[54] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 6) * c[4][4][2];
        buf0[46] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 6) * c[0][4][3];
        buf0[46] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 6) * c[1][4][3];
        buf0[46] += bIn(i + hipThreadIdx_x, j + 8, k + 6) * c[2][4][3];
        buf0[46] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 6) * c[3][4][3];
        buf0[46] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 6) * c[4][4][3];
        buf0[38] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 6) * c[0][4][4];
        buf0[38] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 6) * c[1][4][4];
        buf0[38] += bIn(i + hipThreadIdx_x, j + 8, k + 6) * c[2][4][4];
        buf0[38] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 6) * c[3][4][4];
        buf0[38] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 6) * c[4][4][4];
      }
      {
        buf0[63] += bIn(i + hipThreadIdx_x + -2, j + 9, k + 6) * c[0][4][1];
        buf0[63] += bIn(i + hipThreadIdx_x + -1, j + 9, k + 6) * c[1][4][1];
        buf0[63] += bIn(i + hipThreadIdx_x, j + 9, k + 6) * c[2][4][1];
        buf0[63] += bIn(i + hipThreadIdx_x + 1, j + 9, k + 6) * c[3][4][1];
        buf0[63] += bIn(i + hipThreadIdx_x + 2, j + 9, k + 6) * c[4][4][1];
        buf0[55] += bIn(i + hipThreadIdx_x + -2, j + 9, k + 6) * c[0][4][2];
        buf0[55] += bIn(i + hipThreadIdx_x + -1, j + 9, k + 6) * c[1][4][2];
        buf0[55] += bIn(i + hipThreadIdx_x, j + 9, k + 6) * c[2][4][2];
        buf0[55] += bIn(i + hipThreadIdx_x + 1, j + 9, k + 6) * c[3][4][2];
        buf0[55] += bIn(i + hipThreadIdx_x + 2, j + 9, k + 6) * c[4][4][2];
        buf0[47] += bIn(i + hipThreadIdx_x + -2, j + 9, k + 6) * c[0][4][3];
        buf0[47] += bIn(i + hipThreadIdx_x + -1, j + 9, k + 6) * c[1][4][3];
        buf0[47] += bIn(i + hipThreadIdx_x, j + 9, k + 6) * c[2][4][3];
        buf0[47] += bIn(i + hipThreadIdx_x + 1, j + 9, k + 6) * c[3][4][3];
        buf0[47] += bIn(i + hipThreadIdx_x + 2, j + 9, k + 6) * c[4][4][3];
        buf0[39] += bIn(i + hipThreadIdx_x + -2, j + 9, k + 6) * c[0][4][4];
        buf0[39] += bIn(i + hipThreadIdx_x + -1, j + 9, k + 6) * c[1][4][4];
        buf0[39] += bIn(i + hipThreadIdx_x, j + 9, k + 6) * c[2][4][4];
        buf0[39] += bIn(i + hipThreadIdx_x + 1, j + 9, k + 6) * c[3][4][4];
        buf0[39] += bIn(i + hipThreadIdx_x + 2, j + 9, k + 6) * c[4][4][4];
      }
      {
        buf0[56] += bIn(i + hipThreadIdx_x + -2, j + -2, k + 7) * c[0][0][2];
        buf0[56] += bIn(i + hipThreadIdx_x + -1, j + -2, k + 7) * c[1][0][2];
        buf0[56] += bIn(i + hipThreadIdx_x, j + -2, k + 7) * c[2][0][2];
        buf0[56] += bIn(i + hipThreadIdx_x + 1, j + -2, k + 7) * c[3][0][2];
        buf0[56] += bIn(i + hipThreadIdx_x + 2, j + -2, k + 7) * c[4][0][2];
        buf0[48] += bIn(i + hipThreadIdx_x + -2, j + -2, k + 7) * c[0][0][3];
        buf0[48] += bIn(i + hipThreadIdx_x + -1, j + -2, k + 7) * c[1][0][3];
        buf0[48] += bIn(i + hipThreadIdx_x, j + -2, k + 7) * c[2][0][3];
        buf0[48] += bIn(i + hipThreadIdx_x + 1, j + -2, k + 7) * c[3][0][3];
        buf0[48] += bIn(i + hipThreadIdx_x + 2, j + -2, k + 7) * c[4][0][3];
        buf0[40] += bIn(i + hipThreadIdx_x + -2, j + -2, k + 7) * c[0][0][4];
        buf0[40] += bIn(i + hipThreadIdx_x + -1, j + -2, k + 7) * c[1][0][4];
        buf0[40] += bIn(i + hipThreadIdx_x, j + -2, k + 7) * c[2][0][4];
        buf0[40] += bIn(i + hipThreadIdx_x + 1, j + -2, k + 7) * c[3][0][4];
        buf0[40] += bIn(i + hipThreadIdx_x + 2, j + -2, k + 7) * c[4][0][4];
      }
      {
        buf0[57] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 7) * c[0][0][2];
        buf0[57] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 7) * c[1][0][2];
        buf0[57] += bIn(i + hipThreadIdx_x, j + -1, k + 7) * c[2][0][2];
        buf0[57] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 7) * c[3][0][2];
        buf0[57] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 7) * c[4][0][2];
        buf0[49] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 7) * c[0][0][3];
        buf0[49] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 7) * c[1][0][3];
        buf0[49] += bIn(i + hipThreadIdx_x, j + -1, k + 7) * c[2][0][3];
        buf0[49] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 7) * c[3][0][3];
        buf0[49] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 7) * c[4][0][3];
        buf0[41] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 7) * c[0][0][4];
        buf0[41] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 7) * c[1][0][4];
        buf0[41] += bIn(i + hipThreadIdx_x, j + -1, k + 7) * c[2][0][4];
        buf0[41] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 7) * c[3][0][4];
        buf0[41] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 7) * c[4][0][4];
        buf0[56] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 7) * c[0][1][2];
        buf0[56] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 7) * c[1][1][2];
        buf0[56] += bIn(i + hipThreadIdx_x, j + -1, k + 7) * c[2][1][2];
        buf0[56] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 7) * c[3][1][2];
        buf0[56] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 7) * c[4][1][2];
        buf0[48] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 7) * c[0][1][3];
        buf0[48] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 7) * c[1][1][3];
        buf0[48] += bIn(i + hipThreadIdx_x, j + -1, k + 7) * c[2][1][3];
        buf0[48] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 7) * c[3][1][3];
        buf0[48] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 7) * c[4][1][3];
        buf0[40] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 7) * c[0][1][4];
        buf0[40] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 7) * c[1][1][4];
        buf0[40] += bIn(i + hipThreadIdx_x, j + -1, k + 7) * c[2][1][4];
        buf0[40] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 7) * c[3][1][4];
        buf0[40] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 7) * c[4][1][4];
      }
      {
        buf0[58] += bIn(i + hipThreadIdx_x + -2, j, k + 7) * c[0][0][2];
        buf0[58] += bIn(i + hipThreadIdx_x + -1, j, k + 7) * c[1][0][2];
        buf0[58] += bIn(i + hipThreadIdx_x, j, k + 7) * c[2][0][2];
        buf0[58] += bIn(i + hipThreadIdx_x + 1, j, k + 7) * c[3][0][2];
        buf0[58] += bIn(i + hipThreadIdx_x + 2, j, k + 7) * c[4][0][2];
        buf0[50] += bIn(i + hipThreadIdx_x + -2, j, k + 7) * c[0][0][3];
        buf0[50] += bIn(i + hipThreadIdx_x + -1, j, k + 7) * c[1][0][3];
        buf0[50] += bIn(i + hipThreadIdx_x, j, k + 7) * c[2][0][3];
        buf0[50] += bIn(i + hipThreadIdx_x + 1, j, k + 7) * c[3][0][3];
        buf0[50] += bIn(i + hipThreadIdx_x + 2, j, k + 7) * c[4][0][3];
        buf0[42] += bIn(i + hipThreadIdx_x + -2, j, k + 7) * c[0][0][4];
        buf0[42] += bIn(i + hipThreadIdx_x + -1, j, k + 7) * c[1][0][4];
        buf0[42] += bIn(i + hipThreadIdx_x, j, k + 7) * c[2][0][4];
        buf0[42] += bIn(i + hipThreadIdx_x + 1, j, k + 7) * c[3][0][4];
        buf0[42] += bIn(i + hipThreadIdx_x + 2, j, k + 7) * c[4][0][4];
        buf0[57] += bIn(i + hipThreadIdx_x + -2, j, k + 7) * c[0][1][2];
        buf0[57] += bIn(i + hipThreadIdx_x + -1, j, k + 7) * c[1][1][2];
        buf0[57] += bIn(i + hipThreadIdx_x, j, k + 7) * c[2][1][2];
        buf0[57] += bIn(i + hipThreadIdx_x + 1, j, k + 7) * c[3][1][2];
        buf0[57] += bIn(i + hipThreadIdx_x + 2, j, k + 7) * c[4][1][2];
        buf0[49] += bIn(i + hipThreadIdx_x + -2, j, k + 7) * c[0][1][3];
        buf0[49] += bIn(i + hipThreadIdx_x + -1, j, k + 7) * c[1][1][3];
        buf0[49] += bIn(i + hipThreadIdx_x, j, k + 7) * c[2][1][3];
        buf0[49] += bIn(i + hipThreadIdx_x + 1, j, k + 7) * c[3][1][3];
        buf0[49] += bIn(i + hipThreadIdx_x + 2, j, k + 7) * c[4][1][3];
        buf0[41] += bIn(i + hipThreadIdx_x + -2, j, k + 7) * c[0][1][4];
        buf0[41] += bIn(i + hipThreadIdx_x + -1, j, k + 7) * c[1][1][4];
        buf0[41] += bIn(i + hipThreadIdx_x, j, k + 7) * c[2][1][4];
        buf0[41] += bIn(i + hipThreadIdx_x + 1, j, k + 7) * c[3][1][4];
        buf0[41] += bIn(i + hipThreadIdx_x + 2, j, k + 7) * c[4][1][4];
        buf0[56] += bIn(i + hipThreadIdx_x + -2, j, k + 7) * c[0][2][2];
        buf0[56] += bIn(i + hipThreadIdx_x + -1, j, k + 7) * c[1][2][2];
        buf0[56] += bIn(i + hipThreadIdx_x, j, k + 7) * c[2][2][2];
        buf0[56] += bIn(i + hipThreadIdx_x + 1, j, k + 7) * c[3][2][2];
        buf0[56] += bIn(i + hipThreadIdx_x + 2, j, k + 7) * c[4][2][2];
        buf0[48] += bIn(i + hipThreadIdx_x + -2, j, k + 7) * c[0][2][3];
        buf0[48] += bIn(i + hipThreadIdx_x + -1, j, k + 7) * c[1][2][3];
        buf0[48] += bIn(i + hipThreadIdx_x, j, k + 7) * c[2][2][3];
        buf0[48] += bIn(i + hipThreadIdx_x + 1, j, k + 7) * c[3][2][3];
        buf0[48] += bIn(i + hipThreadIdx_x + 2, j, k + 7) * c[4][2][3];
        buf0[40] += bIn(i + hipThreadIdx_x + -2, j, k + 7) * c[0][2][4];
        buf0[40] += bIn(i + hipThreadIdx_x + -1, j, k + 7) * c[1][2][4];
        buf0[40] += bIn(i + hipThreadIdx_x, j, k + 7) * c[2][2][4];
        buf0[40] += bIn(i + hipThreadIdx_x + 1, j, k + 7) * c[3][2][4];
        buf0[40] += bIn(i + hipThreadIdx_x + 2, j, k + 7) * c[4][2][4];
      }
      {
        buf0[59] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 7) * c[0][0][2];
        buf0[59] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 7) * c[1][0][2];
        buf0[59] += bIn(i + hipThreadIdx_x, j + 1, k + 7) * c[2][0][2];
        buf0[59] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 7) * c[3][0][2];
        buf0[59] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 7) * c[4][0][2];
        buf0[51] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 7) * c[0][0][3];
        buf0[51] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 7) * c[1][0][3];
        buf0[51] += bIn(i + hipThreadIdx_x, j + 1, k + 7) * c[2][0][3];
        buf0[51] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 7) * c[3][0][3];
        buf0[51] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 7) * c[4][0][3];
        buf0[43] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 7) * c[0][0][4];
        buf0[43] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 7) * c[1][0][4];
        buf0[43] += bIn(i + hipThreadIdx_x, j + 1, k + 7) * c[2][0][4];
        buf0[43] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 7) * c[3][0][4];
        buf0[43] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 7) * c[4][0][4];
        buf0[58] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 7) * c[0][1][2];
        buf0[58] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 7) * c[1][1][2];
        buf0[58] += bIn(i + hipThreadIdx_x, j + 1, k + 7) * c[2][1][2];
        buf0[58] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 7) * c[3][1][2];
        buf0[58] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 7) * c[4][1][2];
        buf0[50] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 7) * c[0][1][3];
        buf0[50] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 7) * c[1][1][3];
        buf0[50] += bIn(i + hipThreadIdx_x, j + 1, k + 7) * c[2][1][3];
        buf0[50] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 7) * c[3][1][3];
        buf0[50] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 7) * c[4][1][3];
        buf0[42] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 7) * c[0][1][4];
        buf0[42] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 7) * c[1][1][4];
        buf0[42] += bIn(i + hipThreadIdx_x, j + 1, k + 7) * c[2][1][4];
        buf0[42] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 7) * c[3][1][4];
        buf0[42] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 7) * c[4][1][4];
        buf0[57] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 7) * c[0][2][2];
        buf0[57] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 7) * c[1][2][2];
        buf0[57] += bIn(i + hipThreadIdx_x, j + 1, k + 7) * c[2][2][2];
        buf0[57] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 7) * c[3][2][2];
        buf0[57] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 7) * c[4][2][2];
        buf0[49] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 7) * c[0][2][3];
        buf0[49] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 7) * c[1][2][3];
        buf0[49] += bIn(i + hipThreadIdx_x, j + 1, k + 7) * c[2][2][3];
        buf0[49] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 7) * c[3][2][3];
        buf0[49] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 7) * c[4][2][3];
        buf0[41] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 7) * c[0][2][4];
        buf0[41] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 7) * c[1][2][4];
        buf0[41] += bIn(i + hipThreadIdx_x, j + 1, k + 7) * c[2][2][4];
        buf0[41] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 7) * c[3][2][4];
        buf0[41] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 7) * c[4][2][4];
        buf0[56] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 7) * c[0][3][2];
        buf0[56] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 7) * c[1][3][2];
        buf0[56] += bIn(i + hipThreadIdx_x, j + 1, k + 7) * c[2][3][2];
        buf0[56] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 7) * c[3][3][2];
        buf0[56] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 7) * c[4][3][2];
        buf0[48] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 7) * c[0][3][3];
        buf0[48] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 7) * c[1][3][3];
        buf0[48] += bIn(i + hipThreadIdx_x, j + 1, k + 7) * c[2][3][3];
        buf0[48] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 7) * c[3][3][3];
        buf0[48] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 7) * c[4][3][3];
        buf0[40] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 7) * c[0][3][4];
        buf0[40] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 7) * c[1][3][4];
        buf0[40] += bIn(i + hipThreadIdx_x, j + 1, k + 7) * c[2][3][4];
        buf0[40] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 7) * c[3][3][4];
        buf0[40] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 7) * c[4][3][4];
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 4; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[60 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 7) * c[0][0][2];
            buf0[60 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 7) * c[1][0][2];
            buf0[60 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7) * c[2][0][2];
            buf0[60 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7) * c[3][0][2];
            buf0[60 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 7) * c[4][0][2];
            buf0[52 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 7) * c[0][0][3];
            buf0[52 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 7) * c[1][0][3];
            buf0[52 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7) * c[2][0][3];
            buf0[52 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7) * c[3][0][3];
            buf0[52 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 7) * c[4][0][3];
            buf0[44 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 7) * c[0][0][4];
            buf0[44 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 7) * c[1][0][4];
            buf0[44 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7) * c[2][0][4];
            buf0[44 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7) * c[3][0][4];
            buf0[44 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 7) * c[4][0][4];
            buf0[59 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 7) * c[0][1][2];
            buf0[59 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 7) * c[1][1][2];
            buf0[59 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7) * c[2][1][2];
            buf0[59 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7) * c[3][1][2];
            buf0[59 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 7) * c[4][1][2];
            buf0[51 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 7) * c[0][1][3];
            buf0[51 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 7) * c[1][1][3];
            buf0[51 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7) * c[2][1][3];
            buf0[51 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7) * c[3][1][3];
            buf0[51 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 7) * c[4][1][3];
            buf0[43 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 7) * c[0][1][4];
            buf0[43 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 7) * c[1][1][4];
            buf0[43 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7) * c[2][1][4];
            buf0[43 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7) * c[3][1][4];
            buf0[43 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 7) * c[4][1][4];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 7) * c[0][2][2];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 7) * c[1][2][2];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7) * c[2][2][2];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7) * c[3][2][2];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 7) * c[4][2][2];
            buf0[50 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 7) * c[0][2][3];
            buf0[50 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 7) * c[1][2][3];
            buf0[50 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7) * c[2][2][3];
            buf0[50 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7) * c[3][2][3];
            buf0[50 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 7) * c[4][2][3];
            buf0[42 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 7) * c[0][2][4];
            buf0[42 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 7) * c[1][2][4];
            buf0[42 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7) * c[2][2][4];
            buf0[42 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7) * c[3][2][4];
            buf0[42 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 7) * c[4][2][4];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 7) * c[0][3][2];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 7) * c[1][3][2];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7) * c[2][3][2];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7) * c[3][3][2];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 7) * c[4][3][2];
            buf0[49 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 7) * c[0][3][3];
            buf0[49 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 7) * c[1][3][3];
            buf0[49 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7) * c[2][3][3];
            buf0[49 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7) * c[3][3][3];
            buf0[49 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 7) * c[4][3][3];
            buf0[41 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 7) * c[0][3][4];
            buf0[41 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 7) * c[1][3][4];
            buf0[41 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7) * c[2][3][4];
            buf0[41 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7) * c[3][3][4];
            buf0[41 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 7) * c[4][3][4];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 7) * c[0][4][2];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 7) * c[1][4][2];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7) * c[2][4][2];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7) * c[3][4][2];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 7) * c[4][4][2];
            buf0[48 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 7) * c[0][4][3];
            buf0[48 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 7) * c[1][4][3];
            buf0[48 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7) * c[2][4][3];
            buf0[48 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7) * c[3][4][3];
            buf0[48 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 7) * c[4][4][3];
            buf0[40 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 7) * c[0][4][4];
            buf0[40 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 7) * c[1][4][4];
            buf0[40 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7) * c[2][4][4];
            buf0[40 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7) * c[3][4][4];
            buf0[40 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 7) * c[4][4][4];
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[63] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 7) * c[0][1][2];
        buf0[63] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 7) * c[1][1][2];
        buf0[63] += bIn(i + hipThreadIdx_x, j + 6, k + 7) * c[2][1][2];
        buf0[63] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 7) * c[3][1][2];
        buf0[63] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 7) * c[4][1][2];
        buf0[55] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 7) * c[0][1][3];
        buf0[55] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 7) * c[1][1][3];
        buf0[55] += bIn(i + hipThreadIdx_x, j + 6, k + 7) * c[2][1][3];
        buf0[55] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 7) * c[3][1][3];
        buf0[55] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 7) * c[4][1][3];
        buf0[47] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 7) * c[0][1][4];
        buf0[47] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 7) * c[1][1][4];
        buf0[47] += bIn(i + hipThreadIdx_x, j + 6, k + 7) * c[2][1][4];
        buf0[47] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 7) * c[3][1][4];
        buf0[47] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 7) * c[4][1][4];
        buf0[62] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 7) * c[0][2][2];
        buf0[62] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 7) * c[1][2][2];
        buf0[62] += bIn(i + hipThreadIdx_x, j + 6, k + 7) * c[2][2][2];
        buf0[62] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 7) * c[3][2][2];
        buf0[62] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 7) * c[4][2][2];
        buf0[54] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 7) * c[0][2][3];
        buf0[54] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 7) * c[1][2][3];
        buf0[54] += bIn(i + hipThreadIdx_x, j + 6, k + 7) * c[2][2][3];
        buf0[54] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 7) * c[3][2][3];
        buf0[54] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 7) * c[4][2][3];
        buf0[46] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 7) * c[0][2][4];
        buf0[46] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 7) * c[1][2][4];
        buf0[46] += bIn(i + hipThreadIdx_x, j + 6, k + 7) * c[2][2][4];
        buf0[46] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 7) * c[3][2][4];
        buf0[46] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 7) * c[4][2][4];
        buf0[61] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 7) * c[0][3][2];
        buf0[61] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 7) * c[1][3][2];
        buf0[61] += bIn(i + hipThreadIdx_x, j + 6, k + 7) * c[2][3][2];
        buf0[61] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 7) * c[3][3][2];
        buf0[61] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 7) * c[4][3][2];
        buf0[53] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 7) * c[0][3][3];
        buf0[53] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 7) * c[1][3][3];
        buf0[53] += bIn(i + hipThreadIdx_x, j + 6, k + 7) * c[2][3][3];
        buf0[53] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 7) * c[3][3][3];
        buf0[53] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 7) * c[4][3][3];
        buf0[45] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 7) * c[0][3][4];
        buf0[45] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 7) * c[1][3][4];
        buf0[45] += bIn(i + hipThreadIdx_x, j + 6, k + 7) * c[2][3][4];
        buf0[45] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 7) * c[3][3][4];
        buf0[45] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 7) * c[4][3][4];
        buf0[60] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 7) * c[0][4][2];
        buf0[60] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 7) * c[1][4][2];
        buf0[60] += bIn(i + hipThreadIdx_x, j + 6, k + 7) * c[2][4][2];
        buf0[60] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 7) * c[3][4][2];
        buf0[60] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 7) * c[4][4][2];
        buf0[52] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 7) * c[0][4][3];
        buf0[52] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 7) * c[1][4][3];
        buf0[52] += bIn(i + hipThreadIdx_x, j + 6, k + 7) * c[2][4][3];
        buf0[52] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 7) * c[3][4][3];
        buf0[52] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 7) * c[4][4][3];
        buf0[44] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 7) * c[0][4][4];
        buf0[44] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 7) * c[1][4][4];
        buf0[44] += bIn(i + hipThreadIdx_x, j + 6, k + 7) * c[2][4][4];
        buf0[44] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 7) * c[3][4][4];
        buf0[44] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 7) * c[4][4][4];
      }
      {
        buf0[63] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 7) * c[0][2][2];
        buf0[63] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 7) * c[1][2][2];
        buf0[63] += bIn(i + hipThreadIdx_x, j + 7, k + 7) * c[2][2][2];
        buf0[63] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 7) * c[3][2][2];
        buf0[63] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 7) * c[4][2][2];
        buf0[55] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 7) * c[0][2][3];
        buf0[55] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 7) * c[1][2][3];
        buf0[55] += bIn(i + hipThreadIdx_x, j + 7, k + 7) * c[2][2][3];
        buf0[55] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 7) * c[3][2][3];
        buf0[55] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 7) * c[4][2][3];
        buf0[47] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 7) * c[0][2][4];
        buf0[47] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 7) * c[1][2][4];
        buf0[47] += bIn(i + hipThreadIdx_x, j + 7, k + 7) * c[2][2][4];
        buf0[47] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 7) * c[3][2][4];
        buf0[47] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 7) * c[4][2][4];
        buf0[62] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 7) * c[0][3][2];
        buf0[62] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 7) * c[1][3][2];
        buf0[62] += bIn(i + hipThreadIdx_x, j + 7, k + 7) * c[2][3][2];
        buf0[62] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 7) * c[3][3][2];
        buf0[62] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 7) * c[4][3][2];
        buf0[54] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 7) * c[0][3][3];
        buf0[54] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 7) * c[1][3][3];
        buf0[54] += bIn(i + hipThreadIdx_x, j + 7, k + 7) * c[2][3][3];
        buf0[54] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 7) * c[3][3][3];
        buf0[54] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 7) * c[4][3][3];
        buf0[46] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 7) * c[0][3][4];
        buf0[46] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 7) * c[1][3][4];
        buf0[46] += bIn(i + hipThreadIdx_x, j + 7, k + 7) * c[2][3][4];
        buf0[46] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 7) * c[3][3][4];
        buf0[46] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 7) * c[4][3][4];
        buf0[61] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 7) * c[0][4][2];
        buf0[61] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 7) * c[1][4][2];
        buf0[61] += bIn(i + hipThreadIdx_x, j + 7, k + 7) * c[2][4][2];
        buf0[61] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 7) * c[3][4][2];
        buf0[61] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 7) * c[4][4][2];
        buf0[53] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 7) * c[0][4][3];
        buf0[53] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 7) * c[1][4][3];
        buf0[53] += bIn(i + hipThreadIdx_x, j + 7, k + 7) * c[2][4][3];
        buf0[53] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 7) * c[3][4][3];
        buf0[53] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 7) * c[4][4][3];
        buf0[45] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 7) * c[0][4][4];
        buf0[45] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 7) * c[1][4][4];
        buf0[45] += bIn(i + hipThreadIdx_x, j + 7, k + 7) * c[2][4][4];
        buf0[45] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 7) * c[3][4][4];
        buf0[45] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 7) * c[4][4][4];
      }
      {
        buf0[63] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 7) * c[0][3][2];
        buf0[63] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 7) * c[1][3][2];
        buf0[63] += bIn(i + hipThreadIdx_x, j + 8, k + 7) * c[2][3][2];
        buf0[63] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 7) * c[3][3][2];
        buf0[63] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 7) * c[4][3][2];
        buf0[55] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 7) * c[0][3][3];
        buf0[55] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 7) * c[1][3][3];
        buf0[55] += bIn(i + hipThreadIdx_x, j + 8, k + 7) * c[2][3][3];
        buf0[55] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 7) * c[3][3][3];
        buf0[55] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 7) * c[4][3][3];
        buf0[47] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 7) * c[0][3][4];
        buf0[47] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 7) * c[1][3][4];
        buf0[47] += bIn(i + hipThreadIdx_x, j + 8, k + 7) * c[2][3][4];
        buf0[47] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 7) * c[3][3][4];
        buf0[47] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 7) * c[4][3][4];
        buf0[62] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 7) * c[0][4][2];
        buf0[62] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 7) * c[1][4][2];
        buf0[62] += bIn(i + hipThreadIdx_x, j + 8, k + 7) * c[2][4][2];
        buf0[62] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 7) * c[3][4][2];
        buf0[62] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 7) * c[4][4][2];
        buf0[54] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 7) * c[0][4][3];
        buf0[54] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 7) * c[1][4][3];
        buf0[54] += bIn(i + hipThreadIdx_x, j + 8, k + 7) * c[2][4][3];
        buf0[54] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 7) * c[3][4][3];
        buf0[54] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 7) * c[4][4][3];
        buf0[46] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 7) * c[0][4][4];
        buf0[46] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 7) * c[1][4][4];
        buf0[46] += bIn(i + hipThreadIdx_x, j + 8, k + 7) * c[2][4][4];
        buf0[46] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 7) * c[3][4][4];
        buf0[46] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 7) * c[4][4][4];
      }
      {
        buf0[63] += bIn(i + hipThreadIdx_x + -2, j + 9, k + 7) * c[0][4][2];
        buf0[63] += bIn(i + hipThreadIdx_x + -1, j + 9, k + 7) * c[1][4][2];
        buf0[63] += bIn(i + hipThreadIdx_x, j + 9, k + 7) * c[2][4][2];
        buf0[63] += bIn(i + hipThreadIdx_x + 1, j + 9, k + 7) * c[3][4][2];
        buf0[63] += bIn(i + hipThreadIdx_x + 2, j + 9, k + 7) * c[4][4][2];
        buf0[55] += bIn(i + hipThreadIdx_x + -2, j + 9, k + 7) * c[0][4][3];
        buf0[55] += bIn(i + hipThreadIdx_x + -1, j + 9, k + 7) * c[1][4][3];
        buf0[55] += bIn(i + hipThreadIdx_x, j + 9, k + 7) * c[2][4][3];
        buf0[55] += bIn(i + hipThreadIdx_x + 1, j + 9, k + 7) * c[3][4][3];
        buf0[55] += bIn(i + hipThreadIdx_x + 2, j + 9, k + 7) * c[4][4][3];
        buf0[47] += bIn(i + hipThreadIdx_x + -2, j + 9, k + 7) * c[0][4][4];
        buf0[47] += bIn(i + hipThreadIdx_x + -1, j + 9, k + 7) * c[1][4][4];
        buf0[47] += bIn(i + hipThreadIdx_x, j + 9, k + 7) * c[2][4][4];
        buf0[47] += bIn(i + hipThreadIdx_x + 1, j + 9, k + 7) * c[3][4][4];
        buf0[47] += bIn(i + hipThreadIdx_x + 2, j + 9, k + 7) * c[4][4][4];
      }
      {
        buf0[56] += bIn(i + hipThreadIdx_x + -2, j + -2, k + 8) * c[0][0][3];
        buf0[56] += bIn(i + hipThreadIdx_x + -1, j + -2, k + 8) * c[1][0][3];
        buf0[56] += bIn(i + hipThreadIdx_x, j + -2, k + 8) * c[2][0][3];
        buf0[56] += bIn(i + hipThreadIdx_x + 1, j + -2, k + 8) * c[3][0][3];
        buf0[56] += bIn(i + hipThreadIdx_x + 2, j + -2, k + 8) * c[4][0][3];
        buf0[48] += bIn(i + hipThreadIdx_x + -2, j + -2, k + 8) * c[0][0][4];
        buf0[48] += bIn(i + hipThreadIdx_x + -1, j + -2, k + 8) * c[1][0][4];
        buf0[48] += bIn(i + hipThreadIdx_x, j + -2, k + 8) * c[2][0][4];
        buf0[48] += bIn(i + hipThreadIdx_x + 1, j + -2, k + 8) * c[3][0][4];
        buf0[48] += bIn(i + hipThreadIdx_x + 2, j + -2, k + 8) * c[4][0][4];
      }
      {
        buf0[57] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 8) * c[0][0][3];
        buf0[57] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 8) * c[1][0][3];
        buf0[57] += bIn(i + hipThreadIdx_x, j + -1, k + 8) * c[2][0][3];
        buf0[57] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 8) * c[3][0][3];
        buf0[57] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 8) * c[4][0][3];
        buf0[49] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 8) * c[0][0][4];
        buf0[49] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 8) * c[1][0][4];
        buf0[49] += bIn(i + hipThreadIdx_x, j + -1, k + 8) * c[2][0][4];
        buf0[49] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 8) * c[3][0][4];
        buf0[49] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 8) * c[4][0][4];
        buf0[56] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 8) * c[0][1][3];
        buf0[56] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 8) * c[1][1][3];
        buf0[56] += bIn(i + hipThreadIdx_x, j + -1, k + 8) * c[2][1][3];
        buf0[56] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 8) * c[3][1][3];
        buf0[56] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 8) * c[4][1][3];
        buf0[48] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 8) * c[0][1][4];
        buf0[48] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 8) * c[1][1][4];
        buf0[48] += bIn(i + hipThreadIdx_x, j + -1, k + 8) * c[2][1][4];
        buf0[48] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 8) * c[3][1][4];
        buf0[48] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 8) * c[4][1][4];
      }
      {
        buf0[58] += bIn(i + hipThreadIdx_x + -2, j, k + 8) * c[0][0][3];
        buf0[58] += bIn(i + hipThreadIdx_x + -1, j, k + 8) * c[1][0][3];
        buf0[58] += bIn(i + hipThreadIdx_x, j, k + 8) * c[2][0][3];
        buf0[58] += bIn(i + hipThreadIdx_x + 1, j, k + 8) * c[3][0][3];
        buf0[58] += bIn(i + hipThreadIdx_x + 2, j, k + 8) * c[4][0][3];
        buf0[50] += bIn(i + hipThreadIdx_x + -2, j, k + 8) * c[0][0][4];
        buf0[50] += bIn(i + hipThreadIdx_x + -1, j, k + 8) * c[1][0][4];
        buf0[50] += bIn(i + hipThreadIdx_x, j, k + 8) * c[2][0][4];
        buf0[50] += bIn(i + hipThreadIdx_x + 1, j, k + 8) * c[3][0][4];
        buf0[50] += bIn(i + hipThreadIdx_x + 2, j, k + 8) * c[4][0][4];
        buf0[57] += bIn(i + hipThreadIdx_x + -2, j, k + 8) * c[0][1][3];
        buf0[57] += bIn(i + hipThreadIdx_x + -1, j, k + 8) * c[1][1][3];
        buf0[57] += bIn(i + hipThreadIdx_x, j, k + 8) * c[2][1][3];
        buf0[57] += bIn(i + hipThreadIdx_x + 1, j, k + 8) * c[3][1][3];
        buf0[57] += bIn(i + hipThreadIdx_x + 2, j, k + 8) * c[4][1][3];
        buf0[49] += bIn(i + hipThreadIdx_x + -2, j, k + 8) * c[0][1][4];
        buf0[49] += bIn(i + hipThreadIdx_x + -1, j, k + 8) * c[1][1][4];
        buf0[49] += bIn(i + hipThreadIdx_x, j, k + 8) * c[2][1][4];
        buf0[49] += bIn(i + hipThreadIdx_x + 1, j, k + 8) * c[3][1][4];
        buf0[49] += bIn(i + hipThreadIdx_x + 2, j, k + 8) * c[4][1][4];
        buf0[56] += bIn(i + hipThreadIdx_x + -2, j, k + 8) * c[0][2][3];
        buf0[56] += bIn(i + hipThreadIdx_x + -1, j, k + 8) * c[1][2][3];
        buf0[56] += bIn(i + hipThreadIdx_x, j, k + 8) * c[2][2][3];
        buf0[56] += bIn(i + hipThreadIdx_x + 1, j, k + 8) * c[3][2][3];
        buf0[56] += bIn(i + hipThreadIdx_x + 2, j, k + 8) * c[4][2][3];
        buf0[48] += bIn(i + hipThreadIdx_x + -2, j, k + 8) * c[0][2][4];
        buf0[48] += bIn(i + hipThreadIdx_x + -1, j, k + 8) * c[1][2][4];
        buf0[48] += bIn(i + hipThreadIdx_x, j, k + 8) * c[2][2][4];
        buf0[48] += bIn(i + hipThreadIdx_x + 1, j, k + 8) * c[3][2][4];
        buf0[48] += bIn(i + hipThreadIdx_x + 2, j, k + 8) * c[4][2][4];
      }
      {
        buf0[59] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 8) * c[0][0][3];
        buf0[59] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 8) * c[1][0][3];
        buf0[59] += bIn(i + hipThreadIdx_x, j + 1, k + 8) * c[2][0][3];
        buf0[59] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 8) * c[3][0][3];
        buf0[59] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 8) * c[4][0][3];
        buf0[51] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 8) * c[0][0][4];
        buf0[51] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 8) * c[1][0][4];
        buf0[51] += bIn(i + hipThreadIdx_x, j + 1, k + 8) * c[2][0][4];
        buf0[51] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 8) * c[3][0][4];
        buf0[51] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 8) * c[4][0][4];
        buf0[58] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 8) * c[0][1][3];
        buf0[58] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 8) * c[1][1][3];
        buf0[58] += bIn(i + hipThreadIdx_x, j + 1, k + 8) * c[2][1][3];
        buf0[58] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 8) * c[3][1][3];
        buf0[58] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 8) * c[4][1][3];
        buf0[50] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 8) * c[0][1][4];
        buf0[50] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 8) * c[1][1][4];
        buf0[50] += bIn(i + hipThreadIdx_x, j + 1, k + 8) * c[2][1][4];
        buf0[50] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 8) * c[3][1][4];
        buf0[50] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 8) * c[4][1][4];
        buf0[57] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 8) * c[0][2][3];
        buf0[57] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 8) * c[1][2][3];
        buf0[57] += bIn(i + hipThreadIdx_x, j + 1, k + 8) * c[2][2][3];
        buf0[57] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 8) * c[3][2][3];
        buf0[57] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 8) * c[4][2][3];
        buf0[49] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 8) * c[0][2][4];
        buf0[49] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 8) * c[1][2][4];
        buf0[49] += bIn(i + hipThreadIdx_x, j + 1, k + 8) * c[2][2][4];
        buf0[49] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 8) * c[3][2][4];
        buf0[49] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 8) * c[4][2][4];
        buf0[56] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 8) * c[0][3][3];
        buf0[56] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 8) * c[1][3][3];
        buf0[56] += bIn(i + hipThreadIdx_x, j + 1, k + 8) * c[2][3][3];
        buf0[56] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 8) * c[3][3][3];
        buf0[56] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 8) * c[4][3][3];
        buf0[48] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 8) * c[0][3][4];
        buf0[48] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 8) * c[1][3][4];
        buf0[48] += bIn(i + hipThreadIdx_x, j + 1, k + 8) * c[2][3][4];
        buf0[48] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 8) * c[3][3][4];
        buf0[48] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 8) * c[4][3][4];
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 4; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[60 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 8) * c[0][0][3];
            buf0[60 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 8) * c[1][0][3];
            buf0[60 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 8) * c[2][0][3];
            buf0[60 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 8) * c[3][0][3];
            buf0[60 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 8) * c[4][0][3];
            buf0[52 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 8) * c[0][0][4];
            buf0[52 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 8) * c[1][0][4];
            buf0[52 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 8) * c[2][0][4];
            buf0[52 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 8) * c[3][0][4];
            buf0[52 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 8) * c[4][0][4];
            buf0[59 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 8) * c[0][1][3];
            buf0[59 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 8) * c[1][1][3];
            buf0[59 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 8) * c[2][1][3];
            buf0[59 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 8) * c[3][1][3];
            buf0[59 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 8) * c[4][1][3];
            buf0[51 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 8) * c[0][1][4];
            buf0[51 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 8) * c[1][1][4];
            buf0[51 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 8) * c[2][1][4];
            buf0[51 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 8) * c[3][1][4];
            buf0[51 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 8) * c[4][1][4];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 8) * c[0][2][3];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 8) * c[1][2][3];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 8) * c[2][2][3];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 8) * c[3][2][3];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 8) * c[4][2][3];
            buf0[50 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 8) * c[0][2][4];
            buf0[50 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 8) * c[1][2][4];
            buf0[50 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 8) * c[2][2][4];
            buf0[50 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 8) * c[3][2][4];
            buf0[50 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 8) * c[4][2][4];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 8) * c[0][3][3];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 8) * c[1][3][3];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 8) * c[2][3][3];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 8) * c[3][3][3];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 8) * c[4][3][3];
            buf0[49 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 8) * c[0][3][4];
            buf0[49 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 8) * c[1][3][4];
            buf0[49 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 8) * c[2][3][4];
            buf0[49 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 8) * c[3][3][4];
            buf0[49 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 8) * c[4][3][4];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 8) * c[0][4][3];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 8) * c[1][4][3];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 8) * c[2][4][3];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 8) * c[3][4][3];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 8) * c[4][4][3];
            buf0[48 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 8) * c[0][4][4];
            buf0[48 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 8) * c[1][4][4];
            buf0[48 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 8) * c[2][4][4];
            buf0[48 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 8) * c[3][4][4];
            buf0[48 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 8) * c[4][4][4];
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[63] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 8) * c[0][1][3];
        buf0[63] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 8) * c[1][1][3];
        buf0[63] += bIn(i + hipThreadIdx_x, j + 6, k + 8) * c[2][1][3];
        buf0[63] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 8) * c[3][1][3];
        buf0[63] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 8) * c[4][1][3];
        buf0[55] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 8) * c[0][1][4];
        buf0[55] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 8) * c[1][1][4];
        buf0[55] += bIn(i + hipThreadIdx_x, j + 6, k + 8) * c[2][1][4];
        buf0[55] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 8) * c[3][1][4];
        buf0[55] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 8) * c[4][1][4];
        buf0[62] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 8) * c[0][2][3];
        buf0[62] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 8) * c[1][2][3];
        buf0[62] += bIn(i + hipThreadIdx_x, j + 6, k + 8) * c[2][2][3];
        buf0[62] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 8) * c[3][2][3];
        buf0[62] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 8) * c[4][2][3];
        buf0[54] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 8) * c[0][2][4];
        buf0[54] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 8) * c[1][2][4];
        buf0[54] += bIn(i + hipThreadIdx_x, j + 6, k + 8) * c[2][2][4];
        buf0[54] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 8) * c[3][2][4];
        buf0[54] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 8) * c[4][2][4];
        buf0[61] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 8) * c[0][3][3];
        buf0[61] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 8) * c[1][3][3];
        buf0[61] += bIn(i + hipThreadIdx_x, j + 6, k + 8) * c[2][3][3];
        buf0[61] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 8) * c[3][3][3];
        buf0[61] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 8) * c[4][3][3];
        buf0[53] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 8) * c[0][3][4];
        buf0[53] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 8) * c[1][3][4];
        buf0[53] += bIn(i + hipThreadIdx_x, j + 6, k + 8) * c[2][3][4];
        buf0[53] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 8) * c[3][3][4];
        buf0[53] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 8) * c[4][3][4];
        buf0[60] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 8) * c[0][4][3];
        buf0[60] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 8) * c[1][4][3];
        buf0[60] += bIn(i + hipThreadIdx_x, j + 6, k + 8) * c[2][4][3];
        buf0[60] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 8) * c[3][4][3];
        buf0[60] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 8) * c[4][4][3];
        buf0[52] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 8) * c[0][4][4];
        buf0[52] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 8) * c[1][4][4];
        buf0[52] += bIn(i + hipThreadIdx_x, j + 6, k + 8) * c[2][4][4];
        buf0[52] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 8) * c[3][4][4];
        buf0[52] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 8) * c[4][4][4];
      }
      {
        buf0[63] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 8) * c[0][2][3];
        buf0[63] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 8) * c[1][2][3];
        buf0[63] += bIn(i + hipThreadIdx_x, j + 7, k + 8) * c[2][2][3];
        buf0[63] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 8) * c[3][2][3];
        buf0[63] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 8) * c[4][2][3];
        buf0[55] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 8) * c[0][2][4];
        buf0[55] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 8) * c[1][2][4];
        buf0[55] += bIn(i + hipThreadIdx_x, j + 7, k + 8) * c[2][2][4];
        buf0[55] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 8) * c[3][2][4];
        buf0[55] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 8) * c[4][2][4];
        buf0[62] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 8) * c[0][3][3];
        buf0[62] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 8) * c[1][3][3];
        buf0[62] += bIn(i + hipThreadIdx_x, j + 7, k + 8) * c[2][3][3];
        buf0[62] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 8) * c[3][3][3];
        buf0[62] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 8) * c[4][3][3];
        buf0[54] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 8) * c[0][3][4];
        buf0[54] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 8) * c[1][3][4];
        buf0[54] += bIn(i + hipThreadIdx_x, j + 7, k + 8) * c[2][3][4];
        buf0[54] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 8) * c[3][3][4];
        buf0[54] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 8) * c[4][3][4];
        buf0[61] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 8) * c[0][4][3];
        buf0[61] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 8) * c[1][4][3];
        buf0[61] += bIn(i + hipThreadIdx_x, j + 7, k + 8) * c[2][4][3];
        buf0[61] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 8) * c[3][4][3];
        buf0[61] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 8) * c[4][4][3];
        buf0[53] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 8) * c[0][4][4];
        buf0[53] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 8) * c[1][4][4];
        buf0[53] += bIn(i + hipThreadIdx_x, j + 7, k + 8) * c[2][4][4];
        buf0[53] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 8) * c[3][4][4];
        buf0[53] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 8) * c[4][4][4];
      }
      {
        buf0[63] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 8) * c[0][3][3];
        buf0[63] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 8) * c[1][3][3];
        buf0[63] += bIn(i + hipThreadIdx_x, j + 8, k + 8) * c[2][3][3];
        buf0[63] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 8) * c[3][3][3];
        buf0[63] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 8) * c[4][3][3];
        buf0[55] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 8) * c[0][3][4];
        buf0[55] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 8) * c[1][3][4];
        buf0[55] += bIn(i + hipThreadIdx_x, j + 8, k + 8) * c[2][3][4];
        buf0[55] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 8) * c[3][3][4];
        buf0[55] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 8) * c[4][3][4];
        buf0[62] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 8) * c[0][4][3];
        buf0[62] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 8) * c[1][4][3];
        buf0[62] += bIn(i + hipThreadIdx_x, j + 8, k + 8) * c[2][4][3];
        buf0[62] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 8) * c[3][4][3];
        buf0[62] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 8) * c[4][4][3];
        buf0[54] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 8) * c[0][4][4];
        buf0[54] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 8) * c[1][4][4];
        buf0[54] += bIn(i + hipThreadIdx_x, j + 8, k + 8) * c[2][4][4];
        buf0[54] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 8) * c[3][4][4];
        buf0[54] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 8) * c[4][4][4];
      }
      {
        buf0[63] += bIn(i + hipThreadIdx_x + -2, j + 9, k + 8) * c[0][4][3];
        buf0[63] += bIn(i + hipThreadIdx_x + -1, j + 9, k + 8) * c[1][4][3];
        buf0[63] += bIn(i + hipThreadIdx_x, j + 9, k + 8) * c[2][4][3];
        buf0[63] += bIn(i + hipThreadIdx_x + 1, j + 9, k + 8) * c[3][4][3];
        buf0[63] += bIn(i + hipThreadIdx_x + 2, j + 9, k + 8) * c[4][4][3];
        buf0[55] += bIn(i + hipThreadIdx_x + -2, j + 9, k + 8) * c[0][4][4];
        buf0[55] += bIn(i + hipThreadIdx_x + -1, j + 9, k + 8) * c[1][4][4];
        buf0[55] += bIn(i + hipThreadIdx_x, j + 9, k + 8) * c[2][4][4];
        buf0[55] += bIn(i + hipThreadIdx_x + 1, j + 9, k + 8) * c[3][4][4];
        buf0[55] += bIn(i + hipThreadIdx_x + 2, j + 9, k + 8) * c[4][4][4];
      }
      {
        buf0[56] += bIn(i + hipThreadIdx_x + -2, j + -2, k + 9) * c[0][0][4];
        buf0[56] += bIn(i + hipThreadIdx_x + -1, j + -2, k + 9) * c[1][0][4];
        buf0[56] += bIn(i + hipThreadIdx_x, j + -2, k + 9) * c[2][0][4];
        buf0[56] += bIn(i + hipThreadIdx_x + 1, j + -2, k + 9) * c[3][0][4];
        buf0[56] += bIn(i + hipThreadIdx_x + 2, j + -2, k + 9) * c[4][0][4];
      }
      {
        buf0[57] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 9) * c[0][0][4];
        buf0[57] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 9) * c[1][0][4];
        buf0[57] += bIn(i + hipThreadIdx_x, j + -1, k + 9) * c[2][0][4];
        buf0[57] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 9) * c[3][0][4];
        buf0[57] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 9) * c[4][0][4];
        buf0[56] += bIn(i + hipThreadIdx_x + -2, j + -1, k + 9) * c[0][1][4];
        buf0[56] += bIn(i + hipThreadIdx_x + -1, j + -1, k + 9) * c[1][1][4];
        buf0[56] += bIn(i + hipThreadIdx_x, j + -1, k + 9) * c[2][1][4];
        buf0[56] += bIn(i + hipThreadIdx_x + 1, j + -1, k + 9) * c[3][1][4];
        buf0[56] += bIn(i + hipThreadIdx_x + 2, j + -1, k + 9) * c[4][1][4];
      }
      {
        buf0[58] += bIn(i + hipThreadIdx_x + -2, j, k + 9) * c[0][0][4];
        buf0[58] += bIn(i + hipThreadIdx_x + -1, j, k + 9) * c[1][0][4];
        buf0[58] += bIn(i + hipThreadIdx_x, j, k + 9) * c[2][0][4];
        buf0[58] += bIn(i + hipThreadIdx_x + 1, j, k + 9) * c[3][0][4];
        buf0[58] += bIn(i + hipThreadIdx_x + 2, j, k + 9) * c[4][0][4];
        buf0[57] += bIn(i + hipThreadIdx_x + -2, j, k + 9) * c[0][1][4];
        buf0[57] += bIn(i + hipThreadIdx_x + -1, j, k + 9) * c[1][1][4];
        buf0[57] += bIn(i + hipThreadIdx_x, j, k + 9) * c[2][1][4];
        buf0[57] += bIn(i + hipThreadIdx_x + 1, j, k + 9) * c[3][1][4];
        buf0[57] += bIn(i + hipThreadIdx_x + 2, j, k + 9) * c[4][1][4];
        buf0[56] += bIn(i + hipThreadIdx_x + -2, j, k + 9) * c[0][2][4];
        buf0[56] += bIn(i + hipThreadIdx_x + -1, j, k + 9) * c[1][2][4];
        buf0[56] += bIn(i + hipThreadIdx_x, j, k + 9) * c[2][2][4];
        buf0[56] += bIn(i + hipThreadIdx_x + 1, j, k + 9) * c[3][2][4];
        buf0[56] += bIn(i + hipThreadIdx_x + 2, j, k + 9) * c[4][2][4];
      }
      {
        buf0[59] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 9) * c[0][0][4];
        buf0[59] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 9) * c[1][0][4];
        buf0[59] += bIn(i + hipThreadIdx_x, j + 1, k + 9) * c[2][0][4];
        buf0[59] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 9) * c[3][0][4];
        buf0[59] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 9) * c[4][0][4];
        buf0[58] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 9) * c[0][1][4];
        buf0[58] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 9) * c[1][1][4];
        buf0[58] += bIn(i + hipThreadIdx_x, j + 1, k + 9) * c[2][1][4];
        buf0[58] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 9) * c[3][1][4];
        buf0[58] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 9) * c[4][1][4];
        buf0[57] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 9) * c[0][2][4];
        buf0[57] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 9) * c[1][2][4];
        buf0[57] += bIn(i + hipThreadIdx_x, j + 1, k + 9) * c[2][2][4];
        buf0[57] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 9) * c[3][2][4];
        buf0[57] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 9) * c[4][2][4];
        buf0[56] += bIn(i + hipThreadIdx_x + -2, j + 1, k + 9) * c[0][3][4];
        buf0[56] += bIn(i + hipThreadIdx_x + -1, j + 1, k + 9) * c[1][3][4];
        buf0[56] += bIn(i + hipThreadIdx_x, j + 1, k + 9) * c[2][3][4];
        buf0[56] += bIn(i + hipThreadIdx_x + 1, j + 1, k + 9) * c[3][3][4];
        buf0[56] += bIn(i + hipThreadIdx_x + 2, j + 1, k + 9) * c[4][3][4];
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 4; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[60 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 9) * c[0][0][4];
            buf0[60 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 9) * c[1][0][4];
            buf0[60 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 9) * c[2][0][4];
            buf0[60 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 9) * c[3][0][4];
            buf0[60 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 9) * c[4][0][4];
            buf0[59 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 9) * c[0][1][4];
            buf0[59 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 9) * c[1][1][4];
            buf0[59 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 9) * c[2][1][4];
            buf0[59 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 9) * c[3][1][4];
            buf0[59 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 9) * c[4][1][4];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 9) * c[0][2][4];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 9) * c[1][2][4];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 9) * c[2][2][4];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 9) * c[3][2][4];
            buf0[58 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 9) * c[4][2][4];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 9) * c[0][3][4];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 9) * c[1][3][4];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 9) * c[2][3][4];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 9) * c[3][3][4];
            buf0[57 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 9) * c[4][3][4];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 9) * c[0][4][4];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 9) * c[1][4][4];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 9) * c[2][4][4];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 9) * c[3][4][4];
            buf0[56 + rel] += bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 9) * c[4][4][4];
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[63] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 9) * c[0][1][4];
        buf0[63] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 9) * c[1][1][4];
        buf0[63] += bIn(i + hipThreadIdx_x, j + 6, k + 9) * c[2][1][4];
        buf0[63] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 9) * c[3][1][4];
        buf0[63] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 9) * c[4][1][4];
        buf0[62] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 9) * c[0][2][4];
        buf0[62] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 9) * c[1][2][4];
        buf0[62] += bIn(i + hipThreadIdx_x, j + 6, k + 9) * c[2][2][4];
        buf0[62] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 9) * c[3][2][4];
        buf0[62] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 9) * c[4][2][4];
        buf0[61] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 9) * c[0][3][4];
        buf0[61] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 9) * c[1][3][4];
        buf0[61] += bIn(i + hipThreadIdx_x, j + 6, k + 9) * c[2][3][4];
        buf0[61] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 9) * c[3][3][4];
        buf0[61] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 9) * c[4][3][4];
        buf0[60] += bIn(i + hipThreadIdx_x + -2, j + 6, k + 9) * c[0][4][4];
        buf0[60] += bIn(i + hipThreadIdx_x + -1, j + 6, k + 9) * c[1][4][4];
        buf0[60] += bIn(i + hipThreadIdx_x, j + 6, k + 9) * c[2][4][4];
        buf0[60] += bIn(i + hipThreadIdx_x + 1, j + 6, k + 9) * c[3][4][4];
        buf0[60] += bIn(i + hipThreadIdx_x + 2, j + 6, k + 9) * c[4][4][4];
      }
      {
        buf0[63] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 9) * c[0][2][4];
        buf0[63] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 9) * c[1][2][4];
        buf0[63] += bIn(i + hipThreadIdx_x, j + 7, k + 9) * c[2][2][4];
        buf0[63] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 9) * c[3][2][4];
        buf0[63] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 9) * c[4][2][4];
        buf0[62] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 9) * c[0][3][4];
        buf0[62] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 9) * c[1][3][4];
        buf0[62] += bIn(i + hipThreadIdx_x, j + 7, k + 9) * c[2][3][4];
        buf0[62] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 9) * c[3][3][4];
        buf0[62] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 9) * c[4][3][4];
        buf0[61] += bIn(i + hipThreadIdx_x + -2, j + 7, k + 9) * c[0][4][4];
        buf0[61] += bIn(i + hipThreadIdx_x + -1, j + 7, k + 9) * c[1][4][4];
        buf0[61] += bIn(i + hipThreadIdx_x, j + 7, k + 9) * c[2][4][4];
        buf0[61] += bIn(i + hipThreadIdx_x + 1, j + 7, k + 9) * c[3][4][4];
        buf0[61] += bIn(i + hipThreadIdx_x + 2, j + 7, k + 9) * c[4][4][4];
      }
      {
        buf0[63] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 9) * c[0][3][4];
        buf0[63] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 9) * c[1][3][4];
        buf0[63] += bIn(i + hipThreadIdx_x, j + 8, k + 9) * c[2][3][4];
        buf0[63] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 9) * c[3][3][4];
        buf0[63] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 9) * c[4][3][4];
        buf0[62] += bIn(i + hipThreadIdx_x + -2, j + 8, k + 9) * c[0][4][4];
        buf0[62] += bIn(i + hipThreadIdx_x + -1, j + 8, k + 9) * c[1][4][4];
        buf0[62] += bIn(i + hipThreadIdx_x, j + 8, k + 9) * c[2][4][4];
        buf0[62] += bIn(i + hipThreadIdx_x + 1, j + 8, k + 9) * c[3][4][4];
        buf0[62] += bIn(i + hipThreadIdx_x + 2, j + 8, k + 9) * c[4][4][4];
      }
      {
        buf0[63] += bIn(i + hipThreadIdx_x + -2, j + 9, k + 9) * c[0][4][4];
        buf0[63] += bIn(i + hipThreadIdx_x + -1, j + 9, k + 9) * c[1][4][4];
        buf0[63] += bIn(i + hipThreadIdx_x, j + 9, k + 9) * c[2][4][4];
        buf0[63] += bIn(i + hipThreadIdx_x + 1, j + 9, k + 9) * c[3][4][4];
        buf0[63] += bIn(i + hipThreadIdx_x + 2, j + 9, k + 9) * c[4][4][4];
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
            bOut(i + _cg_idx0, j + _cg_idx1, k + _cg_idx2) = buf0[rel];
          }
        }
      }
    }
  }
}
# 102 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/f3d/intermediate_gen/f3d.cu" 2

}
#undef bIn
#undef bOut
__global__ void f3d_codegen_bricks2(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem (*c)[8][8]) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
# 1 "VSBrick-f3d2.py-HIP-8x8x8-8x8" 1
{
  auto *binfo = bOut.bInfo;
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
      bElem _cg_bIn_1_2_2_vecbuf;
      bElem _cg_bIn0_2_2_vecbuf;
      bElem _cg_bIn_2_2_2_vecbuf;
      {
        // New offset [0, 0, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor3 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor0 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor4 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor1 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 6 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[0] += 0;
        buf0[0] += _cg_bIn_2_2_2_reg * c[0][0][0];
      }
      {
        // New offset [1, 0, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[1][0][0];
      }
      {
        // New offset [2, 0, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[2][0][0];
      }
      {
        // New offset [3, 0, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor5 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor2 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[3][0][0];
      }
      {
        // New offset [4, 0, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[4][0][0];
      }
      {
        // New offset [0, 1, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor3 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor0 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor4 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor1 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[0][1][0];
      }
      {
        // New offset [1, 1, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[1][1][0];
      }
      {
        // New offset [2, 1, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[2][1][0];
      }
      {
        // New offset [3, 1, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor5 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor2 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[3][1][0];
      }
      {
        // New offset [4, 1, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[4][1][0];
      }
      {
        // New offset [0, 2, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_1_2_2_vecbuf = bIn.dat[neighbor3 * bIn.step + 384 + hipThreadIdx_x];
          _cg_bIn0_2_2_vecbuf = bIn.dat[neighbor4 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[0][2][0];
      }
      {
        // New offset [1, 2, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[1][2][0];
      }
      {
        // New offset [2, 2, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[2][2][0];
      }
      {
        // New offset [3, 2, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn0_2_2_vecbuf = bIn.dat[neighbor5 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[3][2][0];
      }
      {
        // New offset [4, 2, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[4][2][0];
      }
      {
        // New offset [0, 3, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor6 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor3 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor7 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor4 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[0][3][0];
      }
      {
        // New offset [1, 3, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[1][3][0];
      }
      {
        // New offset [2, 3, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[2][3][0];
      }
      {
        // New offset [3, 3, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor8 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor5 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[3][3][0];
      }
      {
        // New offset [4, 3, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[4][3][0];
      }
      {
        // New offset [0, 4, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor6 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor3 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor7 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor4 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 2 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[0][4][0];
      }
      {
        // New offset [1, 4, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[1][4][0];
      }
      {
        // New offset [2, 4, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[2][4][0];
      }
      {
        // New offset [3, 4, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor8 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor5 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[3][4][0];
      }
      {
        // New offset [4, 4, 0]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[0] += _cg_bIn_2_2_2_reg * c[4][4][0];
      }
      {
        // New offset [0, 0, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor3 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor0 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor4 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor1 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 6 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[1] += 0;
        buf0[1] += _cg_bIn_2_2_2_reg * c[0][0][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[0][0][1];
      }
      {
        // New offset [1, 0, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[1][0][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[1][0][1];
      }
      {
        // New offset [2, 0, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[2][0][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[2][0][1];
      }
      {
        // New offset [3, 0, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor5 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor2 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[3][0][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[3][0][1];
      }
      {
        // New offset [4, 0, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[4][0][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[4][0][1];
      }
      {
        // New offset [0, 1, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor3 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor0 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor4 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor1 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[0][1][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[0][1][1];
      }
      {
        // New offset [1, 1, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[1][1][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[1][1][1];
      }
      {
        // New offset [2, 1, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[2][1][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[2][1][1];
      }
      {
        // New offset [3, 1, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor5 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor2 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[3][1][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[3][1][1];
      }
      {
        // New offset [4, 1, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[4][1][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[4][1][1];
      }
      {
        // New offset [0, 2, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_1_2_2_vecbuf = bIn.dat[neighbor3 * bIn.step + 448 + hipThreadIdx_x];
          _cg_bIn0_2_2_vecbuf = bIn.dat[neighbor4 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[0][2][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[0][2][1];
      }
      {
        // New offset [1, 2, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[1][2][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[1][2][1];
      }
      {
        // New offset [2, 2, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[2][2][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[2][2][1];
      }
      {
        // New offset [3, 2, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn0_2_2_vecbuf = bIn.dat[neighbor5 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[3][2][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[3][2][1];
      }
      {
        // New offset [4, 2, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[4][2][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[4][2][1];
      }
      {
        // New offset [0, 3, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor6 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor3 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor7 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor4 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[0][3][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[0][3][1];
      }
      {
        // New offset [1, 3, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[1][3][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[1][3][1];
      }
      {
        // New offset [2, 3, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[2][3][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[2][3][1];
      }
      {
        // New offset [3, 3, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor8 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor5 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[3][3][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[3][3][1];
      }
      {
        // New offset [4, 3, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[4][3][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[4][3][1];
      }
      {
        // New offset [0, 4, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor6 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor3 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor7 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor4 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 2 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[0][4][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[0][4][1];
      }
      {
        // New offset [1, 4, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[1][4][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[1][4][1];
      }
      {
        // New offset [2, 4, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[2][4][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[2][4][1];
      }
      {
        // New offset [3, 4, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor8 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor5 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[3][4][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[3][4][1];
      }
      {
        // New offset [4, 4, 1]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[1] += _cg_bIn_2_2_2_reg * c[4][4][0];
        buf0[0] += _cg_bIn_2_2_2_reg * c[4][4][1];
      }
      {
        // New offset [0, 0, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor12 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor9 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor10 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 6 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[2] += 0;
        buf0[2] += _cg_bIn_2_2_2_reg * c[0][0][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[0][0][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[0][0][2];
      }
      {
        // New offset [1, 0, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[1][0][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[1][0][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[1][0][2];
      }
      {
        // New offset [2, 0, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[2][0][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[2][0][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[2][0][2];
      }
      {
        // New offset [3, 0, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor14 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor11 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[3][0][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[3][0][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[3][0][2];
      }
      {
        // New offset [4, 0, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[4][0][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[4][0][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[4][0][2];
      }
      {
        // New offset [0, 1, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor12 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor9 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor10 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[0][1][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[0][1][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[0][1][2];
      }
      {
        // New offset [1, 1, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[1][1][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[1][1][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[1][1][2];
      }
      {
        // New offset [2, 1, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[2][1][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[2][1][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[2][1][2];
      }
      {
        // New offset [3, 1, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor14 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor11 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[3][1][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[3][1][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[3][1][2];
      }
      {
        // New offset [4, 1, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[4][1][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[4][1][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[4][1][2];
      }
      {
        // New offset [0, 2, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_1_2_2_vecbuf = bIn.dat[neighbor12 * bIn.step + hipThreadIdx_x];
          _cg_bIn0_2_2_vecbuf = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[0][2][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[0][2][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[0][2][2];
      }
      {
        // New offset [1, 2, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[1][2][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[1][2][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[1][2][2];
      }
      {
        // New offset [2, 2, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[2][2][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[2][2][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[2][2][2];
      }
      {
        // New offset [3, 2, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn0_2_2_vecbuf = bIn.dat[neighbor14 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[3][2][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[3][2][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[3][2][2];
      }
      {
        // New offset [4, 2, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[4][2][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[4][2][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[4][2][2];
      }
      {
        // New offset [0, 3, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor15 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor12 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor16 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[0][3][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[0][3][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[0][3][2];
      }
      {
        // New offset [1, 3, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[1][3][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[1][3][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[1][3][2];
      }
      {
        // New offset [2, 3, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[2][3][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[2][3][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[2][3][2];
      }
      {
        // New offset [3, 3, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor17 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor14 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[3][3][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[3][3][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[3][3][2];
      }
      {
        // New offset [4, 3, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[4][3][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[4][3][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[4][3][2];
      }
      {
        // New offset [0, 4, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor15 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor12 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor16 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 2 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[0][4][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[0][4][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[0][4][2];
      }
      {
        // New offset [1, 4, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[1][4][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[1][4][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[1][4][2];
      }
      {
        // New offset [2, 4, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[2][4][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[2][4][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[2][4][2];
      }
      {
        // New offset [3, 4, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor17 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor14 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[3][4][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[3][4][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[3][4][2];
      }
      {
        // New offset [4, 4, 2]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[2] += _cg_bIn_2_2_2_reg * c[4][4][0];
        buf0[1] += _cg_bIn_2_2_2_reg * c[4][4][1];
        buf0[0] += _cg_bIn_2_2_2_reg * c[4][4][2];
      }
      {
        // New offset [0, 0, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor12 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor9 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor10 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 6 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[3] += 0;
        buf0[3] += _cg_bIn_2_2_2_reg * c[0][0][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[0][0][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[0][0][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[0][0][3];
      }
      {
        // New offset [1, 0, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[1][0][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[1][0][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[1][0][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[1][0][3];
      }
      {
        // New offset [2, 0, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[2][0][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[2][0][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[2][0][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[2][0][3];
      }
      {
        // New offset [3, 0, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor14 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor11 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[3][0][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[3][0][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[3][0][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[3][0][3];
      }
      {
        // New offset [4, 0, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[4][0][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[4][0][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[4][0][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[4][0][3];
      }
      {
        // New offset [0, 1, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor12 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor9 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor10 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[0][1][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[0][1][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[0][1][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[0][1][3];
      }
      {
        // New offset [1, 1, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[1][1][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[1][1][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[1][1][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[1][1][3];
      }
      {
        // New offset [2, 1, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[2][1][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[2][1][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[2][1][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[2][1][3];
      }
      {
        // New offset [3, 1, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor14 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor11 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[3][1][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[3][1][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[3][1][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[3][1][3];
      }
      {
        // New offset [4, 1, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[4][1][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[4][1][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[4][1][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[4][1][3];
      }
      {
        // New offset [0, 2, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_1_2_2_vecbuf = bIn.dat[neighbor12 * bIn.step + 64 + hipThreadIdx_x];
          _cg_bIn0_2_2_vecbuf = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[0][2][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[0][2][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[0][2][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[0][2][3];
      }
      {
        // New offset [1, 2, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[1][2][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[1][2][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[1][2][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[1][2][3];
      }
      {
        // New offset [2, 2, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[2][2][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[2][2][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[2][2][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[2][2][3];
      }
      {
        // New offset [3, 2, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn0_2_2_vecbuf = bIn.dat[neighbor14 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[3][2][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[3][2][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[3][2][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[3][2][3];
      }
      {
        // New offset [4, 2, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[4][2][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[4][2][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[4][2][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[4][2][3];
      }
      {
        // New offset [0, 3, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor15 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor12 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor16 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[0][3][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[0][3][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[0][3][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[0][3][3];
      }
      {
        // New offset [1, 3, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[1][3][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[1][3][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[1][3][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[1][3][3];
      }
      {
        // New offset [2, 3, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[2][3][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[2][3][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[2][3][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[2][3][3];
      }
      {
        // New offset [3, 3, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor17 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor14 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[3][3][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[3][3][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[3][3][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[3][3][3];
      }
      {
        // New offset [4, 3, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[4][3][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[4][3][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[4][3][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[4][3][3];
      }
      {
        // New offset [0, 4, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor15 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor12 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor16 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 2 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[0][4][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[0][4][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[0][4][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[0][4][3];
      }
      {
        // New offset [1, 4, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[1][4][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[1][4][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[1][4][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[1][4][3];
      }
      {
        // New offset [2, 4, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[2][4][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[2][4][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[2][4][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[2][4][3];
      }
      {
        // New offset [3, 4, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor17 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor14 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[3][4][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[3][4][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[3][4][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[3][4][3];
      }
      {
        // New offset [4, 4, 3]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[3] += _cg_bIn_2_2_2_reg * c[4][4][0];
        buf0[2] += _cg_bIn_2_2_2_reg * c[4][4][1];
        buf0[1] += _cg_bIn_2_2_2_reg * c[4][4][2];
        buf0[0] += _cg_bIn_2_2_2_reg * c[4][4][3];
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 4; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, 0, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor12 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor9 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn_1_2_2_vecbuf
              dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = bIn.dat[neighbor13 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = bIn.dat[neighbor10 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 6 -> _cg_bIn0_2_2_vecbuf
              dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 16, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
              _cg_bIn_2_2_2_reg = _cg_vectmp4;
            }
            buf0[4 + rel] += 0;
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[0][0][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[0][0][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[0][0][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[0][0][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[0][0][4];
          }
          {
            // New offset [1, 0, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
              _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_bIn_2_2_2_reg = _cg_vectmp0;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[1][0][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[1][0][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[1][0][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[1][0][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[1][0][4];
          }
          {
            // New offset [2, 0, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[2][0][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[2][0][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[2][0][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[2][0][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[2][0][4];
          }
          {
            // New offset [3, 0, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor14 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor11 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn0_2_2_vecbuf
              dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_bIn_2_2_2_reg = _cg_vectmp2;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[3][0][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[3][0][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[3][0][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[3][0][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[3][0][4];
          }
          {
            // New offset [4, 0, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
              _cg_bIn_2_2_2_reg = _cg_vectmp0;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[4][0][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[4][0][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[4][0][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[4][0][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[4][0][4];
          }
          {
            // New offset [0, 1, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor12 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor9 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn_1_2_2_vecbuf
              dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = bIn.dat[neighbor13 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = bIn.dat[neighbor10 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_bIn0_2_2_vecbuf
              dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
              _cg_bIn_2_2_2_reg = _cg_vectmp4;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[0][1][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[0][1][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[0][1][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[0][1][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[0][1][4];
          }
          {
            // New offset [1, 1, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
              _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_bIn_2_2_2_reg = _cg_vectmp0;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[1][1][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[1][1][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[1][1][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[1][1][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[1][1][4];
          }
          {
            // New offset [2, 1, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[2][1][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[2][1][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[2][1][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[2][1][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[2][1][4];
          }
          {
            // New offset [3, 1, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor14 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor11 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn0_2_2_vecbuf
              dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_bIn_2_2_2_reg = _cg_vectmp2;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[3][1][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[3][1][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[3][1][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[3][1][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[3][1][4];
          }
          {
            // New offset [4, 1, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
              _cg_bIn_2_2_2_reg = _cg_vectmp0;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[4][1][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[4][1][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[4][1][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[4][1][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[4][1][4];
          }
          {
            // New offset [0, 2, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              _cg_bIn_1_2_2_vecbuf = bIn.dat[neighbor12 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_bIn0_2_2_vecbuf = bIn.dat[neighbor13 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
              _cg_bIn_2_2_2_reg = _cg_vectmp0;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[0][2][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[0][2][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[0][2][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[0][2][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[0][2][4];
          }
          {
            // New offset [1, 2, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
              _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_bIn_2_2_2_reg = _cg_vectmp0;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[1][2][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[1][2][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[1][2][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[1][2][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[1][2][4];
          }
          {
            // New offset [2, 2, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[2][2][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[2][2][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[2][2][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[2][2][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[2][2][4];
          }
          {
            // New offset [3, 2, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              _cg_bIn0_2_2_vecbuf = bIn.dat[neighbor14 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_bIn_2_2_2_reg = _cg_vectmp0;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[3][2][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[3][2][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[3][2][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[3][2][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[3][2][4];
          }
          {
            // New offset [4, 2, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
              _cg_bIn_2_2_2_reg = _cg_vectmp0;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[4][2][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[4][2][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[4][2][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[4][2][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[4][2][4];
          }
          {
            // New offset [0, 3, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor15 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor12 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn_1_2_2_vecbuf
              dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = bIn.dat[neighbor16 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = bIn.dat[neighbor13 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_bIn0_2_2_vecbuf
              dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
              _cg_bIn_2_2_2_reg = _cg_vectmp4;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[0][3][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[0][3][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[0][3][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[0][3][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[0][3][4];
          }
          {
            // New offset [1, 3, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
              _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_bIn_2_2_2_reg = _cg_vectmp0;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[1][3][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[1][3][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[1][3][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[1][3][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[1][3][4];
          }
          {
            // New offset [2, 3, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[2][3][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[2][3][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[2][3][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[2][3][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[2][3][4];
          }
          {
            // New offset [3, 3, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor17 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor14 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn0_2_2_vecbuf
              dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_bIn_2_2_2_reg = _cg_vectmp2;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[3][3][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[3][3][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[3][3][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[3][3][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[3][3][4];
          }
          {
            // New offset [4, 3, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
              _cg_bIn_2_2_2_reg = _cg_vectmp0;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[4][3][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[4][3][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[4][3][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[4][3][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[4][3][4];
          }
          {
            // New offset [0, 4, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor15 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor12 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn_1_2_2_vecbuf
              dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              _cg_vectmp2 = bIn.dat[neighbor16 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp3;
              _cg_vectmp3 = bIn.dat[neighbor13 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp3 ,_cg_vectmp2, 2 -> _cg_bIn0_2_2_vecbuf
              dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 48, 64, hipThreadIdx_x);
              bElem _cg_vectmp4;
              // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
              dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
              _cg_bIn_2_2_2_reg = _cg_vectmp4;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[0][4][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[0][4][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[0][4][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[0][4][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[0][4][4];
          }
          {
            // New offset [1, 4, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
              _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_bIn_2_2_2_reg = _cg_vectmp0;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[1][4][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[1][4][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[1][4][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[1][4][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[1][4][4];
          }
          {
            // New offset [2, 4, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[2][4][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[2][4][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[2][4][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[2][4][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[2][4][4];
          }
          {
            // New offset [3, 4, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor17 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor14 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn0_2_2_vecbuf
              dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_bIn_2_2_2_reg = _cg_vectmp2;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[3][4][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[3][4][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[3][4][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[3][4][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[3][4][4];
          }
          {
            // New offset [4, 4, 4]
            bElem _cg_bIn_2_2_2_reg;
            {
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
              _cg_bIn_2_2_2_reg = _cg_vectmp0;
            }
            buf0[4 + rel] += _cg_bIn_2_2_2_reg * c[4][4][0];
            buf0[3 + rel] += _cg_bIn_2_2_2_reg * c[4][4][1];
            buf0[2 + rel] += _cg_bIn_2_2_2_reg * c[4][4][2];
            buf0[1 + rel] += _cg_bIn_2_2_2_reg * c[4][4][3];
            buf0[0 + rel] += _cg_bIn_2_2_2_reg * c[4][4][4];
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [0, 0, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor12 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor9 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor10 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 6 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[0][0][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[0][0][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[0][0][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[0][0][4];
      }
      {
        // New offset [1, 0, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[1][0][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[1][0][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[1][0][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[1][0][4];
      }
      {
        // New offset [2, 0, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[2][0][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[2][0][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[2][0][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[2][0][4];
      }
      {
        // New offset [3, 0, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor14 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor11 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[3][0][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[3][0][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[3][0][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[3][0][4];
      }
      {
        // New offset [4, 0, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[4][0][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[4][0][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[4][0][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[4][0][4];
      }
      {
        // New offset [0, 1, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor12 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor9 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor10 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[0][1][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[0][1][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[0][1][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[0][1][4];
      }
      {
        // New offset [1, 1, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[1][1][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[1][1][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[1][1][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[1][1][4];
      }
      {
        // New offset [2, 1, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[2][1][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[2][1][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[2][1][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[2][1][4];
      }
      {
        // New offset [3, 1, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor14 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor11 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[3][1][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[3][1][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[3][1][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[3][1][4];
      }
      {
        // New offset [4, 1, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[4][1][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[4][1][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[4][1][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[4][1][4];
      }
      {
        // New offset [0, 2, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_1_2_2_vecbuf = bIn.dat[neighbor12 * bIn.step + 384 + hipThreadIdx_x];
          _cg_bIn0_2_2_vecbuf = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[0][2][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[0][2][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[0][2][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[0][2][4];
      }
      {
        // New offset [1, 2, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[1][2][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[1][2][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[1][2][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[1][2][4];
      }
      {
        // New offset [2, 2, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[2][2][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[2][2][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[2][2][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[2][2][4];
      }
      {
        // New offset [3, 2, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn0_2_2_vecbuf = bIn.dat[neighbor14 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[3][2][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[3][2][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[3][2][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[3][2][4];
      }
      {
        // New offset [4, 2, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[4][2][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[4][2][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[4][2][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[4][2][4];
      }
      {
        // New offset [0, 3, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor15 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor12 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor16 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[0][3][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[0][3][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[0][3][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[0][3][4];
      }
      {
        // New offset [1, 3, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[1][3][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[1][3][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[1][3][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[1][3][4];
      }
      {
        // New offset [2, 3, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[2][3][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[2][3][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[2][3][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[2][3][4];
      }
      {
        // New offset [3, 3, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor17 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor14 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[3][3][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[3][3][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[3][3][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[3][3][4];
      }
      {
        // New offset [4, 3, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[4][3][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[4][3][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[4][3][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[4][3][4];
      }
      {
        // New offset [0, 4, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor15 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor12 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor16 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 2 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[0][4][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[0][4][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[0][4][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[0][4][4];
      }
      {
        // New offset [1, 4, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[1][4][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[1][4][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[1][4][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[1][4][4];
      }
      {
        // New offset [2, 4, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[2][4][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[2][4][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[2][4][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[2][4][4];
      }
      {
        // New offset [3, 4, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor17 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor14 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[3][4][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[3][4][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[3][4][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[3][4][4];
      }
      {
        // New offset [4, 4, 8]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[4][4][1];
        buf0[6] += _cg_bIn_2_2_2_reg * c[4][4][2];
        buf0[5] += _cg_bIn_2_2_2_reg * c[4][4][3];
        buf0[4] += _cg_bIn_2_2_2_reg * c[4][4][4];
      }
      {
        // New offset [0, 0, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor12 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor9 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor10 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 6 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[0][0][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[0][0][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[0][0][4];
      }
      {
        // New offset [1, 0, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[1][0][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[1][0][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[1][0][4];
      }
      {
        // New offset [2, 0, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[2][0][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[2][0][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[2][0][4];
      }
      {
        // New offset [3, 0, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor14 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor11 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[3][0][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[3][0][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[3][0][4];
      }
      {
        // New offset [4, 0, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[4][0][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[4][0][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[4][0][4];
      }
      {
        // New offset [0, 1, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor12 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor9 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor10 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[0][1][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[0][1][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[0][1][4];
      }
      {
        // New offset [1, 1, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[1][1][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[1][1][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[1][1][4];
      }
      {
        // New offset [2, 1, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[2][1][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[2][1][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[2][1][4];
      }
      {
        // New offset [3, 1, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor14 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor11 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[3][1][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[3][1][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[3][1][4];
      }
      {
        // New offset [4, 1, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[4][1][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[4][1][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[4][1][4];
      }
      {
        // New offset [0, 2, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_1_2_2_vecbuf = bIn.dat[neighbor12 * bIn.step + 448 + hipThreadIdx_x];
          _cg_bIn0_2_2_vecbuf = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[0][2][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[0][2][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[0][2][4];
      }
      {
        // New offset [1, 2, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[1][2][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[1][2][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[1][2][4];
      }
      {
        // New offset [2, 2, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[2][2][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[2][2][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[2][2][4];
      }
      {
        // New offset [3, 2, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn0_2_2_vecbuf = bIn.dat[neighbor14 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[3][2][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[3][2][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[3][2][4];
      }
      {
        // New offset [4, 2, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[4][2][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[4][2][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[4][2][4];
      }
      {
        // New offset [0, 3, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor15 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor12 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor16 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[0][3][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[0][3][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[0][3][4];
      }
      {
        // New offset [1, 3, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[1][3][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[1][3][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[1][3][4];
      }
      {
        // New offset [2, 3, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[2][3][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[2][3][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[2][3][4];
      }
      {
        // New offset [3, 3, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor17 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor14 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[3][3][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[3][3][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[3][3][4];
      }
      {
        // New offset [4, 3, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[4][3][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[4][3][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[4][3][4];
      }
      {
        // New offset [0, 4, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor15 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor12 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor16 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 2 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[0][4][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[0][4][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[0][4][4];
      }
      {
        // New offset [1, 4, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[1][4][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[1][4][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[1][4][4];
      }
      {
        // New offset [2, 4, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[2][4][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[2][4][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[2][4][4];
      }
      {
        // New offset [3, 4, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor17 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor14 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[3][4][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[3][4][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[3][4][4];
      }
      {
        // New offset [4, 4, 9]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[4][4][2];
        buf0[6] += _cg_bIn_2_2_2_reg * c[4][4][3];
        buf0[5] += _cg_bIn_2_2_2_reg * c[4][4][4];
      }
      {
        // New offset [0, 0, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor21 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor18 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor22 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor19 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 6 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[0][0][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[0][0][4];
      }
      {
        // New offset [1, 0, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[1][0][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[1][0][4];
      }
      {
        // New offset [2, 0, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[2][0][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[2][0][4];
      }
      {
        // New offset [3, 0, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor23 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor20 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[3][0][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[3][0][4];
      }
      {
        // New offset [4, 0, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[4][0][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[4][0][4];
      }
      {
        // New offset [0, 1, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor21 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor18 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor22 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor19 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[0][1][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[0][1][4];
      }
      {
        // New offset [1, 1, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[1][1][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[1][1][4];
      }
      {
        // New offset [2, 1, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[2][1][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[2][1][4];
      }
      {
        // New offset [3, 1, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor23 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor20 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[3][1][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[3][1][4];
      }
      {
        // New offset [4, 1, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[4][1][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[4][1][4];
      }
      {
        // New offset [0, 2, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_1_2_2_vecbuf = bIn.dat[neighbor21 * bIn.step + hipThreadIdx_x];
          _cg_bIn0_2_2_vecbuf = bIn.dat[neighbor22 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[0][2][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[0][2][4];
      }
      {
        // New offset [1, 2, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[1][2][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[1][2][4];
      }
      {
        // New offset [2, 2, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[2][2][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[2][2][4];
      }
      {
        // New offset [3, 2, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn0_2_2_vecbuf = bIn.dat[neighbor23 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[3][2][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[3][2][4];
      }
      {
        // New offset [4, 2, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[4][2][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[4][2][4];
      }
      {
        // New offset [0, 3, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor24 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor21 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor25 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor22 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[0][3][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[0][3][4];
      }
      {
        // New offset [1, 3, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[1][3][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[1][3][4];
      }
      {
        // New offset [2, 3, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[2][3][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[2][3][4];
      }
      {
        // New offset [3, 3, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor26 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor23 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[3][3][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[3][3][4];
      }
      {
        // New offset [4, 3, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[4][3][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[4][3][4];
      }
      {
        // New offset [0, 4, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor24 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor21 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor25 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor22 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 2 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[0][4][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[0][4][4];
      }
      {
        // New offset [1, 4, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[1][4][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[1][4][4];
      }
      {
        // New offset [2, 4, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[2][4][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[2][4][4];
      }
      {
        // New offset [3, 4, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor26 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor23 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[3][4][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[3][4][4];
      }
      {
        // New offset [4, 4, 10]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[4][4][3];
        buf0[6] += _cg_bIn_2_2_2_reg * c[4][4][4];
      }
      {
        // New offset [0, 0, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor21 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor18 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor22 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor19 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 6 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[0][0][4];
      }
      {
        // New offset [1, 0, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[1][0][4];
      }
      {
        // New offset [2, 0, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[2][0][4];
      }
      {
        // New offset [3, 0, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor23 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor20 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[3][0][4];
      }
      {
        // New offset [4, 0, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[4][0][4];
      }
      {
        // New offset [0, 1, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor21 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor18 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor22 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor19 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 7 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[0][1][4];
      }
      {
        // New offset [1, 1, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[1][1][4];
      }
      {
        // New offset [2, 1, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[2][1][4];
      }
      {
        // New offset [3, 1, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor23 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor20 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[3][1][4];
      }
      {
        // New offset [4, 1, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[4][1][4];
      }
      {
        // New offset [0, 2, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_1_2_2_vecbuf = bIn.dat[neighbor21 * bIn.step + 64 + hipThreadIdx_x];
          _cg_bIn0_2_2_vecbuf = bIn.dat[neighbor22 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[0][2][4];
      }
      {
        // New offset [1, 2, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[1][2][4];
      }
      {
        // New offset [2, 2, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[2][2][4];
      }
      {
        // New offset [3, 2, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn0_2_2_vecbuf = bIn.dat[neighbor23 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[3][2][4];
      }
      {
        // New offset [4, 2, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[4][2][4];
      }
      {
        // New offset [0, 3, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor24 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor21 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor25 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor22 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 1 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[0][3][4];
      }
      {
        // New offset [1, 3, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[1][3][4];
      }
      {
        // New offset [2, 3, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[2][3][4];
      }
      {
        // New offset [3, 3, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor26 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor23 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[3][3][4];
      }
      {
        // New offset [4, 3, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[4][3][4];
      }
      {
        // New offset [0, 4, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor24 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor21 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn_1_2_2_vecbuf
          dev_shl(_cg_bIn_1_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          _cg_vectmp2 = bIn.dat[neighbor25 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp3;
          _cg_vectmp3 = bIn.dat[neighbor22 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp3 ,_cg_vectmp2, 2 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp3, _cg_vectmp2, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp4;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 6 -> _cg_vectmp4
          dev_shl(_cg_vectmp4, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp4;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[0][4][4];
      }
      {
        // New offset [1, 4, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_vecbuf = _cg_bIn_1_2_2_vecbuf;
          _cg_bIn_1_2_2_vecbuf = _cg_bIn0_2_2_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_2_2_2_vecbuf ,_cg_bIn_1_2_2_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_2_2_2_vecbuf, _cg_bIn_1_2_2_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[1][4][4];
      }
      {
        // New offset [2, 4, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          _cg_bIn_2_2_2_reg = _cg_bIn_1_2_2_vecbuf;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[2][4][4];
      }
      {
        // New offset [3, 4, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor26 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor23 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn0_2_2_vecbuf
          dev_shl(_cg_bIn0_2_2_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp2;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[3][4][4];
      }
      {
        // New offset [4, 4, 11]
        bElem _cg_bIn_2_2_2_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_1_2_2_vecbuf ,_cg_bIn0_2_2_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_1_2_2_vecbuf, _cg_bIn0_2_2_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn_2_2_2_reg = _cg_vectmp0;
        }
        buf0[7] += _cg_bIn_2_2_2_reg * c[4][4][4];
      }
    }
    bElem *bOut_ref = &bOut.dat[neighbor13 * bOut.step];
    for (long sti = 0; sti < 8; ++sti)
    {
      bOut_ref[sti * 64 + hipThreadIdx_x] = buf0[sti];
    }
  }
}
# 108 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/f3d/intermediate_gen/f3d.cu" 2

}
