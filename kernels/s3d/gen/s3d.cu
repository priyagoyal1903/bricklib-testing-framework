# 1 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/s3d/intermediate_gen/s3d.cu"
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
# 1 "VSBrick-laplacian2.py-HIP-8x8x8-8x8" 1
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
      bElem _cg_bIn000_vecbuf;
      bElem _cg_bIn_100_vecbuf;
      {
        // New offset [0, 0, -2]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_vecbuf = bIn.dat[neighbor4 * bIn.step + 384 + hipThreadIdx_x];
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, -1]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_vecbuf = bIn.dat[neighbor4 * bIn.step + 448 + hipThreadIdx_x];
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[1] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -2, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -1, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [-2, 0, 0]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = bIn.dat[neighbor12 * bIn.step + hipThreadIdx_x];
          _cg_bIn000_vecbuf = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [-1, 0, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 0]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[0] * _cg_bIn000_reg;
        buf0[1] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[2] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [1, 0, 0]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = _cg_bIn000_vecbuf;
          _cg_bIn000_vecbuf = bIn.dat[neighbor14 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [2, 0, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 1, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 2, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -2, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -1, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [-2, 0, 1]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = bIn.dat[neighbor12 * bIn.step + 64 + hipThreadIdx_x];
          _cg_bIn000_vecbuf = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[1] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [-1, 0, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[1] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 1]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[0] * _cg_bIn000_reg;
        buf0[0] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[2] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[3] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [1, 0, 1]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = _cg_bIn000_vecbuf;
          _cg_bIn000_vecbuf = bIn.dat[neighbor14 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[1] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [2, 0, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[1] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 1, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 2, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 4; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, -2, 2]
            bElem _cg_bIn000_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn000_vecbuf
              dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
              _cg_bIn000_reg = _cg_bIn000_vecbuf;
            }
            buf0[2 + rel] += dev_coeff[2] * _cg_bIn000_reg;
          }
          {
            // New offset [0, -1, 2]
            bElem _cg_bIn000_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn000_vecbuf
              dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              _cg_bIn000_reg = _cg_bIn000_vecbuf;
            }
            buf0[2 + rel] += dev_coeff[1] * _cg_bIn000_reg;
          }
          {
            // New offset [-2, 0, 2]
            bElem _cg_bIn000_reg;
            {
              _cg_bIn_100_vecbuf = bIn.dat[neighbor12 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_bIn000_vecbuf = bIn.dat[neighbor13 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 6 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 2, 8, hipThreadIdx_x & 7);
              _cg_bIn000_reg = _cg_vectmp0;
            }
            buf0[2 + rel] += dev_coeff[2] * _cg_bIn000_reg;
          }
          {
            // New offset [-1, 0, 2]
            bElem _cg_bIn000_reg;
            {
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_bIn000_reg = _cg_vectmp0;
            }
            buf0[2 + rel] += dev_coeff[1] * _cg_bIn000_reg;
          }
          {
            // New offset [0, 0, 2]
            bElem _cg_bIn000_reg;
            {
              _cg_bIn000_reg = _cg_bIn000_vecbuf;
            }
            buf0[2 + rel] += dev_coeff[0] * _cg_bIn000_reg;
            buf0[1 + rel] += dev_coeff[1] * _cg_bIn000_reg;
            buf0[3 + rel] += dev_coeff[1] * _cg_bIn000_reg;
            buf0[0 + rel] += dev_coeff[2] * _cg_bIn000_reg;
            buf0[4 + rel] += dev_coeff[2] * _cg_bIn000_reg;
          }
          {
            // New offset [1, 0, 2]
            bElem _cg_bIn000_reg;
            {
              _cg_bIn_100_vecbuf = _cg_bIn000_vecbuf;
              _cg_bIn000_vecbuf = bIn.dat[neighbor14 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_bIn000_reg = _cg_vectmp0;
            }
            buf0[2 + rel] += dev_coeff[1] * _cg_bIn000_reg;
          }
          {
            // New offset [2, 0, 2]
            bElem _cg_bIn000_reg;
            {
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 2 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 6, 8, hipThreadIdx_x & 7);
              _cg_bIn000_reg = _cg_vectmp0;
            }
            buf0[2 + rel] += dev_coeff[2] * _cg_bIn000_reg;
          }
          {
            // New offset [0, 1, 2]
            bElem _cg_bIn000_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn000_vecbuf
              dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              _cg_bIn000_reg = _cg_bIn000_vecbuf;
            }
            buf0[2 + rel] += dev_coeff[1] * _cg_bIn000_reg;
          }
          {
            // New offset [0, 2, 2]
            bElem _cg_bIn000_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 128 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn000_vecbuf
              dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
              _cg_bIn000_reg = _cg_bIn000_vecbuf;
            }
            buf0[2 + rel] += dev_coeff[2] * _cg_bIn000_reg;
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [0, -2, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -1, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [-2, 0, 6]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = bIn.dat[neighbor12 * bIn.step + 384 + hipThreadIdx_x];
          _cg_bIn000_vecbuf = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[6] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [-1, 0, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[6] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 6]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[0] * _cg_bIn000_reg;
        buf0[5] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[7] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[4] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [1, 0, 6]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = _cg_bIn000_vecbuf;
          _cg_bIn000_vecbuf = bIn.dat[neighbor14 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[6] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [2, 0, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[6] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 1, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 2, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -2, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -1, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [-2, 0, 7]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = bIn.dat[neighbor12 * bIn.step + 448 + hipThreadIdx_x];
          _cg_bIn000_vecbuf = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[7] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [-1, 0, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[7] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 7]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[0] * _cg_bIn000_reg;
        buf0[6] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[5] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [1, 0, 7]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = _cg_bIn000_vecbuf;
          _cg_bIn000_vecbuf = bIn.dat[neighbor14 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[7] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [2, 0, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[7] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 1, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 2, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 8]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_vecbuf = bIn.dat[neighbor22 * bIn.step + hipThreadIdx_x];
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[6] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 9]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_vecbuf = bIn.dat[neighbor22 * bIn.step + 64 + hipThreadIdx_x];
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[2] * _cg_bIn000_reg;
      }
    }
    bElem *bOut_ref = &bOut.dat[neighbor13 * bOut.step];
    for (long sti = 0; sti < 8; ++sti)
    {
      bOut_ref[sti * 64 + hipThreadIdx_x] = buf0[sti];
    }
  }
}
# 40 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/s3d/intermediate_gen/s3d.cu" 2

}
#define bIn(a, b, c) arr_in[c][b][a]
#define bOut(a, b, c) arr_out[c][b][a]
__global__ void s3d_codegen2(bElem (*arr_in)[STRIDE1][STRIDE0], bElem (*arr_out)[STRIDE1][STRIDE0], bElem *dev_coeff) {
    long k = OFF2 + (blockIdx.z * TILE2);
    long j = OFF1 + (blockIdx.y * TILE1);
    long i = OFF0 + (blockIdx.x * VECSIZE);
# 1 "VSTile-laplacian2.py-HIP-8x8x64" 1
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
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[0 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -2);
          }
          _cg_rel1 += 1;
        }
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[0 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf0[8 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[0] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2, k);
      }
      {
        buf0[0] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1, k);
        buf0[1] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1, k);
      }
      {
        buf0[0] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j, k);
        buf0[0] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j, k);
        buf0[0] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j, k);
        buf0[0] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j, k);
        buf0[0] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j, k);
        buf0[8] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k);
        buf0[1] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k);
        buf0[16] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k);
        buf0[2] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k);
      }
      {
        buf0[1] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[1] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1, k);
        buf0[1] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1, k);
        buf0[1] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1, k);
        buf0[1] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1, k);
        buf0[0] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[9] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[2] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[17] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[3] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 4; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[2 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf0[2 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k);
            buf0[2 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k);
            buf0[2 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k);
            buf0[2 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k);
            buf0[1 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf0[10 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf0[3 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf0[0 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf0[18 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
            buf0[4 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[6] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[6] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6, k);
        buf0[6] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6, k);
        buf0[6] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6, k);
        buf0[6] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6, k);
        buf0[5] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[14] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[7] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[4] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[22] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k);
      }
      {
        buf0[7] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7, k);
        buf0[7] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7, k);
        buf0[7] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7, k);
        buf0[7] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7, k);
        buf0[7] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7, k);
        buf0[6] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k);
        buf0[15] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k);
        buf0[5] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k);
        buf0[23] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k);
      }
      {
        buf0[7] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8, k);
        buf0[6] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8, k);
      }
      {
        buf0[7] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9, k);
      }
      {
        buf0[8] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2, k + 1);
      }
      {
        buf0[8] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1, k + 1);
        buf0[9] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1, k + 1);
      }
      {
        buf0[8] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[8] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j, k + 1);
        buf0[8] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j, k + 1);
        buf0[8] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j, k + 1);
        buf0[8] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j, k + 1);
        buf0[0] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[16] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[9] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[24] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[10] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 1);
      }
      {
        buf0[9] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[9] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1, k + 1);
        buf0[9] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf0[9] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1, k + 1);
        buf0[9] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1, k + 1);
        buf0[1] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[8] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[17] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[10] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[25] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[11] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 4; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[10 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf0[10 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 1);
            buf0[10 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 1);
            buf0[10 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 1);
            buf0[10 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 1);
            buf0[2 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf0[9 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf0[18 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf0[11 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf0[8 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf0[26 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
            buf0[12 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[14] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[14] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6, k + 1);
        buf0[14] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6, k + 1);
        buf0[14] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6, k + 1);
        buf0[14] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6, k + 1);
        buf0[6] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[13] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[22] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[15] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[12] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[30] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
      }
      {
        buf0[15] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[15] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf0[15] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf0[15] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7, k + 1);
        buf0[15] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7, k + 1);
        buf0[7] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[14] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[23] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[13] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[31] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
      }
      {
        buf0[15] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8, k + 1);
        buf0[14] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8, k + 1);
      }
      {
        buf0[15] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9, k + 1);
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 4; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            buf0[16 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2, k + _cg_idx2 + 2);
          }
          {
            buf0[16 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2);
            buf0[17 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 2);
          }
          {
            buf0[16 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf0[16 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 2);
            buf0[16 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 2);
            buf0[16 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j, k + _cg_idx2 + 2);
            buf0[16 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j, k + _cg_idx2 + 2);
            buf0[8 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf0[24 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf0[17 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf0[0 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf0[32 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
            buf0[18 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 2);
          }
          {
            buf0[17 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf0[17 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 2);
            buf0[17 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 2);
            buf0[17 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 2);
            buf0[17 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 2);
            buf0[9 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf0[16 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf0[25 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf0[18 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf0[1 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf0[33 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
            buf0[19 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 2);
          }
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 4; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf0[18 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[18 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[18 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[18 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[18 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[10 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[17 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[26 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[19 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[2 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[16 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[34 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
                buf0[20 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + _cg_idx2 + 2);
              }
              _cg_rel1 += 1;
            }
          }
          {
            buf0[22 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf0[22 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 2);
            buf0[22 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 2);
            buf0[22 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 2);
            buf0[22 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 2);
            buf0[14 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf0[21 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf0[30 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf0[23 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf0[6 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf0[20 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
            buf0[38 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 2);
          }
          {
            buf0[23 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf0[23 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 2);
            buf0[23 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 2);
            buf0[23 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7, k + _cg_idx2 + 2);
            buf0[23 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7, k + _cg_idx2 + 2);
            buf0[15 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf0[22 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf0[31 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf0[7 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf0[21 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
            buf0[39 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 2);
          }
          {
            buf0[23 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2);
            buf0[22 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 2);
          }
          {
            buf0[23 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9, k + _cg_idx2 + 2);
          }
          _cg_rel2 += 8;
        }
      }
      {
        buf0[48] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2, k + 6);
      }
      {
        buf0[48] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1, k + 6);
        buf0[49] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1, k + 6);
      }
      {
        buf0[48] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[48] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j, k + 6);
        buf0[48] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j, k + 6);
        buf0[48] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j, k + 6);
        buf0[48] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j, k + 6);
        buf0[40] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[56] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[49] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[32] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[50] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 6);
      }
      {
        buf0[49] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[49] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1, k + 6);
        buf0[49] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf0[49] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1, k + 6);
        buf0[49] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1, k + 6);
        buf0[41] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[48] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[57] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[50] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[33] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[51] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 4; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[50 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf0[50 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 6);
            buf0[50 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 6);
            buf0[50 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 6);
            buf0[50 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 6);
            buf0[42 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf0[49 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf0[58 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf0[51 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf0[34 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf0[48 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
            buf0[52 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 6);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[54] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[54] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6, k + 6);
        buf0[54] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6, k + 6);
        buf0[54] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6, k + 6);
        buf0[54] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6, k + 6);
        buf0[46] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[53] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[62] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[55] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[38] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[52] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
      }
      {
        buf0[55] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[55] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7, k + 6);
        buf0[55] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7, k + 6);
        buf0[55] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7, k + 6);
        buf0[55] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7, k + 6);
        buf0[47] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[54] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[63] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[39] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[53] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
      }
      {
        buf0[55] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8, k + 6);
        buf0[54] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8, k + 6);
      }
      {
        buf0[55] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9, k + 6);
      }
      {
        buf0[56] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2, k + 7);
      }
      {
        buf0[56] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1, k + 7);
        buf0[57] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1, k + 7);
      }
      {
        buf0[56] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j, k + 7);
        buf0[56] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j, k + 7);
        buf0[56] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j, k + 7);
        buf0[56] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j, k + 7);
        buf0[56] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j, k + 7);
        buf0[48] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 7);
        buf0[57] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 7);
        buf0[40] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 7);
        buf0[58] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 7);
      }
      {
        buf0[57] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[57] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1, k + 7);
        buf0[57] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1, k + 7);
        buf0[57] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1, k + 7);
        buf0[57] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1, k + 7);
        buf0[49] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[56] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[58] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[41] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[59] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 4; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[58 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf0[58 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2, k + 7);
            buf0[58 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2, k + 7);
            buf0[58 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2, k + 7);
            buf0[58 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2, k + 7);
            buf0[50 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf0[57 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf0[59 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf0[42 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf0[56 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
            buf0[60 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2, k + 7);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[62] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[62] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6, k + 7);
        buf0[62] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6, k + 7);
        buf0[62] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6, k + 7);
        buf0[62] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6, k + 7);
        buf0[54] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[61] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[63] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[46] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[60] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
      }
      {
        buf0[63] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[63] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf0[63] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf0[63] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7, k + 7);
        buf0[63] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7, k + 7);
        buf0[55] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[62] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[47] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[61] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
      }
      {
        buf0[63] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8, k + 7);
        buf0[62] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8, k + 7);
      }
      {
        buf0[63] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9, k + 7);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[56 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf0[48 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
          }
          _cg_rel1 += 1;
        }
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[56 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 9);
          }
          _cg_rel1 += 1;
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
            bOut(i + _cg_idx0, j + _cg_idx1, k + _cg_idx2) = buf0[rel];
          }
        }
      }
    }
  }
}
# 48 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/s3d/intermediate_gen/s3d.cu" 2

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
# 1 "VSBrick-laplacian3.py-HIP-8x8x8-8x8" 1
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
      bElem _cg_bIn000_vecbuf;
      bElem _cg_bIn_100_vecbuf;
      {
        // New offset [0, 0, -3]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_vecbuf = bIn.dat[neighbor4 * bIn.step + 320 + hipThreadIdx_x];
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, -2]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_vecbuf = bIn.dat[neighbor4 * bIn.step + 384 + hipThreadIdx_x];
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[1] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, -1]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_vecbuf = bIn.dat[neighbor4 * bIn.step + 448 + hipThreadIdx_x];
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[1] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[2] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -3, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -2, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -1, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [-3, 0, 0]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = bIn.dat[neighbor12 * bIn.step + hipThreadIdx_x];
          _cg_bIn000_vecbuf = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [-2, 0, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [-1, 0, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 0]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[0] * _cg_bIn000_reg;
        buf0[1] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[2] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[3] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [1, 0, 0]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = _cg_bIn000_vecbuf;
          _cg_bIn000_vecbuf = bIn.dat[neighbor14 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [2, 0, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [3, 0, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 1, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 2, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 3, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -3, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -2, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -1, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [-3, 0, 1]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = bIn.dat[neighbor12 * bIn.step + 64 + hipThreadIdx_x];
          _cg_bIn000_vecbuf = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[1] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [-2, 0, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[1] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [-1, 0, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[1] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 1]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[0] * _cg_bIn000_reg;
        buf0[0] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[2] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[3] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[4] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [1, 0, 1]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = _cg_bIn000_vecbuf;
          _cg_bIn000_vecbuf = bIn.dat[neighbor14 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[1] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [2, 0, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[1] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [3, 0, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[1] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 1, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 2, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 3, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -3, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 128 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 128 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[2] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -2, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 128 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 128 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[2] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -1, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 128 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 128 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[2] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [-3, 0, 2]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = bIn.dat[neighbor12 * bIn.step + 128 + hipThreadIdx_x];
          _cg_bIn000_vecbuf = bIn.dat[neighbor13 * bIn.step + 128 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[2] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [-2, 0, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[2] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [-1, 0, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[2] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 2]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[2] += dev_coeff[0] * _cg_bIn000_reg;
        buf0[1] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[3] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[0] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[4] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[5] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [1, 0, 2]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = _cg_bIn000_vecbuf;
          _cg_bIn000_vecbuf = bIn.dat[neighbor14 * bIn.step + 128 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[2] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [2, 0, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[2] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [3, 0, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[2] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 1, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 128 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 128 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[2] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 2, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 128 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 128 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[2] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 3, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 128 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 128 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[2] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 2; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, -3, 3]
            bElem _cg_bIn000_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 192 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 192 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
              dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
              _cg_bIn000_reg = _cg_bIn000_vecbuf;
            }
            buf0[3 + rel] += dev_coeff[3] * _cg_bIn000_reg;
          }
          {
            // New offset [0, -2, 3]
            bElem _cg_bIn000_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 192 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 192 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn000_vecbuf
              dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
              _cg_bIn000_reg = _cg_bIn000_vecbuf;
            }
            buf0[3 + rel] += dev_coeff[2] * _cg_bIn000_reg;
          }
          {
            // New offset [0, -1, 3]
            bElem _cg_bIn000_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 192 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 192 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn000_vecbuf
              dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              _cg_bIn000_reg = _cg_bIn000_vecbuf;
            }
            buf0[3 + rel] += dev_coeff[1] * _cg_bIn000_reg;
          }
          {
            // New offset [-3, 0, 3]
            bElem _cg_bIn000_reg;
            {
              _cg_bIn_100_vecbuf = bIn.dat[neighbor12 * bIn.step + 192 + (hipThreadIdx_x + rel * 64)];
              _cg_bIn000_vecbuf = bIn.dat[neighbor13 * bIn.step + 192 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
              _cg_bIn000_reg = _cg_vectmp0;
            }
            buf0[3 + rel] += dev_coeff[3] * _cg_bIn000_reg;
          }
          {
            // New offset [-2, 0, 3]
            bElem _cg_bIn000_reg;
            {
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 6 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 2, 8, hipThreadIdx_x & 7);
              _cg_bIn000_reg = _cg_vectmp0;
            }
            buf0[3 + rel] += dev_coeff[2] * _cg_bIn000_reg;
          }
          {
            // New offset [-1, 0, 3]
            bElem _cg_bIn000_reg;
            {
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_bIn000_reg = _cg_vectmp0;
            }
            buf0[3 + rel] += dev_coeff[1] * _cg_bIn000_reg;
          }
          {
            // New offset [0, 0, 3]
            bElem _cg_bIn000_reg;
            {
              _cg_bIn000_reg = _cg_bIn000_vecbuf;
            }
            buf0[3 + rel] += dev_coeff[0] * _cg_bIn000_reg;
            buf0[2 + rel] += dev_coeff[1] * _cg_bIn000_reg;
            buf0[4 + rel] += dev_coeff[1] * _cg_bIn000_reg;
            buf0[1 + rel] += dev_coeff[2] * _cg_bIn000_reg;
            buf0[5 + rel] += dev_coeff[2] * _cg_bIn000_reg;
            buf0[0 + rel] += dev_coeff[3] * _cg_bIn000_reg;
            buf0[6 + rel] += dev_coeff[3] * _cg_bIn000_reg;
          }
          {
            // New offset [1, 0, 3]
            bElem _cg_bIn000_reg;
            {
              _cg_bIn_100_vecbuf = _cg_bIn000_vecbuf;
              _cg_bIn000_vecbuf = bIn.dat[neighbor14 * bIn.step + 192 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 1 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_bIn000_reg = _cg_vectmp0;
            }
            buf0[3 + rel] += dev_coeff[1] * _cg_bIn000_reg;
          }
          {
            // New offset [2, 0, 3]
            bElem _cg_bIn000_reg;
            {
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 2 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 6, 8, hipThreadIdx_x & 7);
              _cg_bIn000_reg = _cg_vectmp0;
            }
            buf0[3 + rel] += dev_coeff[2] * _cg_bIn000_reg;
          }
          {
            // New offset [3, 0, 3]
            bElem _cg_bIn000_reg;
            {
              bElem _cg_vectmp0;
              // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
              _cg_bIn000_reg = _cg_vectmp0;
            }
            buf0[3 + rel] += dev_coeff[3] * _cg_bIn000_reg;
          }
          {
            // New offset [0, 1, 3]
            bElem _cg_bIn000_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 192 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 192 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn000_vecbuf
              dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              _cg_bIn000_reg = _cg_bIn000_vecbuf;
            }
            buf0[3 + rel] += dev_coeff[1] * _cg_bIn000_reg;
          }
          {
            // New offset [0, 2, 3]
            bElem _cg_bIn000_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 192 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 192 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn000_vecbuf
              dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
              _cg_bIn000_reg = _cg_bIn000_vecbuf;
            }
            buf0[3 + rel] += dev_coeff[2] * _cg_bIn000_reg;
          }
          {
            // New offset [0, 3, 3]
            bElem _cg_bIn000_reg;
            {
              bElem _cg_vectmp0;
              _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 192 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 192 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
              dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
              _cg_bIn000_reg = _cg_bIn000_vecbuf;
            }
            buf0[3 + rel] += dev_coeff[3] * _cg_bIn000_reg;
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [0, -3, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 320 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 320 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[5] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -2, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 320 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 320 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[5] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -1, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 320 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 320 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[5] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [-3, 0, 5]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = bIn.dat[neighbor12 * bIn.step + 320 + hipThreadIdx_x];
          _cg_bIn000_vecbuf = bIn.dat[neighbor13 * bIn.step + 320 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[5] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [-2, 0, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[5] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [-1, 0, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[5] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 5]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[5] += dev_coeff[0] * _cg_bIn000_reg;
        buf0[4] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[6] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[3] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[7] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[2] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [1, 0, 5]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = _cg_bIn000_vecbuf;
          _cg_bIn000_vecbuf = bIn.dat[neighbor14 * bIn.step + 320 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[5] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [2, 0, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[5] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [3, 0, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[5] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 1, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 320 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 320 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[5] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 2, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 320 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 320 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[5] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 3, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 320 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 320 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[5] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -3, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -2, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -1, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [-3, 0, 6]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = bIn.dat[neighbor12 * bIn.step + 384 + hipThreadIdx_x];
          _cg_bIn000_vecbuf = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[6] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [-2, 0, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[6] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [-1, 0, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[6] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 6]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[0] * _cg_bIn000_reg;
        buf0[5] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[7] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[4] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[3] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [1, 0, 6]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = _cg_bIn000_vecbuf;
          _cg_bIn000_vecbuf = bIn.dat[neighbor14 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[6] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [2, 0, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[6] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [3, 0, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[6] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 1, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 2, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 3, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -3, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -2, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -1, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [-3, 0, 7]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = bIn.dat[neighbor12 * bIn.step + 448 + hipThreadIdx_x];
          _cg_bIn000_vecbuf = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[7] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [-2, 0, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[7] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [-1, 0, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[7] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 7]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[0] * _cg_bIn000_reg;
        buf0[6] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[5] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[4] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [1, 0, 7]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = _cg_bIn000_vecbuf;
          _cg_bIn000_vecbuf = bIn.dat[neighbor14 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[7] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [2, 0, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[7] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [3, 0, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[7] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 1, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 2, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 3, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 8]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_vecbuf = bIn.dat[neighbor22 * bIn.step + hipThreadIdx_x];
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[6] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[5] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 9]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_vecbuf = bIn.dat[neighbor22 * bIn.step + 64 + hipThreadIdx_x];
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[6] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 10]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_vecbuf = bIn.dat[neighbor22 * bIn.step + 128 + hipThreadIdx_x];
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[3] * _cg_bIn000_reg;
      }
    }
    bElem *bOut_ref = &bOut.dat[neighbor13 * bOut.step];
    for (long sti = 0; sti < 8; ++sti)
    {
      bOut_ref[sti * 64 + hipThreadIdx_x] = buf0[sti];
    }
  }
}
# 83 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/s3d/intermediate_gen/s3d.cu" 2

}
#define bIn(a, b, c) arr_in[c][b][a]
#define bOut(a, b, c) arr_out[c][b][a]
__global__ void s3d_codegen3(bElem (*arr_in)[STRIDE1][STRIDE0], bElem (*arr_out)[STRIDE1][STRIDE0], bElem *dev_coeff) {
    long k = OFF2 + (blockIdx.z * TILE2);
    long j = OFF1 + (blockIdx.y * TILE1);
    long i = OFF0 + (blockIdx.x * VECSIZE);
# 1 "VSTile-laplacian3.py-HIP-8x8x64" 1
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
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[0 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -3);
          }
          _cg_rel1 += 1;
        }
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[0 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -2);
            buf0[8 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -2);
          }
          _cg_rel1 += 1;
        }
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[0 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf0[8 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf0[16 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[0] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -3, k);
      }
      {
        buf0[0] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2, k);
        buf0[1] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -2, k);
      }
      {
        buf0[0] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1, k);
        buf0[1] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1, k);
        buf0[2] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -1, k);
      }
      {
        buf0[0] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j, k);
        buf0[0] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j, k);
        buf0[0] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j, k);
        buf0[0] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j, k);
        buf0[0] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j, k);
        buf0[0] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j, k);
        buf0[0] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j, k);
        buf0[8] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k);
        buf0[1] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k);
        buf0[16] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k);
        buf0[2] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k);
        buf0[24] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k);
        buf0[3] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k);
      }
      {
        buf0[1] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[1] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1, k);
        buf0[1] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1, k);
        buf0[1] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1, k);
        buf0[1] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1, k);
        buf0[1] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 1, k);
        buf0[1] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 1, k);
        buf0[0] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[9] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[2] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[17] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[3] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[25] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[4] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k);
      }
      {
        buf0[2] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 2, k);
        buf0[2] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 2, k);
        buf0[2] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 2, k);
        buf0[2] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 2, k);
        buf0[2] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 2, k);
        buf0[2] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 2, k);
        buf0[2] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 2, k);
        buf0[1] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k);
        buf0[10] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k);
        buf0[3] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k);
        buf0[0] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k);
        buf0[18] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k);
        buf0[4] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k);
        buf0[26] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k);
        buf0[5] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 2; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[3 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k);
            buf0[3 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 3, k);
            buf0[3 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 3, k);
            buf0[3 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 3, k);
            buf0[3 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 3, k);
            buf0[3 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + _cg_idx1 + 3, k);
            buf0[3 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + _cg_idx1 + 3, k);
            buf0[2 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k);
            buf0[11 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k);
            buf0[4 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k);
            buf0[1 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k);
            buf0[19 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k);
            buf0[5 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k);
            buf0[0 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k);
            buf0[27 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k);
            buf0[6 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[5] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 5, k);
        buf0[5] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 5, k);
        buf0[5] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 5, k);
        buf0[5] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 5, k);
        buf0[5] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 5, k);
        buf0[5] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 5, k);
        buf0[5] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 5, k);
        buf0[4] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k);
        buf0[13] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k);
        buf0[6] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k);
        buf0[3] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k);
        buf0[21] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k);
        buf0[7] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k);
        buf0[2] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k);
        buf0[29] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k);
      }
      {
        buf0[6] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[6] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6, k);
        buf0[6] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6, k);
        buf0[6] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6, k);
        buf0[6] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6, k);
        buf0[6] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 6, k);
        buf0[6] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 6, k);
        buf0[5] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[14] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[7] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[4] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[22] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[3] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[30] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k);
      }
      {
        buf0[7] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7, k);
        buf0[7] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7, k);
        buf0[7] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7, k);
        buf0[7] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7, k);
        buf0[7] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7, k);
        buf0[7] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 7, k);
        buf0[7] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 7, k);
        buf0[6] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k);
        buf0[15] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k);
        buf0[5] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k);
        buf0[23] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k);
        buf0[4] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k);
        buf0[31] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k);
      }
      {
        buf0[7] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8, k);
        buf0[6] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8, k);
        buf0[5] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 8, k);
      }
      {
        buf0[7] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9, k);
        buf0[6] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 9, k);
      }
      {
        buf0[7] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 10, k);
      }
      {
        buf0[8] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -3, k + 1);
      }
      {
        buf0[8] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2, k + 1);
        buf0[9] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -2, k + 1);
      }
      {
        buf0[8] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1, k + 1);
        buf0[9] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1, k + 1);
        buf0[10] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -1, k + 1);
      }
      {
        buf0[8] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[8] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j, k + 1);
        buf0[8] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j, k + 1);
        buf0[8] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j, k + 1);
        buf0[8] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j, k + 1);
        buf0[8] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j, k + 1);
        buf0[8] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j, k + 1);
        buf0[0] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[16] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[9] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[24] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[10] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[32] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[11] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 1);
      }
      {
        buf0[9] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[9] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1, k + 1);
        buf0[9] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf0[9] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1, k + 1);
        buf0[9] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1, k + 1);
        buf0[9] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 1, k + 1);
        buf0[9] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 1, k + 1);
        buf0[1] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[8] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[17] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[10] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[25] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[11] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[33] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[12] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
      }
      {
        buf0[10] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[10] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 2, k + 1);
        buf0[10] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 2, k + 1);
        buf0[10] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 2, k + 1);
        buf0[10] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 2, k + 1);
        buf0[10] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 2, k + 1);
        buf0[10] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 2, k + 1);
        buf0[2] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[9] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[18] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[11] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[8] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[26] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[12] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[34] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[13] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 2; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[11 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 1);
            buf0[11 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 3, k + 1);
            buf0[11 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 3, k + 1);
            buf0[11 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 3, k + 1);
            buf0[11 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 3, k + 1);
            buf0[11 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + _cg_idx1 + 3, k + 1);
            buf0[11 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + _cg_idx1 + 3, k + 1);
            buf0[3 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 1);
            buf0[10 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 1);
            buf0[19 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 1);
            buf0[12 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 1);
            buf0[9 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 1);
            buf0[27 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 1);
            buf0[13 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 1);
            buf0[8 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 1);
            buf0[35 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 1);
            buf0[14 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[13] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[13] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 5, k + 1);
        buf0[13] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 5, k + 1);
        buf0[13] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 5, k + 1);
        buf0[13] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 5, k + 1);
        buf0[13] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 5, k + 1);
        buf0[13] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 5, k + 1);
        buf0[5] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[12] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[21] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[14] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[11] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[29] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[15] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[10] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[37] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
      }
      {
        buf0[14] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[14] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6, k + 1);
        buf0[14] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6, k + 1);
        buf0[14] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6, k + 1);
        buf0[14] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6, k + 1);
        buf0[14] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 6, k + 1);
        buf0[14] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 6, k + 1);
        buf0[6] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[13] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[22] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[15] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[12] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[30] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[11] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[38] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
      }
      {
        buf0[15] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[15] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf0[15] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf0[15] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7, k + 1);
        buf0[15] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7, k + 1);
        buf0[15] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 7, k + 1);
        buf0[15] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 7, k + 1);
        buf0[7] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[14] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[23] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[13] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[31] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[12] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[39] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
      }
      {
        buf0[15] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8, k + 1);
        buf0[14] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8, k + 1);
        buf0[13] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 8, k + 1);
      }
      {
        buf0[15] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9, k + 1);
        buf0[14] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 9, k + 1);
      }
      {
        buf0[15] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 10, k + 1);
      }
      {
        buf0[16] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -3, k + 2);
      }
      {
        buf0[16] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2, k + 2);
        buf0[17] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -2, k + 2);
      }
      {
        buf0[16] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1, k + 2);
        buf0[17] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1, k + 2);
        buf0[18] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -1, k + 2);
      }
      {
        buf0[16] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j, k + 2);
        buf0[16] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j, k + 2);
        buf0[16] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j, k + 2);
        buf0[16] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j, k + 2);
        buf0[16] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j, k + 2);
        buf0[16] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j, k + 2);
        buf0[16] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j, k + 2);
        buf0[8] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 2);
        buf0[24] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 2);
        buf0[17] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 2);
        buf0[0] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 2);
        buf0[32] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 2);
        buf0[18] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 2);
        buf0[40] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 2);
        buf0[19] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 2);
      }
      {
        buf0[17] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[17] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1, k + 2);
        buf0[17] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1, k + 2);
        buf0[17] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1, k + 2);
        buf0[17] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1, k + 2);
        buf0[17] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 1, k + 2);
        buf0[17] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 1, k + 2);
        buf0[9] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[16] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[25] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[18] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[1] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[33] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[19] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[41] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[20] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
      }
      {
        buf0[18] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[18] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 2, k + 2);
        buf0[18] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 2, k + 2);
        buf0[18] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 2, k + 2);
        buf0[18] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 2, k + 2);
        buf0[18] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 2, k + 2);
        buf0[18] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 2, k + 2);
        buf0[10] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[17] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[26] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[19] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[2] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[16] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[34] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[20] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[42] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[21] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 2; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[19 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 2);
            buf0[19 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 3, k + 2);
            buf0[19 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 3, k + 2);
            buf0[19 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 3, k + 2);
            buf0[19 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 3, k + 2);
            buf0[19 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + _cg_idx1 + 3, k + 2);
            buf0[19 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + _cg_idx1 + 3, k + 2);
            buf0[11 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 2);
            buf0[18 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 2);
            buf0[27 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 2);
            buf0[20 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 2);
            buf0[3 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 2);
            buf0[17 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 2);
            buf0[35 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 2);
            buf0[21 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 2);
            buf0[16 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 2);
            buf0[43 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 2);
            buf0[22 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 2);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[21] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[21] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 5, k + 2);
        buf0[21] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 5, k + 2);
        buf0[21] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 5, k + 2);
        buf0[21] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 5, k + 2);
        buf0[21] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 5, k + 2);
        buf0[21] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 5, k + 2);
        buf0[13] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[20] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[29] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[22] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[5] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[19] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[37] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[23] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[18] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[45] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
      }
      {
        buf0[22] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[22] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6, k + 2);
        buf0[22] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6, k + 2);
        buf0[22] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6, k + 2);
        buf0[22] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6, k + 2);
        buf0[22] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 6, k + 2);
        buf0[22] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 6, k + 2);
        buf0[14] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[21] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[30] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[23] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[6] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[20] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[38] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[19] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[46] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
      }
      {
        buf0[23] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
        buf0[23] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7, k + 2);
        buf0[23] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7, k + 2);
        buf0[23] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7, k + 2);
        buf0[23] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7, k + 2);
        buf0[23] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 7, k + 2);
        buf0[23] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 7, k + 2);
        buf0[15] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
        buf0[22] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
        buf0[31] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
        buf0[7] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
        buf0[21] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
        buf0[39] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
        buf0[20] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
        buf0[47] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
      }
      {
        buf0[23] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8, k + 2);
        buf0[22] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8, k + 2);
        buf0[21] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 8, k + 2);
      }
      {
        buf0[23] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9, k + 2);
        buf0[22] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 9, k + 2);
      }
      {
        buf0[23] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 10, k + 2);
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 2; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            buf0[24 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -3, k + _cg_idx2 + 3);
          }
          {
            buf0[24 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2, k + _cg_idx2 + 3);
            buf0[25 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -2, k + _cg_idx2 + 3);
          }
          {
            buf0[24 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 3);
            buf0[25 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 3);
            buf0[26 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -1, k + _cg_idx2 + 3);
          }
          {
            buf0[24 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 3);
            buf0[24 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j, k + _cg_idx2 + 3);
            buf0[24 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j, k + _cg_idx2 + 3);
            buf0[24 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j, k + _cg_idx2 + 3);
            buf0[24 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j, k + _cg_idx2 + 3);
            buf0[24 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j, k + _cg_idx2 + 3);
            buf0[24 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j, k + _cg_idx2 + 3);
            buf0[16 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 3);
            buf0[32 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 3);
            buf0[25 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 3);
            buf0[8 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 3);
            buf0[40 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 3);
            buf0[26 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 3);
            buf0[0 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 3);
            buf0[48 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 3);
            buf0[27 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + _cg_idx2 + 3);
          }
          {
            buf0[25 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 3);
            buf0[25 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1, k + _cg_idx2 + 3);
            buf0[25 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1, k + _cg_idx2 + 3);
            buf0[25 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1, k + _cg_idx2 + 3);
            buf0[25 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1, k + _cg_idx2 + 3);
            buf0[25 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 1, k + _cg_idx2 + 3);
            buf0[25 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 1, k + _cg_idx2 + 3);
            buf0[17 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 3);
            buf0[24 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 3);
            buf0[33 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 3);
            buf0[26 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 3);
            buf0[9 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 3);
            buf0[41 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 3);
            buf0[27 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 3);
            buf0[1 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 3);
            buf0[49 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 3);
            buf0[28 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + _cg_idx2 + 3);
          }
          {
            buf0[26 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 2, k + _cg_idx2 + 3);
            buf0[26 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 2, k + _cg_idx2 + 3);
            buf0[26 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 2, k + _cg_idx2 + 3);
            buf0[26 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 2, k + _cg_idx2 + 3);
            buf0[26 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 2, k + _cg_idx2 + 3);
            buf0[26 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 2, k + _cg_idx2 + 3);
            buf0[26 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 2, k + _cg_idx2 + 3);
            buf0[18 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + _cg_idx2 + 3);
            buf0[25 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + _cg_idx2 + 3);
            buf0[34 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + _cg_idx2 + 3);
            buf0[27 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + _cg_idx2 + 3);
            buf0[10 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + _cg_idx2 + 3);
            buf0[24 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + _cg_idx2 + 3);
            buf0[42 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + _cg_idx2 + 3);
            buf0[28 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + _cg_idx2 + 3);
            buf0[2 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + _cg_idx2 + 3);
            buf0[50 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + _cg_idx2 + 3);
            buf0[29 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + _cg_idx2 + 3);
          }
          {
            long _cg_rel1 = rel;
            for (long _cg_idx1 = 0; _cg_idx1 < 2; _cg_idx1 += 1)
            {
              long rel = _cg_rel1;
              {
                buf0[27 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + _cg_idx2 + 3);
                buf0[27 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 3, k + _cg_idx2 + 3);
                buf0[27 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 3, k + _cg_idx2 + 3);
                buf0[27 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 3, k + _cg_idx2 + 3);
                buf0[27 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 3, k + _cg_idx2 + 3);
                buf0[27 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + _cg_idx1 + 3, k + _cg_idx2 + 3);
                buf0[27 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + _cg_idx1 + 3, k + _cg_idx2 + 3);
                buf0[19 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + _cg_idx2 + 3);
                buf0[26 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + _cg_idx2 + 3);
                buf0[35 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + _cg_idx2 + 3);
                buf0[28 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + _cg_idx2 + 3);
                buf0[11 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + _cg_idx2 + 3);
                buf0[25 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + _cg_idx2 + 3);
                buf0[43 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + _cg_idx2 + 3);
                buf0[29 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + _cg_idx2 + 3);
                buf0[3 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + _cg_idx2 + 3);
                buf0[24 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + _cg_idx2 + 3);
                buf0[51 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + _cg_idx2 + 3);
                buf0[30 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + _cg_idx2 + 3);
              }
              _cg_rel1 += 1;
            }
          }
          {
            buf0[29 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 3);
            buf0[29 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 5, k + _cg_idx2 + 3);
            buf0[29 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 5, k + _cg_idx2 + 3);
            buf0[29 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 5, k + _cg_idx2 + 3);
            buf0[29 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 5, k + _cg_idx2 + 3);
            buf0[29 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 5, k + _cg_idx2 + 3);
            buf0[29 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 5, k + _cg_idx2 + 3);
            buf0[21 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 3);
            buf0[28 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 3);
            buf0[37 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 3);
            buf0[30 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 3);
            buf0[13 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 3);
            buf0[27 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 3);
            buf0[45 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 3);
            buf0[31 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 3);
            buf0[5 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 3);
            buf0[26 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 3);
            buf0[53 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + _cg_idx2 + 3);
          }
          {
            buf0[30 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 3);
            buf0[30 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6, k + _cg_idx2 + 3);
            buf0[30 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6, k + _cg_idx2 + 3);
            buf0[30 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6, k + _cg_idx2 + 3);
            buf0[30 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6, k + _cg_idx2 + 3);
            buf0[30 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 6, k + _cg_idx2 + 3);
            buf0[30 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 6, k + _cg_idx2 + 3);
            buf0[22 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 3);
            buf0[29 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 3);
            buf0[38 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 3);
            buf0[31 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 3);
            buf0[14 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 3);
            buf0[28 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 3);
            buf0[46 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 3);
            buf0[6 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 3);
            buf0[27 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 3);
            buf0[54 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + _cg_idx2 + 3);
          }
          {
            buf0[31 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 3);
            buf0[31 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7, k + _cg_idx2 + 3);
            buf0[31 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7, k + _cg_idx2 + 3);
            buf0[31 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7, k + _cg_idx2 + 3);
            buf0[31 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7, k + _cg_idx2 + 3);
            buf0[31 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 7, k + _cg_idx2 + 3);
            buf0[31 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 7, k + _cg_idx2 + 3);
            buf0[23 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 3);
            buf0[30 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 3);
            buf0[39 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 3);
            buf0[15 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 3);
            buf0[29 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 3);
            buf0[47 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 3);
            buf0[7 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 3);
            buf0[28 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 3);
            buf0[55 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + _cg_idx2 + 3);
          }
          {
            buf0[31 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 3);
            buf0[30 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 3);
            buf0[29 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 8, k + _cg_idx2 + 3);
          }
          {
            buf0[31 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9, k + _cg_idx2 + 3);
            buf0[30 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 9, k + _cg_idx2 + 3);
          }
          {
            buf0[31 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 10, k + _cg_idx2 + 3);
          }
          _cg_rel2 += 8;
        }
      }
      {
        buf0[40] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -3, k + 5);
      }
      {
        buf0[40] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2, k + 5);
        buf0[41] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -2, k + 5);
      }
      {
        buf0[40] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1, k + 5);
        buf0[41] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1, k + 5);
        buf0[42] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -1, k + 5);
      }
      {
        buf0[40] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j, k + 5);
        buf0[40] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j, k + 5);
        buf0[40] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j, k + 5);
        buf0[40] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j, k + 5);
        buf0[40] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j, k + 5);
        buf0[40] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j, k + 5);
        buf0[40] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j, k + 5);
        buf0[32] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 5);
        buf0[48] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 5);
        buf0[41] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 5);
        buf0[24] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 5);
        buf0[56] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 5);
        buf0[42] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 5);
        buf0[16] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 5);
        buf0[43] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 5);
      }
      {
        buf0[41] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[41] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1, k + 5);
        buf0[41] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1, k + 5);
        buf0[41] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1, k + 5);
        buf0[41] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1, k + 5);
        buf0[41] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 1, k + 5);
        buf0[41] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 1, k + 5);
        buf0[33] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[40] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[49] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[42] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[25] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[57] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[43] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[17] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[44] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
      }
      {
        buf0[42] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[42] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 2, k + 5);
        buf0[42] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 2, k + 5);
        buf0[42] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 2, k + 5);
        buf0[42] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 2, k + 5);
        buf0[42] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 2, k + 5);
        buf0[42] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 2, k + 5);
        buf0[34] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[41] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[50] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[43] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[26] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[40] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[58] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[44] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[18] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[45] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 2; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[43 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 5);
            buf0[43 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 3, k + 5);
            buf0[43 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 3, k + 5);
            buf0[43 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 3, k + 5);
            buf0[43 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 3, k + 5);
            buf0[43 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + _cg_idx1 + 3, k + 5);
            buf0[43 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + _cg_idx1 + 3, k + 5);
            buf0[35 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 5);
            buf0[42 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 5);
            buf0[51 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 5);
            buf0[44 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 5);
            buf0[27 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 5);
            buf0[41 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 5);
            buf0[59 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 5);
            buf0[45 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 5);
            buf0[19 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 5);
            buf0[40 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 5);
            buf0[46 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 5);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[45] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[45] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 5, k + 5);
        buf0[45] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 5, k + 5);
        buf0[45] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 5, k + 5);
        buf0[45] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 5, k + 5);
        buf0[45] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 5, k + 5);
        buf0[45] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 5, k + 5);
        buf0[37] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[44] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[53] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[46] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[29] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[43] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[61] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[47] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[21] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[42] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
      }
      {
        buf0[46] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[46] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6, k + 5);
        buf0[46] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6, k + 5);
        buf0[46] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6, k + 5);
        buf0[46] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6, k + 5);
        buf0[46] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 6, k + 5);
        buf0[46] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 6, k + 5);
        buf0[38] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[45] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[54] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[47] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[30] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[44] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[62] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[22] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[43] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
      }
      {
        buf0[47] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
        buf0[47] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7, k + 5);
        buf0[47] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7, k + 5);
        buf0[47] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7, k + 5);
        buf0[47] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7, k + 5);
        buf0[47] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 7, k + 5);
        buf0[47] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 7, k + 5);
        buf0[39] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
        buf0[46] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
        buf0[55] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
        buf0[31] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
        buf0[45] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
        buf0[63] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
        buf0[23] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
        buf0[44] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
      }
      {
        buf0[47] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8, k + 5);
        buf0[46] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8, k + 5);
        buf0[45] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 8, k + 5);
      }
      {
        buf0[47] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9, k + 5);
        buf0[46] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 9, k + 5);
      }
      {
        buf0[47] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 10, k + 5);
      }
      {
        buf0[48] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -3, k + 6);
      }
      {
        buf0[48] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2, k + 6);
        buf0[49] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -2, k + 6);
      }
      {
        buf0[48] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1, k + 6);
        buf0[49] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1, k + 6);
        buf0[50] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -1, k + 6);
      }
      {
        buf0[48] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[48] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j, k + 6);
        buf0[48] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j, k + 6);
        buf0[48] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j, k + 6);
        buf0[48] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j, k + 6);
        buf0[48] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j, k + 6);
        buf0[48] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j, k + 6);
        buf0[40] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[56] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[49] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[32] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[50] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[24] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[51] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 6);
      }
      {
        buf0[49] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[49] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1, k + 6);
        buf0[49] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf0[49] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1, k + 6);
        buf0[49] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1, k + 6);
        buf0[49] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 1, k + 6);
        buf0[49] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 1, k + 6);
        buf0[41] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[48] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[57] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[50] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[33] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[51] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[25] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[52] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
      }
      {
        buf0[50] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[50] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 2, k + 6);
        buf0[50] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 2, k + 6);
        buf0[50] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 2, k + 6);
        buf0[50] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 2, k + 6);
        buf0[50] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 2, k + 6);
        buf0[50] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 2, k + 6);
        buf0[42] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[49] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[58] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[51] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[34] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[48] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[52] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[26] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[53] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 2; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[51 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 6);
            buf0[51 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 3, k + 6);
            buf0[51 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 3, k + 6);
            buf0[51 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 3, k + 6);
            buf0[51 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 3, k + 6);
            buf0[51 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + _cg_idx1 + 3, k + 6);
            buf0[51 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + _cg_idx1 + 3, k + 6);
            buf0[43 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 6);
            buf0[50 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 6);
            buf0[59 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 6);
            buf0[52 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 6);
            buf0[35 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 6);
            buf0[49 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 6);
            buf0[53 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 6);
            buf0[27 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 6);
            buf0[48 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 6);
            buf0[54 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 6);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[53] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[53] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 5, k + 6);
        buf0[53] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 5, k + 6);
        buf0[53] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 5, k + 6);
        buf0[53] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 5, k + 6);
        buf0[53] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 5, k + 6);
        buf0[53] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 5, k + 6);
        buf0[45] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[52] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[61] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[54] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[37] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[51] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[55] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[29] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[50] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
      }
      {
        buf0[54] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[54] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6, k + 6);
        buf0[54] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6, k + 6);
        buf0[54] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6, k + 6);
        buf0[54] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6, k + 6);
        buf0[54] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 6, k + 6);
        buf0[54] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 6, k + 6);
        buf0[46] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[53] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[62] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[55] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[38] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[52] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[30] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[51] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
      }
      {
        buf0[55] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[55] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7, k + 6);
        buf0[55] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7, k + 6);
        buf0[55] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7, k + 6);
        buf0[55] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7, k + 6);
        buf0[55] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 7, k + 6);
        buf0[55] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 7, k + 6);
        buf0[47] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[54] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[63] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[39] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[53] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[31] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[52] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
      }
      {
        buf0[55] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8, k + 6);
        buf0[54] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8, k + 6);
        buf0[53] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 8, k + 6);
      }
      {
        buf0[55] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9, k + 6);
        buf0[54] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 9, k + 6);
      }
      {
        buf0[55] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 10, k + 6);
      }
      {
        buf0[56] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -3, k + 7);
      }
      {
        buf0[56] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2, k + 7);
        buf0[57] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -2, k + 7);
      }
      {
        buf0[56] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1, k + 7);
        buf0[57] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1, k + 7);
        buf0[58] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -1, k + 7);
      }
      {
        buf0[56] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j, k + 7);
        buf0[56] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j, k + 7);
        buf0[56] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j, k + 7);
        buf0[56] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j, k + 7);
        buf0[56] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j, k + 7);
        buf0[56] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j, k + 7);
        buf0[56] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j, k + 7);
        buf0[48] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 7);
        buf0[57] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 7);
        buf0[40] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 7);
        buf0[58] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 7);
        buf0[32] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 7);
        buf0[59] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 7);
      }
      {
        buf0[57] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[57] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1, k + 7);
        buf0[57] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1, k + 7);
        buf0[57] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1, k + 7);
        buf0[57] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1, k + 7);
        buf0[57] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 1, k + 7);
        buf0[57] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 1, k + 7);
        buf0[49] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[56] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[58] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[41] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[59] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[33] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[60] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
      }
      {
        buf0[58] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
        buf0[58] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 2, k + 7);
        buf0[58] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 2, k + 7);
        buf0[58] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 2, k + 7);
        buf0[58] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 2, k + 7);
        buf0[58] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 2, k + 7);
        buf0[58] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 2, k + 7);
        buf0[50] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
        buf0[57] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
        buf0[59] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
        buf0[42] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
        buf0[56] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
        buf0[60] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
        buf0[34] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
        buf0[61] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 2; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[59 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 7);
            buf0[59 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 3, k + 7);
            buf0[59 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 3, k + 7);
            buf0[59 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 3, k + 7);
            buf0[59 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 3, k + 7);
            buf0[59 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + _cg_idx1 + 3, k + 7);
            buf0[59 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + _cg_idx1 + 3, k + 7);
            buf0[51 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 7);
            buf0[58 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 7);
            buf0[60 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 7);
            buf0[43 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 7);
            buf0[57 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 7);
            buf0[61 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 7);
            buf0[35 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 7);
            buf0[56 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 7);
            buf0[62 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 3, k + 7);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[61] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[61] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 5, k + 7);
        buf0[61] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 5, k + 7);
        buf0[61] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 5, k + 7);
        buf0[61] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 5, k + 7);
        buf0[61] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 5, k + 7);
        buf0[61] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 5, k + 7);
        buf0[53] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[60] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[62] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[45] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[59] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[63] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[37] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[58] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
      }
      {
        buf0[62] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[62] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6, k + 7);
        buf0[62] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6, k + 7);
        buf0[62] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6, k + 7);
        buf0[62] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6, k + 7);
        buf0[62] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 6, k + 7);
        buf0[62] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 6, k + 7);
        buf0[54] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[61] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[63] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[46] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[60] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[38] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[59] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
      }
      {
        buf0[63] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[63] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf0[63] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf0[63] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7, k + 7);
        buf0[63] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7, k + 7);
        buf0[63] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 7, k + 7);
        buf0[63] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 7, k + 7);
        buf0[55] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[62] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[47] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[61] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[39] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[60] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
      }
      {
        buf0[63] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8, k + 7);
        buf0[62] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8, k + 7);
        buf0[61] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 8, k + 7);
      }
      {
        buf0[63] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9, k + 7);
        buf0[62] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 9, k + 7);
      }
      {
        buf0[63] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 10, k + 7);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[56 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf0[48 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf0[40 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
          }
          _cg_rel1 += 1;
        }
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[56 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 9);
            buf0[48 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 9);
          }
          _cg_rel1 += 1;
        }
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[56 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 10);
          }
          _cg_rel1 += 1;
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
            bOut(i + _cg_idx0, j + _cg_idx1, k + _cg_idx2) = buf0[rel];
          }
        }
      }
    }
  }
}
# 91 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/s3d/intermediate_gen/s3d.cu" 2

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
# 1 "VSBrick-laplacian5.py-HIP-8x8x8-8x8" 1
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
      bElem _cg_bIn000_vecbuf;
      bElem _cg_bIn_100_vecbuf;
      {
        // New offset [0, 0, -5]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_vecbuf = bIn.dat[neighbor4 * bIn.step + 192 + hipThreadIdx_x];
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, -4]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_vecbuf = bIn.dat[neighbor4 * bIn.step + 256 + hipThreadIdx_x];
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[4] * _cg_bIn000_reg;
        buf0[1] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, -3]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_vecbuf = bIn.dat[neighbor4 * bIn.step + 320 + hipThreadIdx_x];
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[3] * _cg_bIn000_reg;
        buf0[1] += dev_coeff[4] * _cg_bIn000_reg;
        buf0[2] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, -2]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_vecbuf = bIn.dat[neighbor4 * bIn.step + 384 + hipThreadIdx_x];
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[1] += dev_coeff[3] * _cg_bIn000_reg;
        buf0[2] += dev_coeff[4] * _cg_bIn000_reg;
        buf0[3] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, -1]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_vecbuf = bIn.dat[neighbor4 * bIn.step + 448 + hipThreadIdx_x];
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[1] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[2] += dev_coeff[3] * _cg_bIn000_reg;
        buf0[3] += dev_coeff[4] * _cg_bIn000_reg;
        buf0[4] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -5, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -4, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 4 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 32, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -3, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -2, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -1, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [-5, 0, 0]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = bIn.dat[neighbor12 * bIn.step + hipThreadIdx_x];
          _cg_bIn000_vecbuf = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [-4, 0, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 4 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 4, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [-3, 0, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [-2, 0, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [-1, 0, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 0]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[0] * _cg_bIn000_reg;
        buf0[1] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[2] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[3] += dev_coeff[3] * _cg_bIn000_reg;
        buf0[4] += dev_coeff[4] * _cg_bIn000_reg;
        buf0[5] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [1, 0, 0]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = _cg_bIn000_vecbuf;
          _cg_bIn000_vecbuf = bIn.dat[neighbor14 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [2, 0, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [3, 0, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [4, 0, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 4 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 4, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [5, 0, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 1, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 2, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 3, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 4, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 4 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 32, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 5, 0]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[0] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -5, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -4, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 4 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 32, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -3, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -2, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -1, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [-5, 0, 1]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = bIn.dat[neighbor12 * bIn.step + 64 + hipThreadIdx_x];
          _cg_bIn000_vecbuf = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[1] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [-4, 0, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 4 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 4, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[1] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [-3, 0, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[1] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [-2, 0, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[1] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [-1, 0, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[1] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 1]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[0] * _cg_bIn000_reg;
        buf0[0] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[2] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[3] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[4] += dev_coeff[3] * _cg_bIn000_reg;
        buf0[5] += dev_coeff[4] * _cg_bIn000_reg;
        buf0[6] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [1, 0, 1]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = _cg_bIn000_vecbuf;
          _cg_bIn000_vecbuf = bIn.dat[neighbor14 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[1] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [2, 0, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[1] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [3, 0, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[1] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [4, 0, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 4 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 4, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[1] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [5, 0, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[1] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 1, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 2, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 3, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 4, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 4 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 32, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 5, 1]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 64 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 64 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[1] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -5, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 128 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 128 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[2] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -4, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 128 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 128 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 4 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 32, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[2] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -3, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 128 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 128 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[2] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -2, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 128 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 128 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[2] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -1, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 128 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 128 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[2] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [-5, 0, 2]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = bIn.dat[neighbor12 * bIn.step + 128 + hipThreadIdx_x];
          _cg_bIn000_vecbuf = bIn.dat[neighbor13 * bIn.step + 128 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[2] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [-4, 0, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 4 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 4, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[2] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [-3, 0, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[2] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [-2, 0, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[2] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [-1, 0, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[2] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 2]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[2] += dev_coeff[0] * _cg_bIn000_reg;
        buf0[1] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[3] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[0] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[4] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[5] += dev_coeff[3] * _cg_bIn000_reg;
        buf0[6] += dev_coeff[4] * _cg_bIn000_reg;
        buf0[7] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [1, 0, 2]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = _cg_bIn000_vecbuf;
          _cg_bIn000_vecbuf = bIn.dat[neighbor14 * bIn.step + 128 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[2] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [2, 0, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[2] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [3, 0, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[2] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [4, 0, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 4 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 4, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[2] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [5, 0, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[2] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 1, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 128 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 128 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[2] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 2, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 128 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 128 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[2] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 3, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 128 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 128 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[2] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 4, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 128 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 128 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 4 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 32, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[2] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 5, 2]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 128 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 128 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[2] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -5, 3]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 192 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 192 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[3] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -4, 3]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 192 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 192 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 4 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 32, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[3] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -3, 3]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 192 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 192 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[3] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -2, 3]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 192 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 192 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[3] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -1, 3]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 192 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 192 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[3] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [-5, 0, 3]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = bIn.dat[neighbor12 * bIn.step + 192 + hipThreadIdx_x];
          _cg_bIn000_vecbuf = bIn.dat[neighbor13 * bIn.step + 192 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[3] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [-4, 0, 3]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 4 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 4, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[3] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [-3, 0, 3]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[3] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [-2, 0, 3]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[3] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [-1, 0, 3]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[3] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 3]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[3] += dev_coeff[0] * _cg_bIn000_reg;
        buf0[2] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[4] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[1] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[5] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[0] += dev_coeff[3] * _cg_bIn000_reg;
        buf0[6] += dev_coeff[3] * _cg_bIn000_reg;
        buf0[7] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [1, 0, 3]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = _cg_bIn000_vecbuf;
          _cg_bIn000_vecbuf = bIn.dat[neighbor14 * bIn.step + 192 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[3] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [2, 0, 3]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[3] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [3, 0, 3]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[3] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [4, 0, 3]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 4 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 4, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[3] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [5, 0, 3]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[3] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 1, 3]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 192 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 192 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[3] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 2, 3]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 192 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 192 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[3] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 3, 3]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 192 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 192 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[3] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 4, 3]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 192 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 192 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 4 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 32, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[3] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 5, 3]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 192 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 192 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[3] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -5, 4]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 256 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 256 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[4] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -4, 4]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 256 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 256 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 4 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 32, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[4] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -3, 4]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 256 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 256 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[4] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -2, 4]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 256 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 256 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[4] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -1, 4]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 256 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 256 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[4] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [-5, 0, 4]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = bIn.dat[neighbor12 * bIn.step + 256 + hipThreadIdx_x];
          _cg_bIn000_vecbuf = bIn.dat[neighbor13 * bIn.step + 256 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[4] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [-4, 0, 4]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 4 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 4, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[4] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [-3, 0, 4]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[4] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [-2, 0, 4]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[4] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [-1, 0, 4]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[4] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 4]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[4] += dev_coeff[0] * _cg_bIn000_reg;
        buf0[3] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[5] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[2] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[6] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[1] += dev_coeff[3] * _cg_bIn000_reg;
        buf0[7] += dev_coeff[3] * _cg_bIn000_reg;
        buf0[0] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [1, 0, 4]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = _cg_bIn000_vecbuf;
          _cg_bIn000_vecbuf = bIn.dat[neighbor14 * bIn.step + 256 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[4] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [2, 0, 4]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[4] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [3, 0, 4]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[4] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [4, 0, 4]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 4 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 4, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[4] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [5, 0, 4]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[4] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 1, 4]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 256 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 256 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[4] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 2, 4]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 256 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 256 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[4] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 3, 4]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 256 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 256 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[4] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 4, 4]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 256 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 256 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 4 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 32, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[4] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 5, 4]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 256 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 256 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[4] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -5, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 320 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 320 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[5] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -4, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 320 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 320 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 4 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 32, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[5] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -3, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 320 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 320 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[5] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -2, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 320 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 320 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[5] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -1, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 320 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 320 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[5] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [-5, 0, 5]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = bIn.dat[neighbor12 * bIn.step + 320 + hipThreadIdx_x];
          _cg_bIn000_vecbuf = bIn.dat[neighbor13 * bIn.step + 320 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[5] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [-4, 0, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 4 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 4, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[5] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [-3, 0, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[5] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [-2, 0, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[5] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [-1, 0, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[5] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 5]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[5] += dev_coeff[0] * _cg_bIn000_reg;
        buf0[4] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[6] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[3] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[7] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[2] += dev_coeff[3] * _cg_bIn000_reg;
        buf0[1] += dev_coeff[4] * _cg_bIn000_reg;
        buf0[0] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [1, 0, 5]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = _cg_bIn000_vecbuf;
          _cg_bIn000_vecbuf = bIn.dat[neighbor14 * bIn.step + 320 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[5] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [2, 0, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[5] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [3, 0, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[5] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [4, 0, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 4 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 4, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[5] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [5, 0, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[5] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 1, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 320 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 320 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[5] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 2, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 320 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 320 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[5] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 3, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 320 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 320 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[5] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 4, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 320 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 320 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 4 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 32, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[5] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 5, 5]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 320 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 320 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[5] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -5, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -4, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 4 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 32, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -3, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -2, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -1, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [-5, 0, 6]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = bIn.dat[neighbor12 * bIn.step + 384 + hipThreadIdx_x];
          _cg_bIn000_vecbuf = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[6] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [-4, 0, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 4 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 4, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[6] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [-3, 0, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[6] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [-2, 0, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[6] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [-1, 0, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[6] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 6]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[0] * _cg_bIn000_reg;
        buf0[5] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[7] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[4] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[3] += dev_coeff[3] * _cg_bIn000_reg;
        buf0[2] += dev_coeff[4] * _cg_bIn000_reg;
        buf0[1] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [1, 0, 6]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = _cg_bIn000_vecbuf;
          _cg_bIn000_vecbuf = bIn.dat[neighbor14 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[6] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [2, 0, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[6] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [3, 0, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[6] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [4, 0, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 4 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 4, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[6] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [5, 0, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[6] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 1, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 2, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 3, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 4, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 4 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 32, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 5, 6]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 384 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 384 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[6] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -5, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -4, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 4 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 32, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -3, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -2, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, -1, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor10 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [-5, 0, 7]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = bIn.dat[neighbor12 * bIn.step + 448 + hipThreadIdx_x];
          _cg_bIn000_vecbuf = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[7] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [-4, 0, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 4 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 4, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[7] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [-3, 0, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[7] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [-2, 0, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[7] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [-1, 0, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[7] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 7]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[0] * _cg_bIn000_reg;
        buf0[6] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[5] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[4] += dev_coeff[3] * _cg_bIn000_reg;
        buf0[3] += dev_coeff[4] * _cg_bIn000_reg;
        buf0[2] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [1, 0, 7]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn_100_vecbuf = _cg_bIn000_vecbuf;
          _cg_bIn000_vecbuf = bIn.dat[neighbor14 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[7] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [2, 0, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[7] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [3, 0, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 3 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 5, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[7] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [4, 0, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 4 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 4, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[7] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [5, 0, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_100_vecbuf ,_cg_bIn000_vecbuf, 5 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_100_vecbuf, _cg_bIn000_vecbuf, 3, 8, hipThreadIdx_x & 7);
          _cg_bIn000_reg = _cg_vectmp0;
        }
        buf0[7] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 1, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[1] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 2, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[2] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 3, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 3 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 40, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[3] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 4, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 4 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 32, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[4] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 5, 7]
        bElem _cg_bIn000_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor16 * bIn.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor13 * bIn.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 5 -> _cg_bIn000_vecbuf
          dev_shl(_cg_bIn000_vecbuf, _cg_vectmp1, _cg_vectmp0, 24, 64, hipThreadIdx_x);
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 8]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_vecbuf = bIn.dat[neighbor22 * bIn.step + hipThreadIdx_x];
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[1] * _cg_bIn000_reg;
        buf0[6] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[5] += dev_coeff[3] * _cg_bIn000_reg;
        buf0[4] += dev_coeff[4] * _cg_bIn000_reg;
        buf0[3] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 9]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_vecbuf = bIn.dat[neighbor22 * bIn.step + 64 + hipThreadIdx_x];
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[2] * _cg_bIn000_reg;
        buf0[6] += dev_coeff[3] * _cg_bIn000_reg;
        buf0[5] += dev_coeff[4] * _cg_bIn000_reg;
        buf0[4] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 10]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_vecbuf = bIn.dat[neighbor22 * bIn.step + 128 + hipThreadIdx_x];
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[3] * _cg_bIn000_reg;
        buf0[6] += dev_coeff[4] * _cg_bIn000_reg;
        buf0[5] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 11]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_vecbuf = bIn.dat[neighbor22 * bIn.step + 192 + hipThreadIdx_x];
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[4] * _cg_bIn000_reg;
        buf0[6] += dev_coeff[5] * _cg_bIn000_reg;
      }
      {
        // New offset [0, 0, 12]
        bElem _cg_bIn000_reg;
        {
          _cg_bIn000_vecbuf = bIn.dat[neighbor22 * bIn.step + 256 + hipThreadIdx_x];
          _cg_bIn000_reg = _cg_bIn000_vecbuf;
        }
        buf0[7] += dev_coeff[5] * _cg_bIn000_reg;
      }
    }
    bElem *bOut_ref = &bOut.dat[neighbor13 * bOut.step];
    for (long sti = 0; sti < 8; ++sti)
    {
      bOut_ref[sti * 64 + hipThreadIdx_x] = buf0[sti];
    }
  }
}
# 126 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/s3d/intermediate_gen/s3d.cu" 2

}
#define bIn(a, b, c) arr_in[c][b][a]
#define bOut(a, b, c) arr_out[c][b][a]
__global__ void s3d_codegen5(bElem (*arr_in)[STRIDE1][STRIDE0], bElem (*arr_out)[STRIDE1][STRIDE0], bElem *dev_coeff) {
    long k = OFF2 + (blockIdx.z * TILE2);
    long j = OFF1 + (blockIdx.y * TILE1);
    long i = OFF0 + (blockIdx.x * VECSIZE);
# 1 "VSTile-laplacian5.py-HIP-8x8x64" 1
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
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[0 + rel] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -5);
          }
          _cg_rel1 += 1;
        }
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[0 + rel] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -4);
            buf0[8 + rel] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -4);
          }
          _cg_rel1 += 1;
        }
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[0 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -3);
            buf0[8 + rel] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -3);
            buf0[16 + rel] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -3);
          }
          _cg_rel1 += 1;
        }
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[0 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -2);
            buf0[8 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -2);
            buf0[16 + rel] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -2);
            buf0[24 + rel] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -2);
          }
          _cg_rel1 += 1;
        }
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[0 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf0[8 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf0[16 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf0[24 + rel] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
            buf0[32 + rel] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + -1);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[0] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -5, k);
      }
      {
        buf0[0] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -4, k);
        buf0[1] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -4, k);
      }
      {
        buf0[0] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -3, k);
        buf0[1] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -3, k);
        buf0[2] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -3, k);
      }
      {
        buf0[0] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2, k);
        buf0[1] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -2, k);
        buf0[2] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -2, k);
        buf0[3] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -2, k);
      }
      {
        buf0[0] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1, k);
        buf0[1] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1, k);
        buf0[2] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -1, k);
        buf0[3] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -1, k);
        buf0[4] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -1, k);
      }
      {
        buf0[0] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j, k);
        buf0[0] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j, k);
        buf0[0] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j, k);
        buf0[0] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j, k);
        buf0[0] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j, k);
        buf0[0] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j, k);
        buf0[0] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j, k);
        buf0[0] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j, k);
        buf0[0] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j, k);
        buf0[0] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j, k);
        buf0[0] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j, k);
        buf0[8] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k);
        buf0[1] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k);
        buf0[16] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k);
        buf0[2] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k);
        buf0[24] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k);
        buf0[3] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k);
        buf0[32] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j, k);
        buf0[4] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j, k);
        buf0[40] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j, k);
        buf0[5] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j, k);
      }
      {
        buf0[1] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[1] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1, k);
        buf0[1] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1, k);
        buf0[1] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1, k);
        buf0[1] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1, k);
        buf0[1] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 1, k);
        buf0[1] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 1, k);
        buf0[1] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 1, k);
        buf0[1] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 1, k);
        buf0[1] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 1, k);
        buf0[1] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 1, k);
        buf0[0] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[9] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[2] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[17] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[3] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[25] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[4] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[33] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[5] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[41] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 1, k);
        buf0[6] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 1, k);
      }
      {
        buf0[2] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 2, k);
        buf0[2] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 2, k);
        buf0[2] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 2, k);
        buf0[2] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 2, k);
        buf0[2] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 2, k);
        buf0[2] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 2, k);
        buf0[2] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 2, k);
        buf0[2] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 2, k);
        buf0[2] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 2, k);
        buf0[2] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 2, k);
        buf0[2] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 2, k);
        buf0[1] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k);
        buf0[10] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k);
        buf0[3] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k);
        buf0[0] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k);
        buf0[18] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k);
        buf0[4] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k);
        buf0[26] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k);
        buf0[5] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k);
        buf0[34] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 2, k);
        buf0[6] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 2, k);
        buf0[42] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 2, k);
        buf0[7] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 2, k);
      }
      {
        buf0[3] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 3, k);
        buf0[3] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 3, k);
        buf0[3] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 3, k);
        buf0[3] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 3, k);
        buf0[3] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 3, k);
        buf0[3] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 3, k);
        buf0[3] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 3, k);
        buf0[3] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 3, k);
        buf0[3] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 3, k);
        buf0[3] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 3, k);
        buf0[3] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 3, k);
        buf0[2] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k);
        buf0[11] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k);
        buf0[4] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k);
        buf0[1] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k);
        buf0[19] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k);
        buf0[5] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k);
        buf0[0] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k);
        buf0[27] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k);
        buf0[6] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k);
        buf0[35] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 3, k);
        buf0[7] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 3, k);
        buf0[43] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 3, k);
      }
      {
        buf0[4] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 4, k);
        buf0[4] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 4, k);
        buf0[4] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 4, k);
        buf0[4] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 4, k);
        buf0[4] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 4, k);
        buf0[4] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 4, k);
        buf0[4] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 4, k);
        buf0[4] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 4, k);
        buf0[4] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 4, k);
        buf0[4] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 4, k);
        buf0[4] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 4, k);
        buf0[3] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k);
        buf0[12] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k);
        buf0[5] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k);
        buf0[2] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k);
        buf0[20] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k);
        buf0[6] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k);
        buf0[1] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k);
        buf0[28] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k);
        buf0[7] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k);
        buf0[0] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 4, k);
        buf0[36] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 4, k);
        buf0[44] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 4, k);
      }
      {
        buf0[5] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 5, k);
        buf0[5] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 5, k);
        buf0[5] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 5, k);
        buf0[5] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 5, k);
        buf0[5] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 5, k);
        buf0[5] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 5, k);
        buf0[5] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 5, k);
        buf0[5] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 5, k);
        buf0[5] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 5, k);
        buf0[5] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 5, k);
        buf0[5] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 5, k);
        buf0[4] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k);
        buf0[13] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k);
        buf0[6] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k);
        buf0[3] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k);
        buf0[21] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k);
        buf0[7] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k);
        buf0[2] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k);
        buf0[29] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k);
        buf0[1] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 5, k);
        buf0[37] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 5, k);
        buf0[0] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 5, k);
        buf0[45] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 5, k);
      }
      {
        buf0[6] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[6] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6, k);
        buf0[6] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6, k);
        buf0[6] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6, k);
        buf0[6] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6, k);
        buf0[6] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 6, k);
        buf0[6] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 6, k);
        buf0[6] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 6, k);
        buf0[6] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 6, k);
        buf0[6] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 6, k);
        buf0[6] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 6, k);
        buf0[5] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[14] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[7] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[4] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[22] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[3] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[30] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[2] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[38] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[1] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 6, k);
        buf0[46] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 6, k);
      }
      {
        buf0[7] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7, k);
        buf0[7] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7, k);
        buf0[7] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7, k);
        buf0[7] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7, k);
        buf0[7] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7, k);
        buf0[7] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 7, k);
        buf0[7] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 7, k);
        buf0[7] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 7, k);
        buf0[7] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 7, k);
        buf0[7] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 7, k);
        buf0[7] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 7, k);
        buf0[6] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k);
        buf0[15] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k);
        buf0[5] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k);
        buf0[23] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k);
        buf0[4] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k);
        buf0[31] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k);
        buf0[3] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 7, k);
        buf0[39] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 7, k);
        buf0[2] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 7, k);
        buf0[47] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 7, k);
      }
      {
        buf0[7] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8, k);
        buf0[6] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8, k);
        buf0[5] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 8, k);
        buf0[4] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 8, k);
        buf0[3] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 8, k);
      }
      {
        buf0[7] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9, k);
        buf0[6] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 9, k);
        buf0[5] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 9, k);
        buf0[4] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 9, k);
      }
      {
        buf0[7] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 10, k);
        buf0[6] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 10, k);
        buf0[5] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 10, k);
      }
      {
        buf0[7] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 11, k);
        buf0[6] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 11, k);
      }
      {
        buf0[7] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 12, k);
      }
      {
        buf0[8] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -5, k + 1);
      }
      {
        buf0[8] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -4, k + 1);
        buf0[9] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -4, k + 1);
      }
      {
        buf0[8] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -3, k + 1);
        buf0[9] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -3, k + 1);
        buf0[10] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -3, k + 1);
      }
      {
        buf0[8] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2, k + 1);
        buf0[9] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -2, k + 1);
        buf0[10] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -2, k + 1);
        buf0[11] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -2, k + 1);
      }
      {
        buf0[8] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1, k + 1);
        buf0[9] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1, k + 1);
        buf0[10] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -1, k + 1);
        buf0[11] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -1, k + 1);
        buf0[12] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -1, k + 1);
      }
      {
        buf0[8] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[8] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j, k + 1);
        buf0[8] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j, k + 1);
        buf0[8] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j, k + 1);
        buf0[8] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j, k + 1);
        buf0[8] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j, k + 1);
        buf0[8] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j, k + 1);
        buf0[8] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j, k + 1);
        buf0[8] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j, k + 1);
        buf0[8] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j, k + 1);
        buf0[8] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j, k + 1);
        buf0[0] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[16] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[9] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[24] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[10] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[32] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[11] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[40] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[12] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[48] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j, k + 1);
        buf0[13] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j, k + 1);
      }
      {
        buf0[9] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[9] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1, k + 1);
        buf0[9] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1, k + 1);
        buf0[9] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1, k + 1);
        buf0[9] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1, k + 1);
        buf0[9] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 1, k + 1);
        buf0[9] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 1, k + 1);
        buf0[9] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 1, k + 1);
        buf0[9] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 1, k + 1);
        buf0[9] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 1, k + 1);
        buf0[9] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 1, k + 1);
        buf0[1] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[8] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[17] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[10] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[25] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[11] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[33] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[12] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[41] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[13] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[49] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
        buf0[14] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 1, k + 1);
      }
      {
        buf0[10] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[10] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 2, k + 1);
        buf0[10] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 2, k + 1);
        buf0[10] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 2, k + 1);
        buf0[10] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 2, k + 1);
        buf0[10] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 2, k + 1);
        buf0[10] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 2, k + 1);
        buf0[10] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 2, k + 1);
        buf0[10] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 2, k + 1);
        buf0[10] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 2, k + 1);
        buf0[10] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 2, k + 1);
        buf0[2] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[9] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[18] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[11] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[8] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[26] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[12] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[34] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[13] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[42] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[14] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[50] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
        buf0[15] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 2, k + 1);
      }
      {
        buf0[11] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 3, k + 1);
        buf0[11] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 3, k + 1);
        buf0[11] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 3, k + 1);
        buf0[11] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 3, k + 1);
        buf0[11] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 3, k + 1);
        buf0[11] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 3, k + 1);
        buf0[11] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 3, k + 1);
        buf0[11] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 3, k + 1);
        buf0[11] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 3, k + 1);
        buf0[11] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 3, k + 1);
        buf0[11] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 3, k + 1);
        buf0[3] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 1);
        buf0[10] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 1);
        buf0[19] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 1);
        buf0[12] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 1);
        buf0[9] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 1);
        buf0[27] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 1);
        buf0[13] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 1);
        buf0[8] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 1);
        buf0[35] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 1);
        buf0[14] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 1);
        buf0[43] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 3, k + 1);
        buf0[15] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 3, k + 1);
        buf0[51] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 3, k + 1);
      }
      {
        buf0[12] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 4, k + 1);
        buf0[12] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 4, k + 1);
        buf0[12] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 4, k + 1);
        buf0[12] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 4, k + 1);
        buf0[12] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 4, k + 1);
        buf0[12] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 4, k + 1);
        buf0[12] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 4, k + 1);
        buf0[12] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 4, k + 1);
        buf0[12] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 4, k + 1);
        buf0[12] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 4, k + 1);
        buf0[12] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 4, k + 1);
        buf0[4] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 1);
        buf0[11] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 1);
        buf0[20] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 1);
        buf0[13] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 1);
        buf0[10] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 1);
        buf0[28] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 1);
        buf0[14] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 1);
        buf0[9] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 1);
        buf0[36] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 1);
        buf0[15] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 1);
        buf0[8] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 4, k + 1);
        buf0[44] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 4, k + 1);
        buf0[52] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 4, k + 1);
      }
      {
        buf0[13] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[13] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 5, k + 1);
        buf0[13] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 5, k + 1);
        buf0[13] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 5, k + 1);
        buf0[13] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 5, k + 1);
        buf0[13] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 5, k + 1);
        buf0[13] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 5, k + 1);
        buf0[13] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 5, k + 1);
        buf0[13] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 5, k + 1);
        buf0[13] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 5, k + 1);
        buf0[13] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 5, k + 1);
        buf0[5] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[12] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[21] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[14] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[11] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[29] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[15] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[10] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[37] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[9] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[45] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[8] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
        buf0[53] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 5, k + 1);
      }
      {
        buf0[14] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[14] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6, k + 1);
        buf0[14] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6, k + 1);
        buf0[14] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6, k + 1);
        buf0[14] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6, k + 1);
        buf0[14] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 6, k + 1);
        buf0[14] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 6, k + 1);
        buf0[14] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 6, k + 1);
        buf0[14] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 6, k + 1);
        buf0[14] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 6, k + 1);
        buf0[14] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 6, k + 1);
        buf0[6] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[13] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[22] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[15] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[12] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[30] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[11] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[38] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[10] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[46] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[9] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
        buf0[54] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 6, k + 1);
      }
      {
        buf0[15] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[15] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7, k + 1);
        buf0[15] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7, k + 1);
        buf0[15] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7, k + 1);
        buf0[15] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7, k + 1);
        buf0[15] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 7, k + 1);
        buf0[15] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 7, k + 1);
        buf0[15] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 7, k + 1);
        buf0[15] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 7, k + 1);
        buf0[15] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 7, k + 1);
        buf0[15] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 7, k + 1);
        buf0[7] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[14] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[23] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[13] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[31] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[12] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[39] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[11] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[47] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[10] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
        buf0[55] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 7, k + 1);
      }
      {
        buf0[15] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8, k + 1);
        buf0[14] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8, k + 1);
        buf0[13] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 8, k + 1);
        buf0[12] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 8, k + 1);
        buf0[11] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 8, k + 1);
      }
      {
        buf0[15] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9, k + 1);
        buf0[14] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 9, k + 1);
        buf0[13] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 9, k + 1);
        buf0[12] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 9, k + 1);
      }
      {
        buf0[15] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 10, k + 1);
        buf0[14] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 10, k + 1);
        buf0[13] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 10, k + 1);
      }
      {
        buf0[15] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 11, k + 1);
        buf0[14] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 11, k + 1);
      }
      {
        buf0[15] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 12, k + 1);
      }
      {
        buf0[16] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -5, k + 2);
      }
      {
        buf0[16] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -4, k + 2);
        buf0[17] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -4, k + 2);
      }
      {
        buf0[16] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -3, k + 2);
        buf0[17] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -3, k + 2);
        buf0[18] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -3, k + 2);
      }
      {
        buf0[16] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2, k + 2);
        buf0[17] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -2, k + 2);
        buf0[18] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -2, k + 2);
        buf0[19] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -2, k + 2);
      }
      {
        buf0[16] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1, k + 2);
        buf0[17] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1, k + 2);
        buf0[18] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -1, k + 2);
        buf0[19] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -1, k + 2);
        buf0[20] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -1, k + 2);
      }
      {
        buf0[16] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j, k + 2);
        buf0[16] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j, k + 2);
        buf0[16] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j, k + 2);
        buf0[16] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j, k + 2);
        buf0[16] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j, k + 2);
        buf0[16] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j, k + 2);
        buf0[16] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j, k + 2);
        buf0[16] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j, k + 2);
        buf0[16] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j, k + 2);
        buf0[16] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j, k + 2);
        buf0[16] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j, k + 2);
        buf0[8] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 2);
        buf0[24] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 2);
        buf0[17] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 2);
        buf0[0] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 2);
        buf0[32] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 2);
        buf0[18] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 2);
        buf0[40] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 2);
        buf0[19] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 2);
        buf0[48] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j, k + 2);
        buf0[20] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j, k + 2);
        buf0[56] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j, k + 2);
        buf0[21] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j, k + 2);
      }
      {
        buf0[17] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[17] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1, k + 2);
        buf0[17] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1, k + 2);
        buf0[17] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1, k + 2);
        buf0[17] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1, k + 2);
        buf0[17] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 1, k + 2);
        buf0[17] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 1, k + 2);
        buf0[17] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 1, k + 2);
        buf0[17] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 1, k + 2);
        buf0[17] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 1, k + 2);
        buf0[17] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 1, k + 2);
        buf0[9] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[16] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[25] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[18] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[1] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[33] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[19] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[41] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[20] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[49] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[21] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[57] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
        buf0[22] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 1, k + 2);
      }
      {
        buf0[18] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[18] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 2, k + 2);
        buf0[18] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 2, k + 2);
        buf0[18] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 2, k + 2);
        buf0[18] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 2, k + 2);
        buf0[18] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 2, k + 2);
        buf0[18] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 2, k + 2);
        buf0[18] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 2, k + 2);
        buf0[18] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 2, k + 2);
        buf0[18] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 2, k + 2);
        buf0[18] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 2, k + 2);
        buf0[10] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[17] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[26] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[19] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[2] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[16] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[34] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[20] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[42] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[21] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[50] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[22] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[58] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
        buf0[23] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 2, k + 2);
      }
      {
        buf0[19] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 3, k + 2);
        buf0[19] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 3, k + 2);
        buf0[19] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 3, k + 2);
        buf0[19] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 3, k + 2);
        buf0[19] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 3, k + 2);
        buf0[19] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 3, k + 2);
        buf0[19] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 3, k + 2);
        buf0[19] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 3, k + 2);
        buf0[19] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 3, k + 2);
        buf0[19] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 3, k + 2);
        buf0[19] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 3, k + 2);
        buf0[11] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 2);
        buf0[18] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 2);
        buf0[27] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 2);
        buf0[20] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 2);
        buf0[3] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 2);
        buf0[17] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 2);
        buf0[35] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 2);
        buf0[21] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 2);
        buf0[16] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 2);
        buf0[43] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 2);
        buf0[22] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 2);
        buf0[51] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 3, k + 2);
        buf0[23] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 3, k + 2);
        buf0[59] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 3, k + 2);
      }
      {
        buf0[20] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 4, k + 2);
        buf0[20] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 4, k + 2);
        buf0[20] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 4, k + 2);
        buf0[20] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 4, k + 2);
        buf0[20] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 4, k + 2);
        buf0[20] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 4, k + 2);
        buf0[20] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 4, k + 2);
        buf0[20] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 4, k + 2);
        buf0[20] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 4, k + 2);
        buf0[20] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 4, k + 2);
        buf0[20] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 4, k + 2);
        buf0[12] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 2);
        buf0[19] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 2);
        buf0[28] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 2);
        buf0[21] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 2);
        buf0[4] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 2);
        buf0[18] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 2);
        buf0[36] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 2);
        buf0[22] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 2);
        buf0[17] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 2);
        buf0[44] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 2);
        buf0[23] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 2);
        buf0[16] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 4, k + 2);
        buf0[52] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 4, k + 2);
        buf0[60] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 4, k + 2);
      }
      {
        buf0[21] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[21] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 5, k + 2);
        buf0[21] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 5, k + 2);
        buf0[21] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 5, k + 2);
        buf0[21] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 5, k + 2);
        buf0[21] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 5, k + 2);
        buf0[21] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 5, k + 2);
        buf0[21] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 5, k + 2);
        buf0[21] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 5, k + 2);
        buf0[21] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 5, k + 2);
        buf0[21] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 5, k + 2);
        buf0[13] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[20] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[29] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[22] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[5] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[19] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[37] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[23] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[18] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[45] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[17] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[53] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[16] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
        buf0[61] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 5, k + 2);
      }
      {
        buf0[22] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[22] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6, k + 2);
        buf0[22] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6, k + 2);
        buf0[22] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6, k + 2);
        buf0[22] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6, k + 2);
        buf0[22] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 6, k + 2);
        buf0[22] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 6, k + 2);
        buf0[22] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 6, k + 2);
        buf0[22] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 6, k + 2);
        buf0[22] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 6, k + 2);
        buf0[22] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 6, k + 2);
        buf0[14] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[21] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[30] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[23] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[6] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[20] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[38] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[19] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[46] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[18] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[54] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[17] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
        buf0[62] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 6, k + 2);
      }
      {
        buf0[23] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
        buf0[23] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7, k + 2);
        buf0[23] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7, k + 2);
        buf0[23] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7, k + 2);
        buf0[23] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7, k + 2);
        buf0[23] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 7, k + 2);
        buf0[23] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 7, k + 2);
        buf0[23] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 7, k + 2);
        buf0[23] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 7, k + 2);
        buf0[23] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 7, k + 2);
        buf0[23] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 7, k + 2);
        buf0[15] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
        buf0[22] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
        buf0[31] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
        buf0[7] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
        buf0[21] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
        buf0[39] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
        buf0[20] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
        buf0[47] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
        buf0[19] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
        buf0[55] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
        buf0[18] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
        buf0[63] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 7, k + 2);
      }
      {
        buf0[23] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8, k + 2);
        buf0[22] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8, k + 2);
        buf0[21] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 8, k + 2);
        buf0[20] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 8, k + 2);
        buf0[19] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 8, k + 2);
      }
      {
        buf0[23] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9, k + 2);
        buf0[22] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 9, k + 2);
        buf0[21] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 9, k + 2);
        buf0[20] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 9, k + 2);
      }
      {
        buf0[23] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 10, k + 2);
        buf0[22] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 10, k + 2);
        buf0[21] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 10, k + 2);
      }
      {
        buf0[23] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 11, k + 2);
        buf0[22] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 11, k + 2);
      }
      {
        buf0[23] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 12, k + 2);
      }
      {
        buf0[24] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -5, k + 3);
      }
      {
        buf0[24] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -4, k + 3);
        buf0[25] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -4, k + 3);
      }
      {
        buf0[24] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -3, k + 3);
        buf0[25] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -3, k + 3);
        buf0[26] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -3, k + 3);
      }
      {
        buf0[24] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2, k + 3);
        buf0[25] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -2, k + 3);
        buf0[26] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -2, k + 3);
        buf0[27] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -2, k + 3);
      }
      {
        buf0[24] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1, k + 3);
        buf0[25] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1, k + 3);
        buf0[26] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -1, k + 3);
        buf0[27] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -1, k + 3);
        buf0[28] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -1, k + 3);
      }
      {
        buf0[24] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j, k + 3);
        buf0[24] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j, k + 3);
        buf0[24] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j, k + 3);
        buf0[24] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j, k + 3);
        buf0[24] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j, k + 3);
        buf0[24] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j, k + 3);
        buf0[24] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j, k + 3);
        buf0[24] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j, k + 3);
        buf0[24] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j, k + 3);
        buf0[24] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j, k + 3);
        buf0[24] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j, k + 3);
        buf0[16] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 3);
        buf0[32] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 3);
        buf0[25] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 3);
        buf0[8] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 3);
        buf0[40] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 3);
        buf0[26] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 3);
        buf0[0] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 3);
        buf0[48] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 3);
        buf0[27] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 3);
        buf0[56] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j, k + 3);
        buf0[28] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j, k + 3);
        buf0[29] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j, k + 3);
      }
      {
        buf0[25] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1, k + 3);
        buf0[25] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1, k + 3);
        buf0[25] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1, k + 3);
        buf0[25] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1, k + 3);
        buf0[25] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1, k + 3);
        buf0[25] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 1, k + 3);
        buf0[25] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 1, k + 3);
        buf0[25] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 1, k + 3);
        buf0[25] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 1, k + 3);
        buf0[25] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 1, k + 3);
        buf0[25] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 1, k + 3);
        buf0[17] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 3);
        buf0[24] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 3);
        buf0[33] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 3);
        buf0[26] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 3);
        buf0[9] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 3);
        buf0[41] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 3);
        buf0[27] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 3);
        buf0[1] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 3);
        buf0[49] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 3);
        buf0[28] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 3);
        buf0[57] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 1, k + 3);
        buf0[29] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 1, k + 3);
        buf0[30] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 1, k + 3);
      }
      {
        buf0[26] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 2, k + 3);
        buf0[26] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 2, k + 3);
        buf0[26] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 2, k + 3);
        buf0[26] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 2, k + 3);
        buf0[26] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 2, k + 3);
        buf0[26] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 2, k + 3);
        buf0[26] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 2, k + 3);
        buf0[26] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 2, k + 3);
        buf0[26] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 2, k + 3);
        buf0[26] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 2, k + 3);
        buf0[26] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 2, k + 3);
        buf0[18] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 3);
        buf0[25] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 3);
        buf0[34] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 3);
        buf0[27] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 3);
        buf0[10] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 3);
        buf0[24] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 3);
        buf0[42] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 3);
        buf0[28] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 3);
        buf0[2] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 3);
        buf0[50] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 3);
        buf0[29] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 3);
        buf0[58] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 2, k + 3);
        buf0[30] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 2, k + 3);
        buf0[31] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 2, k + 3);
      }
      {
        buf0[27] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 3, k + 3);
        buf0[27] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 3, k + 3);
        buf0[27] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 3, k + 3);
        buf0[27] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 3, k + 3);
        buf0[27] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 3, k + 3);
        buf0[27] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 3, k + 3);
        buf0[27] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 3, k + 3);
        buf0[27] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 3, k + 3);
        buf0[27] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 3, k + 3);
        buf0[27] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 3, k + 3);
        buf0[27] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 3, k + 3);
        buf0[19] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 3);
        buf0[26] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 3);
        buf0[35] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 3);
        buf0[28] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 3);
        buf0[11] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 3);
        buf0[25] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 3);
        buf0[43] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 3);
        buf0[29] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 3);
        buf0[3] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 3);
        buf0[24] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 3);
        buf0[51] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 3);
        buf0[30] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 3);
        buf0[59] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 3, k + 3);
        buf0[31] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 3, k + 3);
      }
      {
        buf0[28] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 4, k + 3);
        buf0[28] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 4, k + 3);
        buf0[28] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 4, k + 3);
        buf0[28] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 4, k + 3);
        buf0[28] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 4, k + 3);
        buf0[28] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 4, k + 3);
        buf0[28] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 4, k + 3);
        buf0[28] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 4, k + 3);
        buf0[28] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 4, k + 3);
        buf0[28] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 4, k + 3);
        buf0[28] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 4, k + 3);
        buf0[20] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 3);
        buf0[27] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 3);
        buf0[36] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 3);
        buf0[29] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 3);
        buf0[12] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 3);
        buf0[26] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 3);
        buf0[44] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 3);
        buf0[30] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 3);
        buf0[4] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 3);
        buf0[25] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 3);
        buf0[52] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 3);
        buf0[31] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 3);
        buf0[24] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 4, k + 3);
        buf0[60] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 4, k + 3);
      }
      {
        buf0[29] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 5, k + 3);
        buf0[29] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 5, k + 3);
        buf0[29] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 5, k + 3);
        buf0[29] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 5, k + 3);
        buf0[29] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 5, k + 3);
        buf0[29] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 5, k + 3);
        buf0[29] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 5, k + 3);
        buf0[29] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 5, k + 3);
        buf0[29] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 5, k + 3);
        buf0[29] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 5, k + 3);
        buf0[29] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 5, k + 3);
        buf0[21] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 3);
        buf0[28] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 3);
        buf0[37] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 3);
        buf0[30] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 3);
        buf0[13] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 3);
        buf0[27] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 3);
        buf0[45] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 3);
        buf0[31] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 3);
        buf0[5] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 3);
        buf0[26] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 3);
        buf0[53] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 3);
        buf0[25] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 5, k + 3);
        buf0[61] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 5, k + 3);
        buf0[24] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 5, k + 3);
      }
      {
        buf0[30] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6, k + 3);
        buf0[30] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6, k + 3);
        buf0[30] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6, k + 3);
        buf0[30] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6, k + 3);
        buf0[30] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6, k + 3);
        buf0[30] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 6, k + 3);
        buf0[30] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 6, k + 3);
        buf0[30] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 6, k + 3);
        buf0[30] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 6, k + 3);
        buf0[30] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 6, k + 3);
        buf0[30] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 6, k + 3);
        buf0[22] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 3);
        buf0[29] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 3);
        buf0[38] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 3);
        buf0[31] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 3);
        buf0[14] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 3);
        buf0[28] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 3);
        buf0[46] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 3);
        buf0[6] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 3);
        buf0[27] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 3);
        buf0[54] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 3);
        buf0[26] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 6, k + 3);
        buf0[62] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 6, k + 3);
        buf0[25] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 6, k + 3);
      }
      {
        buf0[31] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7, k + 3);
        buf0[31] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7, k + 3);
        buf0[31] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7, k + 3);
        buf0[31] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7, k + 3);
        buf0[31] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7, k + 3);
        buf0[31] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 7, k + 3);
        buf0[31] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 7, k + 3);
        buf0[31] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 7, k + 3);
        buf0[31] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 7, k + 3);
        buf0[31] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 7, k + 3);
        buf0[31] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 7, k + 3);
        buf0[23] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 3);
        buf0[30] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 3);
        buf0[39] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 3);
        buf0[15] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 3);
        buf0[29] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 3);
        buf0[47] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 3);
        buf0[7] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 3);
        buf0[28] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 3);
        buf0[55] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 3);
        buf0[27] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 7, k + 3);
        buf0[63] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 7, k + 3);
        buf0[26] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 7, k + 3);
      }
      {
        buf0[31] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8, k + 3);
        buf0[30] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8, k + 3);
        buf0[29] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 8, k + 3);
        buf0[28] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 8, k + 3);
        buf0[27] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 8, k + 3);
      }
      {
        buf0[31] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9, k + 3);
        buf0[30] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 9, k + 3);
        buf0[29] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 9, k + 3);
        buf0[28] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 9, k + 3);
      }
      {
        buf0[31] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 10, k + 3);
        buf0[30] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 10, k + 3);
        buf0[29] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 10, k + 3);
      }
      {
        buf0[31] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 11, k + 3);
        buf0[30] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 11, k + 3);
      }
      {
        buf0[31] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 12, k + 3);
      }
      {
        buf0[32] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -5, k + 4);
      }
      {
        buf0[32] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -4, k + 4);
        buf0[33] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -4, k + 4);
      }
      {
        buf0[32] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -3, k + 4);
        buf0[33] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -3, k + 4);
        buf0[34] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -3, k + 4);
      }
      {
        buf0[32] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2, k + 4);
        buf0[33] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -2, k + 4);
        buf0[34] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -2, k + 4);
        buf0[35] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -2, k + 4);
      }
      {
        buf0[32] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1, k + 4);
        buf0[33] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1, k + 4);
        buf0[34] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -1, k + 4);
        buf0[35] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -1, k + 4);
        buf0[36] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -1, k + 4);
      }
      {
        buf0[32] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j, k + 4);
        buf0[32] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j, k + 4);
        buf0[32] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j, k + 4);
        buf0[32] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j, k + 4);
        buf0[32] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j, k + 4);
        buf0[32] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j, k + 4);
        buf0[32] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j, k + 4);
        buf0[32] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j, k + 4);
        buf0[32] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j, k + 4);
        buf0[32] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j, k + 4);
        buf0[32] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j, k + 4);
        buf0[24] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 4);
        buf0[40] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 4);
        buf0[33] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 4);
        buf0[16] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 4);
        buf0[48] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 4);
        buf0[34] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 4);
        buf0[8] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 4);
        buf0[56] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 4);
        buf0[35] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 4);
        buf0[0] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j, k + 4);
        buf0[36] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j, k + 4);
        buf0[37] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j, k + 4);
      }
      {
        buf0[33] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1, k + 4);
        buf0[33] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1, k + 4);
        buf0[33] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1, k + 4);
        buf0[33] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1, k + 4);
        buf0[33] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1, k + 4);
        buf0[33] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 1, k + 4);
        buf0[33] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 1, k + 4);
        buf0[33] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 1, k + 4);
        buf0[33] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 1, k + 4);
        buf0[33] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 1, k + 4);
        buf0[33] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 1, k + 4);
        buf0[25] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 4);
        buf0[32] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 4);
        buf0[41] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 4);
        buf0[34] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 4);
        buf0[17] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 4);
        buf0[49] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 4);
        buf0[35] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 4);
        buf0[9] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 4);
        buf0[57] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 4);
        buf0[36] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 4);
        buf0[1] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 1, k + 4);
        buf0[37] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 1, k + 4);
        buf0[38] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 1, k + 4);
      }
      {
        buf0[34] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 2, k + 4);
        buf0[34] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 2, k + 4);
        buf0[34] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 2, k + 4);
        buf0[34] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 2, k + 4);
        buf0[34] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 2, k + 4);
        buf0[34] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 2, k + 4);
        buf0[34] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 2, k + 4);
        buf0[34] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 2, k + 4);
        buf0[34] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 2, k + 4);
        buf0[34] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 2, k + 4);
        buf0[34] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 2, k + 4);
        buf0[26] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 4);
        buf0[33] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 4);
        buf0[42] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 4);
        buf0[35] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 4);
        buf0[18] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 4);
        buf0[32] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 4);
        buf0[50] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 4);
        buf0[36] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 4);
        buf0[10] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 4);
        buf0[58] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 4);
        buf0[37] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 4);
        buf0[2] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 2, k + 4);
        buf0[38] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 2, k + 4);
        buf0[39] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 2, k + 4);
      }
      {
        buf0[35] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 3, k + 4);
        buf0[35] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 3, k + 4);
        buf0[35] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 3, k + 4);
        buf0[35] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 3, k + 4);
        buf0[35] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 3, k + 4);
        buf0[35] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 3, k + 4);
        buf0[35] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 3, k + 4);
        buf0[35] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 3, k + 4);
        buf0[35] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 3, k + 4);
        buf0[35] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 3, k + 4);
        buf0[35] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 3, k + 4);
        buf0[27] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 4);
        buf0[34] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 4);
        buf0[43] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 4);
        buf0[36] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 4);
        buf0[19] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 4);
        buf0[33] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 4);
        buf0[51] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 4);
        buf0[37] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 4);
        buf0[11] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 4);
        buf0[32] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 4);
        buf0[59] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 4);
        buf0[38] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 4);
        buf0[3] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 3, k + 4);
        buf0[39] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 3, k + 4);
      }
      {
        buf0[36] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 4, k + 4);
        buf0[36] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 4, k + 4);
        buf0[36] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 4, k + 4);
        buf0[36] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 4, k + 4);
        buf0[36] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 4, k + 4);
        buf0[36] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 4, k + 4);
        buf0[36] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 4, k + 4);
        buf0[36] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 4, k + 4);
        buf0[36] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 4, k + 4);
        buf0[36] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 4, k + 4);
        buf0[36] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 4, k + 4);
        buf0[28] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 4);
        buf0[35] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 4);
        buf0[44] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 4);
        buf0[37] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 4);
        buf0[20] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 4);
        buf0[34] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 4);
        buf0[52] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 4);
        buf0[38] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 4);
        buf0[12] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 4);
        buf0[33] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 4);
        buf0[60] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 4);
        buf0[39] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 4);
        buf0[4] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 4, k + 4);
        buf0[32] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 4, k + 4);
      }
      {
        buf0[37] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 5, k + 4);
        buf0[37] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 5, k + 4);
        buf0[37] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 5, k + 4);
        buf0[37] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 5, k + 4);
        buf0[37] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 5, k + 4);
        buf0[37] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 5, k + 4);
        buf0[37] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 5, k + 4);
        buf0[37] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 5, k + 4);
        buf0[37] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 5, k + 4);
        buf0[37] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 5, k + 4);
        buf0[37] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 5, k + 4);
        buf0[29] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 4);
        buf0[36] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 4);
        buf0[45] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 4);
        buf0[38] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 4);
        buf0[21] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 4);
        buf0[35] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 4);
        buf0[53] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 4);
        buf0[39] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 4);
        buf0[13] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 4);
        buf0[34] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 4);
        buf0[61] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 4);
        buf0[5] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 5, k + 4);
        buf0[33] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 5, k + 4);
        buf0[32] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 5, k + 4);
      }
      {
        buf0[38] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6, k + 4);
        buf0[38] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6, k + 4);
        buf0[38] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6, k + 4);
        buf0[38] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6, k + 4);
        buf0[38] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6, k + 4);
        buf0[38] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 6, k + 4);
        buf0[38] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 6, k + 4);
        buf0[38] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 6, k + 4);
        buf0[38] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 6, k + 4);
        buf0[38] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 6, k + 4);
        buf0[38] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 6, k + 4);
        buf0[30] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 4);
        buf0[37] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 4);
        buf0[46] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 4);
        buf0[39] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 4);
        buf0[22] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 4);
        buf0[36] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 4);
        buf0[54] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 4);
        buf0[14] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 4);
        buf0[35] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 4);
        buf0[62] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 4);
        buf0[6] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 6, k + 4);
        buf0[34] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 6, k + 4);
        buf0[33] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 6, k + 4);
      }
      {
        buf0[39] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7, k + 4);
        buf0[39] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7, k + 4);
        buf0[39] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7, k + 4);
        buf0[39] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7, k + 4);
        buf0[39] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7, k + 4);
        buf0[39] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 7, k + 4);
        buf0[39] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 7, k + 4);
        buf0[39] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 7, k + 4);
        buf0[39] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 7, k + 4);
        buf0[39] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 7, k + 4);
        buf0[39] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 7, k + 4);
        buf0[31] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 4);
        buf0[38] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 4);
        buf0[47] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 4);
        buf0[23] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 4);
        buf0[37] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 4);
        buf0[55] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 4);
        buf0[15] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 4);
        buf0[36] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 4);
        buf0[63] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 4);
        buf0[7] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 7, k + 4);
        buf0[35] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 7, k + 4);
        buf0[34] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 7, k + 4);
      }
      {
        buf0[39] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8, k + 4);
        buf0[38] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8, k + 4);
        buf0[37] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 8, k + 4);
        buf0[36] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 8, k + 4);
        buf0[35] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 8, k + 4);
      }
      {
        buf0[39] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9, k + 4);
        buf0[38] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 9, k + 4);
        buf0[37] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 9, k + 4);
        buf0[36] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 9, k + 4);
      }
      {
        buf0[39] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 10, k + 4);
        buf0[38] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 10, k + 4);
        buf0[37] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 10, k + 4);
      }
      {
        buf0[39] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 11, k + 4);
        buf0[38] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 11, k + 4);
      }
      {
        buf0[39] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 12, k + 4);
      }
      {
        buf0[40] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -5, k + 5);
      }
      {
        buf0[40] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -4, k + 5);
        buf0[41] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -4, k + 5);
      }
      {
        buf0[40] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -3, k + 5);
        buf0[41] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -3, k + 5);
        buf0[42] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -3, k + 5);
      }
      {
        buf0[40] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2, k + 5);
        buf0[41] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -2, k + 5);
        buf0[42] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -2, k + 5);
        buf0[43] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -2, k + 5);
      }
      {
        buf0[40] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1, k + 5);
        buf0[41] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1, k + 5);
        buf0[42] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -1, k + 5);
        buf0[43] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -1, k + 5);
        buf0[44] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -1, k + 5);
      }
      {
        buf0[40] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j, k + 5);
        buf0[40] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j, k + 5);
        buf0[40] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j, k + 5);
        buf0[40] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j, k + 5);
        buf0[40] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j, k + 5);
        buf0[40] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j, k + 5);
        buf0[40] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j, k + 5);
        buf0[40] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j, k + 5);
        buf0[40] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j, k + 5);
        buf0[40] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j, k + 5);
        buf0[40] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j, k + 5);
        buf0[32] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 5);
        buf0[48] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 5);
        buf0[41] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 5);
        buf0[24] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 5);
        buf0[56] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 5);
        buf0[42] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 5);
        buf0[16] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 5);
        buf0[43] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 5);
        buf0[8] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j, k + 5);
        buf0[44] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j, k + 5);
        buf0[0] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j, k + 5);
        buf0[45] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j, k + 5);
      }
      {
        buf0[41] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[41] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1, k + 5);
        buf0[41] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1, k + 5);
        buf0[41] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1, k + 5);
        buf0[41] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1, k + 5);
        buf0[41] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 1, k + 5);
        buf0[41] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 1, k + 5);
        buf0[41] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 1, k + 5);
        buf0[41] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 1, k + 5);
        buf0[41] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 1, k + 5);
        buf0[41] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 1, k + 5);
        buf0[33] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[40] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[49] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[42] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[25] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[57] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[43] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[17] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[44] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[9] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[45] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[1] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
        buf0[46] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 1, k + 5);
      }
      {
        buf0[42] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[42] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 2, k + 5);
        buf0[42] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 2, k + 5);
        buf0[42] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 2, k + 5);
        buf0[42] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 2, k + 5);
        buf0[42] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 2, k + 5);
        buf0[42] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 2, k + 5);
        buf0[42] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 2, k + 5);
        buf0[42] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 2, k + 5);
        buf0[42] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 2, k + 5);
        buf0[42] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 2, k + 5);
        buf0[34] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[41] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[50] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[43] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[26] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[40] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[58] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[44] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[18] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[45] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[10] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[46] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[2] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
        buf0[47] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 2, k + 5);
      }
      {
        buf0[43] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 3, k + 5);
        buf0[43] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 3, k + 5);
        buf0[43] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 3, k + 5);
        buf0[43] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 3, k + 5);
        buf0[43] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 3, k + 5);
        buf0[43] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 3, k + 5);
        buf0[43] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 3, k + 5);
        buf0[43] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 3, k + 5);
        buf0[43] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 3, k + 5);
        buf0[43] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 3, k + 5);
        buf0[43] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 3, k + 5);
        buf0[35] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 5);
        buf0[42] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 5);
        buf0[51] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 5);
        buf0[44] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 5);
        buf0[27] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 5);
        buf0[41] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 5);
        buf0[59] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 5);
        buf0[45] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 5);
        buf0[19] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 5);
        buf0[40] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 5);
        buf0[46] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 5);
        buf0[11] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 3, k + 5);
        buf0[47] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 3, k + 5);
        buf0[3] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 3, k + 5);
      }
      {
        buf0[44] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 4, k + 5);
        buf0[44] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 4, k + 5);
        buf0[44] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 4, k + 5);
        buf0[44] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 4, k + 5);
        buf0[44] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 4, k + 5);
        buf0[44] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 4, k + 5);
        buf0[44] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 4, k + 5);
        buf0[44] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 4, k + 5);
        buf0[44] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 4, k + 5);
        buf0[44] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 4, k + 5);
        buf0[44] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 4, k + 5);
        buf0[36] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 5);
        buf0[43] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 5);
        buf0[52] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 5);
        buf0[45] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 5);
        buf0[28] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 5);
        buf0[42] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 5);
        buf0[60] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 5);
        buf0[46] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 5);
        buf0[20] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 5);
        buf0[41] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 5);
        buf0[47] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 5);
        buf0[12] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 4, k + 5);
        buf0[40] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 4, k + 5);
        buf0[4] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 4, k + 5);
      }
      {
        buf0[45] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[45] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 5, k + 5);
        buf0[45] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 5, k + 5);
        buf0[45] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 5, k + 5);
        buf0[45] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 5, k + 5);
        buf0[45] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 5, k + 5);
        buf0[45] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 5, k + 5);
        buf0[45] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 5, k + 5);
        buf0[45] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 5, k + 5);
        buf0[45] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 5, k + 5);
        buf0[45] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 5, k + 5);
        buf0[37] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[44] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[53] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[46] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[29] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[43] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[61] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[47] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[21] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[42] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[13] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[41] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[5] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
        buf0[40] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 5, k + 5);
      }
      {
        buf0[46] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[46] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6, k + 5);
        buf0[46] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6, k + 5);
        buf0[46] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6, k + 5);
        buf0[46] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6, k + 5);
        buf0[46] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 6, k + 5);
        buf0[46] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 6, k + 5);
        buf0[46] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 6, k + 5);
        buf0[46] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 6, k + 5);
        buf0[46] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 6, k + 5);
        buf0[46] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 6, k + 5);
        buf0[38] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[45] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[54] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[47] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[30] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[44] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[62] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[22] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[43] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[14] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[42] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[6] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
        buf0[41] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 6, k + 5);
      }
      {
        buf0[47] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
        buf0[47] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7, k + 5);
        buf0[47] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7, k + 5);
        buf0[47] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7, k + 5);
        buf0[47] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7, k + 5);
        buf0[47] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 7, k + 5);
        buf0[47] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 7, k + 5);
        buf0[47] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 7, k + 5);
        buf0[47] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 7, k + 5);
        buf0[47] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 7, k + 5);
        buf0[47] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 7, k + 5);
        buf0[39] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
        buf0[46] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
        buf0[55] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
        buf0[31] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
        buf0[45] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
        buf0[63] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
        buf0[23] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
        buf0[44] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
        buf0[15] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
        buf0[43] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
        buf0[7] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
        buf0[42] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 7, k + 5);
      }
      {
        buf0[47] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8, k + 5);
        buf0[46] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8, k + 5);
        buf0[45] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 8, k + 5);
        buf0[44] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 8, k + 5);
        buf0[43] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 8, k + 5);
      }
      {
        buf0[47] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9, k + 5);
        buf0[46] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 9, k + 5);
        buf0[45] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 9, k + 5);
        buf0[44] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 9, k + 5);
      }
      {
        buf0[47] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 10, k + 5);
        buf0[46] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 10, k + 5);
        buf0[45] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 10, k + 5);
      }
      {
        buf0[47] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 11, k + 5);
        buf0[46] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 11, k + 5);
      }
      {
        buf0[47] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 12, k + 5);
      }
      {
        buf0[48] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -5, k + 6);
      }
      {
        buf0[48] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -4, k + 6);
        buf0[49] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -4, k + 6);
      }
      {
        buf0[48] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -3, k + 6);
        buf0[49] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -3, k + 6);
        buf0[50] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -3, k + 6);
      }
      {
        buf0[48] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2, k + 6);
        buf0[49] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -2, k + 6);
        buf0[50] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -2, k + 6);
        buf0[51] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -2, k + 6);
      }
      {
        buf0[48] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1, k + 6);
        buf0[49] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1, k + 6);
        buf0[50] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -1, k + 6);
        buf0[51] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -1, k + 6);
        buf0[52] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -1, k + 6);
      }
      {
        buf0[48] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[48] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j, k + 6);
        buf0[48] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j, k + 6);
        buf0[48] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j, k + 6);
        buf0[48] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j, k + 6);
        buf0[48] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j, k + 6);
        buf0[48] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j, k + 6);
        buf0[48] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j, k + 6);
        buf0[48] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j, k + 6);
        buf0[48] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j, k + 6);
        buf0[48] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j, k + 6);
        buf0[40] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[56] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[49] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[32] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[50] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[24] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[51] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[16] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[52] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[8] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j, k + 6);
        buf0[53] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j, k + 6);
      }
      {
        buf0[49] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[49] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1, k + 6);
        buf0[49] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1, k + 6);
        buf0[49] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1, k + 6);
        buf0[49] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1, k + 6);
        buf0[49] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 1, k + 6);
        buf0[49] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 1, k + 6);
        buf0[49] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 1, k + 6);
        buf0[49] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 1, k + 6);
        buf0[49] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 1, k + 6);
        buf0[49] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 1, k + 6);
        buf0[41] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[48] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[57] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[50] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[33] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[51] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[25] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[52] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[17] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[53] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[9] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
        buf0[54] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 1, k + 6);
      }
      {
        buf0[50] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[50] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 2, k + 6);
        buf0[50] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 2, k + 6);
        buf0[50] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 2, k + 6);
        buf0[50] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 2, k + 6);
        buf0[50] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 2, k + 6);
        buf0[50] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 2, k + 6);
        buf0[50] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 2, k + 6);
        buf0[50] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 2, k + 6);
        buf0[50] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 2, k + 6);
        buf0[50] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 2, k + 6);
        buf0[42] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[49] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[58] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[51] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[34] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[48] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[52] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[26] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[53] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[18] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[54] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[10] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
        buf0[55] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 2, k + 6);
      }
      {
        buf0[51] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 3, k + 6);
        buf0[51] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 3, k + 6);
        buf0[51] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 3, k + 6);
        buf0[51] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 3, k + 6);
        buf0[51] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 3, k + 6);
        buf0[51] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 3, k + 6);
        buf0[51] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 3, k + 6);
        buf0[51] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 3, k + 6);
        buf0[51] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 3, k + 6);
        buf0[51] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 3, k + 6);
        buf0[51] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 3, k + 6);
        buf0[43] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 6);
        buf0[50] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 6);
        buf0[59] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 6);
        buf0[52] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 6);
        buf0[35] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 6);
        buf0[49] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 6);
        buf0[53] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 6);
        buf0[27] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 6);
        buf0[48] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 6);
        buf0[54] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 6);
        buf0[19] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 3, k + 6);
        buf0[55] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 3, k + 6);
        buf0[11] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 3, k + 6);
      }
      {
        buf0[52] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 4, k + 6);
        buf0[52] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 4, k + 6);
        buf0[52] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 4, k + 6);
        buf0[52] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 4, k + 6);
        buf0[52] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 4, k + 6);
        buf0[52] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 4, k + 6);
        buf0[52] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 4, k + 6);
        buf0[52] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 4, k + 6);
        buf0[52] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 4, k + 6);
        buf0[52] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 4, k + 6);
        buf0[52] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 4, k + 6);
        buf0[44] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 6);
        buf0[51] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 6);
        buf0[60] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 6);
        buf0[53] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 6);
        buf0[36] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 6);
        buf0[50] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 6);
        buf0[54] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 6);
        buf0[28] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 6);
        buf0[49] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 6);
        buf0[55] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 6);
        buf0[20] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 4, k + 6);
        buf0[48] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 4, k + 6);
        buf0[12] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 4, k + 6);
      }
      {
        buf0[53] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[53] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 5, k + 6);
        buf0[53] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 5, k + 6);
        buf0[53] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 5, k + 6);
        buf0[53] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 5, k + 6);
        buf0[53] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 5, k + 6);
        buf0[53] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 5, k + 6);
        buf0[53] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 5, k + 6);
        buf0[53] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 5, k + 6);
        buf0[53] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 5, k + 6);
        buf0[53] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 5, k + 6);
        buf0[45] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[52] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[61] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[54] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[37] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[51] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[55] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[29] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[50] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[21] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[49] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[13] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
        buf0[48] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 5, k + 6);
      }
      {
        buf0[54] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[54] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6, k + 6);
        buf0[54] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6, k + 6);
        buf0[54] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6, k + 6);
        buf0[54] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6, k + 6);
        buf0[54] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 6, k + 6);
        buf0[54] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 6, k + 6);
        buf0[54] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 6, k + 6);
        buf0[54] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 6, k + 6);
        buf0[54] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 6, k + 6);
        buf0[54] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 6, k + 6);
        buf0[46] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[53] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[62] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[55] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[38] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[52] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[30] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[51] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[22] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[50] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[14] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
        buf0[49] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 6, k + 6);
      }
      {
        buf0[55] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[55] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7, k + 6);
        buf0[55] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7, k + 6);
        buf0[55] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7, k + 6);
        buf0[55] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7, k + 6);
        buf0[55] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 7, k + 6);
        buf0[55] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 7, k + 6);
        buf0[55] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 7, k + 6);
        buf0[55] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 7, k + 6);
        buf0[55] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 7, k + 6);
        buf0[55] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 7, k + 6);
        buf0[47] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[54] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[63] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[39] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[53] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[31] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[52] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[23] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[51] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[15] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
        buf0[50] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 7, k + 6);
      }
      {
        buf0[55] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8, k + 6);
        buf0[54] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8, k + 6);
        buf0[53] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 8, k + 6);
        buf0[52] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 8, k + 6);
        buf0[51] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 8, k + 6);
      }
      {
        buf0[55] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9, k + 6);
        buf0[54] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 9, k + 6);
        buf0[53] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 9, k + 6);
        buf0[52] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 9, k + 6);
      }
      {
        buf0[55] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 10, k + 6);
        buf0[54] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 10, k + 6);
        buf0[53] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 10, k + 6);
      }
      {
        buf0[55] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 11, k + 6);
        buf0[54] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 11, k + 6);
      }
      {
        buf0[55] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 12, k + 6);
      }
      {
        buf0[56] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -5, k + 7);
      }
      {
        buf0[56] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -4, k + 7);
        buf0[57] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -4, k + 7);
      }
      {
        buf0[56] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -3, k + 7);
        buf0[57] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -3, k + 7);
        buf0[58] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -3, k + 7);
      }
      {
        buf0[56] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2, k + 7);
        buf0[57] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -2, k + 7);
        buf0[58] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -2, k + 7);
        buf0[59] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -2, k + 7);
      }
      {
        buf0[56] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1, k + 7);
        buf0[57] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1, k + 7);
        buf0[58] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + -1, k + 7);
        buf0[59] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + -1, k + 7);
        buf0[60] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + -1, k + 7);
      }
      {
        buf0[56] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j, k + 7);
        buf0[56] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j, k + 7);
        buf0[56] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j, k + 7);
        buf0[56] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j, k + 7);
        buf0[56] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j, k + 7);
        buf0[56] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j, k + 7);
        buf0[56] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j, k + 7);
        buf0[56] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j, k + 7);
        buf0[56] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j, k + 7);
        buf0[56] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j, k + 7);
        buf0[56] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j, k + 7);
        buf0[48] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 7);
        buf0[57] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j, k + 7);
        buf0[40] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 7);
        buf0[58] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j, k + 7);
        buf0[32] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 7);
        buf0[59] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j, k + 7);
        buf0[24] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j, k + 7);
        buf0[60] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j, k + 7);
        buf0[16] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j, k + 7);
        buf0[61] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j, k + 7);
      }
      {
        buf0[57] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[57] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1, k + 7);
        buf0[57] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1, k + 7);
        buf0[57] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1, k + 7);
        buf0[57] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1, k + 7);
        buf0[57] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 1, k + 7);
        buf0[57] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 1, k + 7);
        buf0[57] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 1, k + 7);
        buf0[57] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 1, k + 7);
        buf0[57] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 1, k + 7);
        buf0[57] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 1, k + 7);
        buf0[49] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[56] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[58] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[41] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[59] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[33] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[60] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[25] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[61] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[17] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
        buf0[62] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 1, k + 7);
      }
      {
        buf0[58] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
        buf0[58] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 2, k + 7);
        buf0[58] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 2, k + 7);
        buf0[58] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 2, k + 7);
        buf0[58] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 2, k + 7);
        buf0[58] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 2, k + 7);
        buf0[58] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 2, k + 7);
        buf0[58] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 2, k + 7);
        buf0[58] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 2, k + 7);
        buf0[58] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 2, k + 7);
        buf0[58] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 2, k + 7);
        buf0[50] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
        buf0[57] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
        buf0[59] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
        buf0[42] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
        buf0[56] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
        buf0[60] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
        buf0[34] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
        buf0[61] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
        buf0[26] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
        buf0[62] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
        buf0[18] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
        buf0[63] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 2, k + 7);
      }
      {
        buf0[59] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 3, k + 7);
        buf0[59] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 3, k + 7);
        buf0[59] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 3, k + 7);
        buf0[59] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 3, k + 7);
        buf0[59] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 3, k + 7);
        buf0[59] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 3, k + 7);
        buf0[59] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 3, k + 7);
        buf0[59] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 3, k + 7);
        buf0[59] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 3, k + 7);
        buf0[59] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 3, k + 7);
        buf0[59] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 3, k + 7);
        buf0[51] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 7);
        buf0[58] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 7);
        buf0[60] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 3, k + 7);
        buf0[43] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 7);
        buf0[57] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 7);
        buf0[61] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 3, k + 7);
        buf0[35] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 7);
        buf0[56] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 7);
        buf0[62] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 3, k + 7);
        buf0[27] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 3, k + 7);
        buf0[63] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 3, k + 7);
        buf0[19] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 3, k + 7);
      }
      {
        buf0[60] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 4, k + 7);
        buf0[60] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 4, k + 7);
        buf0[60] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 4, k + 7);
        buf0[60] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 4, k + 7);
        buf0[60] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 4, k + 7);
        buf0[60] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 4, k + 7);
        buf0[60] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 4, k + 7);
        buf0[60] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 4, k + 7);
        buf0[60] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 4, k + 7);
        buf0[60] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 4, k + 7);
        buf0[60] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 4, k + 7);
        buf0[52] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 7);
        buf0[59] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 7);
        buf0[61] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 4, k + 7);
        buf0[44] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 7);
        buf0[58] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 7);
        buf0[62] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 4, k + 7);
        buf0[36] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 7);
        buf0[57] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 7);
        buf0[63] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 4, k + 7);
        buf0[28] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 4, k + 7);
        buf0[56] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 4, k + 7);
        buf0[20] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 4, k + 7);
      }
      {
        buf0[61] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[61] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 5, k + 7);
        buf0[61] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 5, k + 7);
        buf0[61] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 5, k + 7);
        buf0[61] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 5, k + 7);
        buf0[61] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 5, k + 7);
        buf0[61] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 5, k + 7);
        buf0[61] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 5, k + 7);
        buf0[61] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 5, k + 7);
        buf0[61] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 5, k + 7);
        buf0[61] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 5, k + 7);
        buf0[53] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[60] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[62] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[45] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[59] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[63] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[37] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[58] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[29] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[57] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[21] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
        buf0[56] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 5, k + 7);
      }
      {
        buf0[62] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[62] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6, k + 7);
        buf0[62] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6, k + 7);
        buf0[62] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6, k + 7);
        buf0[62] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6, k + 7);
        buf0[62] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 6, k + 7);
        buf0[62] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 6, k + 7);
        buf0[62] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 6, k + 7);
        buf0[62] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 6, k + 7);
        buf0[62] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 6, k + 7);
        buf0[62] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 6, k + 7);
        buf0[54] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[61] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[63] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[46] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[60] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[38] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[59] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[30] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[58] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[22] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
        buf0[57] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 6, k + 7);
      }
      {
        buf0[63] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[63] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7, k + 7);
        buf0[63] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7, k + 7);
        buf0[63] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7, k + 7);
        buf0[63] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7, k + 7);
        buf0[63] += dev_coeff[3] * bIn(i + hipThreadIdx_x + 3, j + 7, k + 7);
        buf0[63] += dev_coeff[3] * bIn(i + hipThreadIdx_x + -3, j + 7, k + 7);
        buf0[63] += dev_coeff[4] * bIn(i + hipThreadIdx_x + 4, j + 7, k + 7);
        buf0[63] += dev_coeff[4] * bIn(i + hipThreadIdx_x + -4, j + 7, k + 7);
        buf0[63] += dev_coeff[5] * bIn(i + hipThreadIdx_x + 5, j + 7, k + 7);
        buf0[63] += dev_coeff[5] * bIn(i + hipThreadIdx_x + -5, j + 7, k + 7);
        buf0[55] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[62] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[47] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[61] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[39] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[60] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[31] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[59] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[23] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
        buf0[58] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 7, k + 7);
      }
      {
        buf0[63] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8, k + 7);
        buf0[62] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8, k + 7);
        buf0[61] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 8, k + 7);
        buf0[60] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 8, k + 7);
        buf0[59] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 8, k + 7);
      }
      {
        buf0[63] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9, k + 7);
        buf0[62] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 9, k + 7);
        buf0[61] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 9, k + 7);
        buf0[60] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 9, k + 7);
      }
      {
        buf0[63] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + 10, k + 7);
        buf0[62] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 10, k + 7);
        buf0[61] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 10, k + 7);
      }
      {
        buf0[63] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + 11, k + 7);
        buf0[62] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 11, k + 7);
      }
      {
        buf0[63] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + 12, k + 7);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[56 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf0[48 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf0[40 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf0[32 + rel] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
            buf0[24 + rel] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 8);
          }
          _cg_rel1 += 1;
        }
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[56 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 9);
            buf0[48 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 9);
            buf0[40 + rel] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 9);
            buf0[32 + rel] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 9);
          }
          _cg_rel1 += 1;
        }
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[56 + rel] += dev_coeff[3] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 10);
            buf0[48 + rel] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 10);
            buf0[40 + rel] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 10);
          }
          _cg_rel1 += 1;
        }
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[56 + rel] += dev_coeff[4] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 11);
            buf0[48 + rel] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 11);
          }
          _cg_rel1 += 1;
        }
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[56 + rel] += dev_coeff[5] * bIn(i + hipThreadIdx_x, j + _cg_idx1, k + 12);
          }
          _cg_rel1 += 1;
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
            bOut(i + _cg_idx0, j + _cg_idx1, k + _cg_idx2) = buf0[rel];
          }
        }
      }
    }
  }
}
# 134 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/s3d/intermediate_gen/s3d.cu" 2

}
#undef bIn
#undef bOut
