# 1 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/s2d/intermediate_gen/s2d.cu"
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
# 1 "VSBrick-laplacian2d2.py-HIP-8x8-8x8" 1
{
  auto *binfo = bOut.bInfo;
  long neighbor0 = binfo->adj[b][0];
  long neighbor1 = binfo->adj[b][1];
  long neighbor2 = binfo->adj[b][2];
  long neighbor3 = binfo->adj[b][3];
  long neighbor4 = b;
  long neighbor5 = binfo->adj[b][5];
  long neighbor6 = binfo->adj[b][6];
  long neighbor7 = binfo->adj[b][7];
  long neighbor8 = binfo->adj[b][8];
  bElem buf0[1];
  {
    {
      {
        // New offset [0, 0]
        buf0[0] = 0;
      }
    }
    {
      bElem _cg_bIn00_vecbuf;
      bElem _cg_bIn_10_vecbuf;
      {
        // New offset [0, -2]
        bElem _cg_bIn00_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor4 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor1 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 6 -> _cg_bIn00_vecbuf
          dev_shl(_cg_bIn00_vecbuf, _cg_vectmp1, _cg_vectmp0, 16, 64, hipThreadIdx_x);
          _cg_bIn00_reg = _cg_bIn00_vecbuf;
        }
        buf0[0] += dev_coeff[2] * _cg_bIn00_reg;
      }
      {
        // New offset [0, -1]
        bElem _cg_bIn00_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor4 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor1 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_bIn00_vecbuf
          dev_shl(_cg_bIn00_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_bIn00_reg = _cg_bIn00_vecbuf;
        }
        buf0[0] += dev_coeff[1] * _cg_bIn00_reg;
      }
      {
        // New offset [-2, 0]
        bElem _cg_bIn00_reg;
        {
          _cg_bIn_10_vecbuf = bIn.dat[neighbor3 * bIn.step + hipThreadIdx_x];
          _cg_bIn00_vecbuf = bIn.dat[neighbor4 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_10_vecbuf ,_cg_bIn00_vecbuf, 6 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_10_vecbuf, _cg_bIn00_vecbuf, 2, 8, hipThreadIdx_x & 7);
          _cg_bIn00_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[2] * _cg_bIn00_reg;
      }
      {
        // New offset [-1, 0]
        bElem _cg_bIn00_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_10_vecbuf ,_cg_bIn00_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_10_vecbuf, _cg_bIn00_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_bIn00_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[1] * _cg_bIn00_reg;
      }
      {
        // New offset [0, 0]
        bElem _cg_bIn00_reg;
        {
          _cg_bIn00_reg = _cg_bIn00_vecbuf;
        }
        buf0[0] += dev_coeff[0] * _cg_bIn00_reg;
      }
      {
        // New offset [1, 0]
        bElem _cg_bIn00_reg;
        {
          _cg_bIn_10_vecbuf = _cg_bIn00_vecbuf;
          _cg_bIn00_vecbuf = bIn.dat[neighbor5 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_10_vecbuf ,_cg_bIn00_vecbuf, 1 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_10_vecbuf, _cg_bIn00_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_bIn00_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[1] * _cg_bIn00_reg;
      }
      {
        // New offset [2, 0]
        bElem _cg_bIn00_reg;
        {
          bElem _cg_vectmp0;
          // merge0 _cg_bIn_10_vecbuf ,_cg_bIn00_vecbuf, 2 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_bIn_10_vecbuf, _cg_bIn00_vecbuf, 6, 8, hipThreadIdx_x & 7);
          _cg_bIn00_reg = _cg_vectmp0;
        }
        buf0[0] += dev_coeff[2] * _cg_bIn00_reg;
      }
      {
        // New offset [0, 1]
        bElem _cg_bIn00_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor7 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor4 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_bIn00_vecbuf
          dev_shl(_cg_bIn00_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_bIn00_reg = _cg_bIn00_vecbuf;
        }
        buf0[0] += dev_coeff[1] * _cg_bIn00_reg;
      }
      {
        // New offset [0, 2]
        bElem _cg_bIn00_reg;
        {
          bElem _cg_vectmp0;
          _cg_vectmp0 = bIn.dat[neighbor7 * bIn.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = bIn.dat[neighbor4 * bIn.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 2 -> _cg_bIn00_vecbuf
          dev_shl(_cg_bIn00_vecbuf, _cg_vectmp1, _cg_vectmp0, 48, 64, hipThreadIdx_x);
          _cg_bIn00_reg = _cg_bIn00_vecbuf;
        }
        buf0[0] += dev_coeff[2] * _cg_bIn00_reg;
      }
    }
    bElem *bOut_ref = &bOut.dat[neighbor4 * bOut.step];
    for (long sti = 0; sti < 1; ++sti)
    {
      bOut_ref[sti * 64 + hipThreadIdx_x] = buf0[sti];
    }
  }
}
# 38 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/s2d/intermediate_gen/s2d.cu" 2

}
#define bIn(a, b) arr_in[c][b]
#define bOut(a, b) arr_out[c][b]
__global__ void laplacian_codegen(bElem (*arr_in)[STRIDE0], bElem (*arr_out)[STRIDE0], bElem *dev_coeff) {
    long j = OFF1 + (blockIdx.y * TILE1);
    long i = OFF0 + (blockIdx.x * VECSIZE);
# 1 "VSTile-laplacian2d2.py-HIP-8x64" 1
{
  bElem buf0[8];
  {
    {
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[0 + rel] = 0;
          }
          _cg_rel1 += 1;
        }
      }
    }
    {
      {
        buf0[0] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -2);
      }
      {
        buf0[0] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + -1);
        buf0[1] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + -1);
      }
      {
        buf0[0] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j);
        buf0[0] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j);
        buf0[0] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j);
        buf0[0] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j);
        buf0[0] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j);
        buf0[1] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j);
        buf0[2] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j);
      }
      {
        buf0[1] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 1);
        buf0[1] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 1);
        buf0[1] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 1);
        buf0[1] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 1);
        buf0[1] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 1);
        buf0[0] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1);
        buf0[2] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 1);
        buf0[3] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 1);
      }
      {
        long _cg_rel1 = 0;
        for (long _cg_idx1 = 0; _cg_idx1 < 4; _cg_idx1 += 1)
        {
          long rel = _cg_rel1;
          {
            buf0[2 + rel] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2);
            buf0[2 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + _cg_idx1 + 2);
            buf0[2 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + _cg_idx1 + 2);
            buf0[2 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + _cg_idx1 + 2);
            buf0[2 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + _cg_idx1 + 2);
            buf0[1 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2);
            buf0[3 + rel] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2);
            buf0[0 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2);
            buf0[4 + rel] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + _cg_idx1 + 2);
          }
          _cg_rel1 += 1;
        }
      }
      {
        buf0[6] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 6);
        buf0[6] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 6);
        buf0[6] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 6);
        buf0[6] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 6);
        buf0[6] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 6);
        buf0[5] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6);
        buf0[7] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 6);
        buf0[4] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 6);
      }
      {
        buf0[7] += dev_coeff[0] * bIn(i + hipThreadIdx_x, j + 7);
        buf0[7] += dev_coeff[1] * bIn(i + hipThreadIdx_x + 1, j + 7);
        buf0[7] += dev_coeff[1] * bIn(i + hipThreadIdx_x + -1, j + 7);
        buf0[7] += dev_coeff[2] * bIn(i + hipThreadIdx_x + 2, j + 7);
        buf0[7] += dev_coeff[2] * bIn(i + hipThreadIdx_x + -2, j + 7);
        buf0[6] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 7);
        buf0[5] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 7);
      }
      {
        buf0[7] += dev_coeff[1] * bIn(i + hipThreadIdx_x, j + 8);
        buf0[6] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 8);
      }
      {
        buf0[7] += dev_coeff[2] * bIn(i + hipThreadIdx_x, j + 9);
      }
    }
    {
      long rel = 0;
      for (long _cg_idx1 = 0; _cg_idx1 < 8; _cg_idx1 += 1)
      {
        for (long _cg_idx0 = hipThreadIdx_x; _cg_idx0 < 64; _cg_idx0 += 64, ++rel)
        {
          bOut(i + _cg_idx0, j + _cg_idx1) = buf0[rel];
        }
      }
    }
  }
}
# 45 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/s2d/intermediate_gen/s2d.cu" 2

}
#undef bIn
#undef bOut
