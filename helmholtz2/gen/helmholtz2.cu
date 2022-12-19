# 1 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/helmholtz2/intermediate_gen/helmholtz2.cu"
#include <omp.h>
#include "vecscatter.h"
#include "brick.h"
#include "../../../gen/consts.h"
#include "./helmholtz2.h"
#include <brick-hip.h>

__global__ void helmholtz2_naive2(bElem (*x)[STRIDE1][STRIDE0], bElem (*alpha)[STRIDE1][STRIDE0], bElem (*beta_i)[STRIDE1][STRIDE0], bElem (*beta_j)[STRIDE1][STRIDE0], bElem (*beta_k)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0], bElem *c) {
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;
    bElem c1 = c[0];
    bElem c2 = c[1];
    bElem h2inv = c[2];
    out[k][j][i] = c1 * alpha[k][j][i] * x[k][j][i] - 
        c2 * h2inv * (
            beta_i[k][j][i + 1] * (x[k][j][i + 1] - x[k][j][i]) + 
            beta_i[k][j][i]     * (x[k][j][i - 1] - x[k][j][i]) +
            beta_j[k][j + 1][i] * (x[k][j + 1][i] - x[k][j][i]) +
            beta_j[k][j - 1][i] * (x[k][j - 1][i] - x[k][j][i]) +
            beta_k[k + 1][j][i] * (x[k + 1][j][i] - x[k][j][i]) +
            beta_k[k - 1][j][i] * (x[k - 1][j][i] - x[k][j][i])
        );
}
#define x(a, b, c) x_arr[c][b][a]
#define alpha(a, b, c) alpha_arr[c][b][a]
#define beta_i(a, b, c) beta_i_arr[c][b][a]
#define beta_j(a, b, c) beta_j_arr[c][b][a]
#define beta_k(a, b, c) beta_k_arr[c][b][a] 
#define out(a, b, c) out_arr[c][b][a]
__global__ void helmholtz2_codegen2(bElem (*x_arr)[STRIDE1][STRIDE0], bElem (*alpha_arr)[STRIDE1][STRIDE0], bElem (*beta_i_arr)[STRIDE1][STRIDE0], bElem (*beta_j_arr)[STRIDE1][STRIDE0], bElem (*beta_k_arr)[STRIDE1][STRIDE0], bElem (*out_arr)[STRIDE1][STRIDE0], bElem *c) {
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;
# 1 "VSTile-helmholtz22.py-HIP-8x8x64" 1
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
                buf0[0 + rel] = c[0] * alpha(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) - c[1] * c[2] * (beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2) - beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) + beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + _cg_idx2) - beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) + beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2) - beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) + beta_j(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1 + -1, k + _cg_idx2) - beta_j(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) + beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) + beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + -1) - beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2));
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
            out(i + _cg_idx0, j + _cg_idx1, k + _cg_idx2) = buf0[rel];
          }
        }
      }
    }
  }
}
# 35 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/helmholtz2/intermediate_gen/helmholtz2.cu" 2

}
#undef x
#undef alpha
#undef beta_i
#undef beta_j
#undef beta_k
#undef out
__global__ void helmholtz2_naive_bricks2(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType x, BType alpha, BType beta_i, BType beta_j, BType beta_k, BType out, bElem *c) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;
    bElem c1 = c[0];
    bElem c2 = c[1];
    bElem h2inv = c[2];
    out[b][k][j][i] = c1 * alpha[b][k][j][i] * x[b][k][j][i] - 
        c2 * h2inv * (
            beta_i[b][k][j][i + 1] * (x[b][k][j][i + 1] - x[b][k][j][i]) + 
            beta_i[b][k][j][i]     * (x[b][k][j][i - 1] - x[b][k][j][i]) +
            beta_j[b][k][j + 1][i] * (x[b][k][j + 1][i] - x[b][k][j][i]) +
            beta_j[b][k][j - 1][i] * (x[b][k][j - 1][i] - x[b][k][j][i]) +
            beta_k[b][k + 1][j][i] * (x[b][k + 1][j][i] - x[b][k][j][i]) +
            beta_k[b][k - 1][j][i] * (x[b][k - 1][j][i] - x[b][k][j][i])
        );
}
__global__ void helmholtz2_codegen_bricks2(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType x, BType alpha, BType beta_i, BType beta_j, BType beta_k, BType out, bElem *c) {
  unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
# 1 "VSBrick-helmholtz22.py-HIP-8x8x8-8x8" 1
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
      bElem _cg_beta_i000_vecbuf;
      bElem _cg_beta_i100_vecbuf;
      bElem _cg_x000_vecbuf;
      bElem _cg_x100_vecbuf;
      bElem _cg_x_100_vecbuf;
      bElem _cg_beta_j000_vecbuf;
      bElem _cg_beta_j010_vecbuf;
      bElem _cg_beta_k000_vecbuf;
      bElem _cg_beta_k001_vecbuf;
      {
        // New offset [-1, 0, -1]
        bElem _cg_beta_k101_reg;
        bElem _cg_x100_reg;
        {
          _cg_beta_k001_vecbuf = beta_k.dat[neighbor13 * beta_k.step + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          _cg_beta_k101_reg = _cg_beta_k001_vecbuf;
          _cg_x100_reg = _cg_x000_vecbuf;
        }
        buf0[0] += _cg_beta_k101_reg * _cg_x100_reg;
      }
      {
        // New offset [0, -1, 0]
        bElem _cg_beta_j010_reg;
        bElem _cg_x000_reg;
        {
          _cg_beta_j010_vecbuf = beta_j.dat[neighbor13 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x000_vecbuf
          dev_shl(_cg_x000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_beta_j010_reg = _cg_beta_j010_vecbuf;
          _cg_x000_reg = _cg_x000_vecbuf;
        }
        buf0[0] += _cg_beta_j010_reg * _cg_x000_reg;
      }
      {
        // New offset [-1, 0, 0]
        bElem _cg_beta_i100_reg;
        bElem _cg_x000_reg;
        bElem _cg_x100_reg;
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_k101_reg;
        bElem _cg_beta_k100_reg;
        {
          _cg_beta_j000_vecbuf = _cg_beta_j010_vecbuf;
          _cg_beta_k000_vecbuf = _cg_beta_k001_vecbuf;
          _cg_beta_i000_vecbuf = beta_i.dat[neighbor13 * beta_i.step + hipThreadIdx_x];
          _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          _cg_beta_k001_vecbuf = beta_k.dat[neighbor13 * beta_k.step + 64 + hipThreadIdx_x];
          _cg_beta_i100_reg = _cg_beta_i000_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x000_reg = _cg_vectmp0;
          _cg_x100_reg = _cg_x000_vecbuf;
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_beta_k101_reg = _cg_beta_k001_vecbuf;
          _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
        }
        buf0[0] += _cg_beta_i100_reg * _cg_x000_reg;
        buf0[0] -= _cg_beta_i100_reg * _cg_x100_reg;
        buf0[0] -= _cg_beta_j100_reg * _cg_x100_reg;
        buf0[0] -= _cg_beta_k101_reg * _cg_x100_reg;
        buf0[0] -= _cg_beta_k100_reg * _cg_x100_reg;
        buf0[1] += _cg_beta_k101_reg * _cg_x100_reg;
      }
      {
        // New offset [0, 0, 0]
        bElem _cg_beta_i100_reg;
        bElem _cg_x100_reg;
        bElem _cg_x000_reg;
        bElem _cg_beta_j010_reg;
        {
          _cg_beta_i100_vecbuf = beta_i.dat[neighbor14 * beta_i.step + hipThreadIdx_x];
          _cg_x100_vecbuf = x.dat[neighbor14 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_j.dat[neighbor16 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_j.dat[neighbor13 * beta_j.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j010_vecbuf
          dev_shl(_cg_beta_j010_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_beta_i000_vecbuf ,_cg_beta_i100_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_beta_i000_vecbuf, _cg_beta_i100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i100_reg = _cg_vectmp2;
          bElem _cg_vectmp3;
          // merge0 _cg_x000_vecbuf ,_cg_x100_vecbuf, 1 -> _cg_vectmp3
          dev_shl(_cg_vectmp3, _cg_x000_vecbuf, _cg_x100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x100_reg = _cg_vectmp3;
          _cg_x000_reg = _cg_x000_vecbuf;
          _cg_beta_j010_reg = _cg_beta_j010_vecbuf;
        }
        buf0[0] += _cg_beta_i100_reg * _cg_x100_reg;
        buf0[0] -= _cg_beta_i100_reg * _cg_x000_reg;
        buf0[0] -= _cg_beta_j010_reg * _cg_x000_reg;
      }
      {
        // New offset [-1, 1, 0]
        bElem _cg_beta_j100_reg;
        bElem _cg_x100_reg;
        {
          _cg_beta_j000_vecbuf = _cg_beta_j010_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x000_vecbuf
          dev_shl(_cg_x000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_x100_reg = _cg_x000_vecbuf;
        }
        buf0[0] += _cg_beta_j100_reg * _cg_x100_reg;
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 6; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, -1, 1]
            bElem _cg_beta_j010_reg;
            bElem _cg_x000_reg;
            {
              _cg_beta_j010_vecbuf = beta_j.dat[neighbor13 * beta_j.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor10 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x000_vecbuf
              dev_shl(_cg_x000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              _cg_beta_j010_reg = _cg_beta_j010_vecbuf;
              _cg_x000_reg = _cg_x000_vecbuf;
            }
            buf0[1 + rel] += _cg_beta_j010_reg * _cg_x000_reg;
          }
          {
            // New offset [-1, 0, 1]
            bElem _cg_beta_i100_reg;
            bElem _cg_x000_reg;
            bElem _cg_x100_reg;
            bElem _cg_beta_j100_reg;
            bElem _cg_beta_k101_reg;
            bElem _cg_beta_k100_reg;
            {
              _cg_beta_j000_vecbuf = _cg_beta_j010_vecbuf;
              _cg_beta_k000_vecbuf = _cg_beta_k001_vecbuf;
              _cg_beta_i000_vecbuf = beta_i.dat[neighbor13 * beta_i.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_x000_vecbuf = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_k001_vecbuf = beta_k.dat[neighbor13 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_i100_reg = _cg_beta_i000_vecbuf;
              bElem _cg_vectmp0;
              // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_x000_reg = _cg_vectmp0;
              _cg_x100_reg = _cg_x000_vecbuf;
              _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
              _cg_beta_k101_reg = _cg_beta_k001_vecbuf;
              _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
            }
            buf0[1 + rel] += _cg_beta_i100_reg * _cg_x000_reg;
            buf0[1 + rel] -= _cg_beta_i100_reg * _cg_x100_reg;
            buf0[1 + rel] -= _cg_beta_j100_reg * _cg_x100_reg;
            buf0[1 + rel] -= _cg_beta_k101_reg * _cg_x100_reg;
            buf0[1 + rel] -= _cg_beta_k100_reg * _cg_x100_reg;
            buf0[0 + rel] += _cg_beta_k100_reg * _cg_x100_reg;
            buf0[2 + rel] += _cg_beta_k101_reg * _cg_x100_reg;
          }
          {
            // New offset [0, 0, 1]
            bElem _cg_beta_i100_reg;
            bElem _cg_x100_reg;
            bElem _cg_x000_reg;
            bElem _cg_beta_j010_reg;
            {
              _cg_beta_i100_vecbuf = beta_i.dat[neighbor14 * beta_i.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_x100_vecbuf = x.dat[neighbor14 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              _cg_vectmp0 = beta_j.dat[neighbor16 * beta_j.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = beta_j.dat[neighbor13 * beta_j.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j010_vecbuf
              dev_shl(_cg_beta_j010_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_beta_i000_vecbuf ,_cg_beta_i100_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_beta_i000_vecbuf, _cg_beta_i100_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_i100_reg = _cg_vectmp2;
              bElem _cg_vectmp3;
              // merge0 _cg_x000_vecbuf ,_cg_x100_vecbuf, 1 -> _cg_vectmp3
              dev_shl(_cg_vectmp3, _cg_x000_vecbuf, _cg_x100_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_x100_reg = _cg_vectmp3;
              _cg_x000_reg = _cg_x000_vecbuf;
              _cg_beta_j010_reg = _cg_beta_j010_vecbuf;
            }
            buf0[1 + rel] += _cg_beta_i100_reg * _cg_x100_reg;
            buf0[1 + rel] -= _cg_beta_i100_reg * _cg_x000_reg;
            buf0[1 + rel] -= _cg_beta_j010_reg * _cg_x000_reg;
          }
          {
            // New offset [-1, 1, 1]
            bElem _cg_beta_j100_reg;
            bElem _cg_x100_reg;
            {
              _cg_beta_j000_vecbuf = _cg_beta_j010_vecbuf;
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor16 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x000_vecbuf
              dev_shl(_cg_x000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
              _cg_x100_reg = _cg_x000_vecbuf;
            }
            buf0[1 + rel] += _cg_beta_j100_reg * _cg_x100_reg;
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [0, -1, 7]
        bElem _cg_beta_j010_reg;
        bElem _cg_x000_reg;
        {
          _cg_beta_j010_vecbuf = beta_j.dat[neighbor13 * beta_j.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x000_vecbuf
          dev_shl(_cg_x000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_beta_j010_reg = _cg_beta_j010_vecbuf;
          _cg_x000_reg = _cg_x000_vecbuf;
        }
        buf0[7] += _cg_beta_j010_reg * _cg_x000_reg;
      }
      {
        // New offset [-1, 0, 7]
        bElem _cg_beta_i100_reg;
        bElem _cg_x000_reg;
        bElem _cg_x100_reg;
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_k101_reg;
        bElem _cg_beta_k100_reg;
        {
          _cg_beta_j000_vecbuf = _cg_beta_j010_vecbuf;
          _cg_beta_k000_vecbuf = _cg_beta_k001_vecbuf;
          _cg_beta_i000_vecbuf = beta_i.dat[neighbor13 * beta_i.step + 448 + hipThreadIdx_x];
          _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + 448 + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          _cg_beta_k001_vecbuf = beta_k.dat[neighbor22 * beta_k.step + hipThreadIdx_x];
          _cg_beta_i100_reg = _cg_beta_i000_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x000_reg = _cg_vectmp0;
          _cg_x100_reg = _cg_x000_vecbuf;
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_beta_k101_reg = _cg_beta_k001_vecbuf;
          _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
        }
        buf0[7] += _cg_beta_i100_reg * _cg_x000_reg;
        buf0[7] -= _cg_beta_i100_reg * _cg_x100_reg;
        buf0[7] -= _cg_beta_j100_reg * _cg_x100_reg;
        buf0[7] -= _cg_beta_k101_reg * _cg_x100_reg;
        buf0[7] -= _cg_beta_k100_reg * _cg_x100_reg;
        buf0[6] += _cg_beta_k100_reg * _cg_x100_reg;
      }
      {
        // New offset [0, 0, 7]
        bElem _cg_beta_i100_reg;
        bElem _cg_x100_reg;
        bElem _cg_x000_reg;
        bElem _cg_beta_j010_reg;
        {
          _cg_beta_i100_vecbuf = beta_i.dat[neighbor14 * beta_i.step + 448 + hipThreadIdx_x];
          _cg_x100_vecbuf = x.dat[neighbor14 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_j.dat[neighbor16 * beta_j.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_j.dat[neighbor13 * beta_j.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j010_vecbuf
          dev_shl(_cg_beta_j010_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_beta_i000_vecbuf ,_cg_beta_i100_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_beta_i000_vecbuf, _cg_beta_i100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i100_reg = _cg_vectmp2;
          bElem _cg_vectmp3;
          // merge0 _cg_x000_vecbuf ,_cg_x100_vecbuf, 1 -> _cg_vectmp3
          dev_shl(_cg_vectmp3, _cg_x000_vecbuf, _cg_x100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x100_reg = _cg_vectmp3;
          _cg_x000_reg = _cg_x000_vecbuf;
          _cg_beta_j010_reg = _cg_beta_j010_vecbuf;
        }
        buf0[7] += _cg_beta_i100_reg * _cg_x100_reg;
        buf0[7] -= _cg_beta_i100_reg * _cg_x000_reg;
        buf0[7] -= _cg_beta_j010_reg * _cg_x000_reg;
      }
      {
        // New offset [-1, 1, 7]
        bElem _cg_beta_j100_reg;
        bElem _cg_x100_reg;
        {
          _cg_beta_j000_vecbuf = _cg_beta_j010_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x000_vecbuf
          dev_shl(_cg_x000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_x100_reg = _cg_x000_vecbuf;
        }
        buf0[7] += _cg_beta_j100_reg * _cg_x100_reg;
      }
      {
        // New offset [-1, 0, 8]
        bElem _cg_beta_k100_reg;
        bElem _cg_x100_reg;
        {
          _cg_beta_k000_vecbuf = _cg_beta_k001_vecbuf;
          _cg_x000_vecbuf = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
          _cg_x100_reg = _cg_x000_vecbuf;
        }
        buf0[7] += _cg_beta_k100_reg * _cg_x100_reg;
      }
    }
  }
  bElem buf1[8];
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
            buf1[0 + rel] = c[0] * _cg_alpha000_reg * _cg_x000_reg - c[1] * c[2] * buf0[0 + rel];
          }
          _cg_rel2 += 1;
        }
      }
    }
    bElem *out_ref = &out.dat[neighbor13 * out.step];
    for (long sti = 0; sti < 8; ++sti)
    {
      out_ref[sti * 64 + hipThreadIdx_x] = buf1[sti];
    }
  }
}
# 63 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/helmholtz2/intermediate_gen/helmholtz2.cu" 2

}
__global__ void helmholtz2_naive3(bElem (*x)[STRIDE1][STRIDE0], bElem (*alpha)[STRIDE1][STRIDE0], bElem (*beta_i)[STRIDE1][STRIDE0], bElem (*beta_j)[STRIDE1][STRIDE0], bElem (*beta_k)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0], bElem *c) {
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;
    bElem c1 = c[0];
    bElem c2 = c[1];
    bElem h2inv = c[2];
    out[k][j][i] = c1 * alpha[k][j][i] * x[k][j][i] - 
        c2 * h2inv * (
            beta_i[k][j][i + 1] * (x[k][j][i + 1] - x[k][j][i]) + 
            beta_i[k][j][i]     * (x[k][j][i - 1] - x[k][j][i]) +
            beta_j[k][j + 1][i] * (x[k][j + 1][i] - x[k][j][i]) +
            beta_j[k][j - 1][i] * (x[k][j - 1][i] - x[k][j][i]) +
            beta_k[k + 1][j][i] * (x[k + 1][j][i] - x[k][j][i]) +
            beta_k[k - 1][j][i] * (x[k - 1][j][i] - x[k][j][i])
        );
}
#define x(a, b, c) x_arr[c][b][a]
#define alpha(a, b, c) alpha_arr[c][b][a]
#define beta_i(a, b, c) beta_i_arr[c][b][a]
#define beta_j(a, b, c) beta_j_arr[c][b][a]
#define beta_k(a, b, c) beta_k_arr[c][b][a] 
#define out(a, b, c) out_arr[c][b][a]
__global__ void helmholtz2_codegen3(bElem (*x_arr)[STRIDE1][STRIDE0], bElem (*alpha_arr)[STRIDE1][STRIDE0], bElem (*beta_i_arr)[STRIDE1][STRIDE0], bElem (*beta_j_arr)[STRIDE1][STRIDE0], bElem (*beta_k_arr)[STRIDE1][STRIDE0], bElem (*out_arr)[STRIDE1][STRIDE0], bElem *c) {
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;
# 1 "VSTile-helmholtz23.py-HIP-8x8x64" 1
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
                buf0[0 + rel] = c[0] * alpha(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) - c[1] * c[2] * (beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2) - beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) + beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + _cg_idx2) - beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) + beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2) - beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) + beta_j(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1 + -1, k + _cg_idx2) - beta_j(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) + beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) + beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + -1) - beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2));
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
            out(i + _cg_idx0, j + _cg_idx1, k + _cg_idx2) = buf0[rel];
          }
        }
      }
    }
  }
}
# 92 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/helmholtz2/intermediate_gen/helmholtz2.cu" 2

}
#undef x
#undef alpha
#undef beta_i
#undef beta_j
#undef beta_k
#undef out
__global__ void helmholtz2_naive_bricks3(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType x, BType alpha, BType beta_i, BType beta_j, BType beta_k, BType out, bElem *c) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;
    bElem c1 = c[0];
    bElem c2 = c[1];
    bElem h2inv = c[2];
    out[b][k][j][i] = c1 * alpha[b][k][j][i] * x[b][k][j][i] - 
        c2 * h2inv * (
            beta_i[b][k][j][i + 1] * (x[b][k][j][i + 1] - x[b][k][j][i]) + 
            beta_i[b][k][j][i]     * (x[b][k][j][i - 1] - x[b][k][j][i]) +
            beta_j[b][k][j + 1][i] * (x[b][k][j + 1][i] - x[b][k][j][i]) +
            beta_j[b][k][j - 1][i] * (x[b][k][j - 1][i] - x[b][k][j][i]) +
            beta_k[b][k + 1][j][i] * (x[b][k + 1][j][i] - x[b][k][j][i]) +
            beta_k[b][k - 1][j][i] * (x[b][k - 1][j][i] - x[b][k][j][i])
        );
}
__global__ void helmholtz2_codegen_bricks3(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType x, BType alpha, BType beta_i, BType beta_j, BType beta_k, BType out, bElem *c) {
  unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
# 1 "VSBrick-helmholtz23.py-HIP-8x8x8-8x8" 1
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
      bElem _cg_beta_i000_vecbuf;
      bElem _cg_beta_i100_vecbuf;
      bElem _cg_x000_vecbuf;
      bElem _cg_x100_vecbuf;
      bElem _cg_x_100_vecbuf;
      bElem _cg_beta_j000_vecbuf;
      bElem _cg_beta_j010_vecbuf;
      bElem _cg_beta_k000_vecbuf;
      bElem _cg_beta_k001_vecbuf;
      {
        // New offset [-1, 0, -1]
        bElem _cg_beta_k101_reg;
        bElem _cg_x100_reg;
        {
          _cg_beta_k001_vecbuf = beta_k.dat[neighbor13 * beta_k.step + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          _cg_beta_k101_reg = _cg_beta_k001_vecbuf;
          _cg_x100_reg = _cg_x000_vecbuf;
        }
        buf0[0] += _cg_beta_k101_reg * _cg_x100_reg;
      }
      {
        // New offset [0, -1, 0]
        bElem _cg_beta_j010_reg;
        bElem _cg_x000_reg;
        {
          _cg_beta_j010_vecbuf = beta_j.dat[neighbor13 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x000_vecbuf
          dev_shl(_cg_x000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_beta_j010_reg = _cg_beta_j010_vecbuf;
          _cg_x000_reg = _cg_x000_vecbuf;
        }
        buf0[0] += _cg_beta_j010_reg * _cg_x000_reg;
      }
      {
        // New offset [-1, 0, 0]
        bElem _cg_beta_i100_reg;
        bElem _cg_x000_reg;
        bElem _cg_x100_reg;
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_k101_reg;
        bElem _cg_beta_k100_reg;
        {
          _cg_beta_j000_vecbuf = _cg_beta_j010_vecbuf;
          _cg_beta_k000_vecbuf = _cg_beta_k001_vecbuf;
          _cg_beta_i000_vecbuf = beta_i.dat[neighbor13 * beta_i.step + hipThreadIdx_x];
          _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          _cg_beta_k001_vecbuf = beta_k.dat[neighbor13 * beta_k.step + 64 + hipThreadIdx_x];
          _cg_beta_i100_reg = _cg_beta_i000_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x000_reg = _cg_vectmp0;
          _cg_x100_reg = _cg_x000_vecbuf;
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_beta_k101_reg = _cg_beta_k001_vecbuf;
          _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
        }
        buf0[0] += _cg_beta_i100_reg * _cg_x000_reg;
        buf0[0] -= _cg_beta_i100_reg * _cg_x100_reg;
        buf0[0] -= _cg_beta_j100_reg * _cg_x100_reg;
        buf0[0] -= _cg_beta_k101_reg * _cg_x100_reg;
        buf0[0] -= _cg_beta_k100_reg * _cg_x100_reg;
        buf0[1] += _cg_beta_k101_reg * _cg_x100_reg;
      }
      {
        // New offset [0, 0, 0]
        bElem _cg_beta_i100_reg;
        bElem _cg_x100_reg;
        bElem _cg_x000_reg;
        bElem _cg_beta_j010_reg;
        {
          _cg_beta_i100_vecbuf = beta_i.dat[neighbor14 * beta_i.step + hipThreadIdx_x];
          _cg_x100_vecbuf = x.dat[neighbor14 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_j.dat[neighbor16 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_j.dat[neighbor13 * beta_j.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j010_vecbuf
          dev_shl(_cg_beta_j010_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_beta_i000_vecbuf ,_cg_beta_i100_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_beta_i000_vecbuf, _cg_beta_i100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i100_reg = _cg_vectmp2;
          bElem _cg_vectmp3;
          // merge0 _cg_x000_vecbuf ,_cg_x100_vecbuf, 1 -> _cg_vectmp3
          dev_shl(_cg_vectmp3, _cg_x000_vecbuf, _cg_x100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x100_reg = _cg_vectmp3;
          _cg_x000_reg = _cg_x000_vecbuf;
          _cg_beta_j010_reg = _cg_beta_j010_vecbuf;
        }
        buf0[0] += _cg_beta_i100_reg * _cg_x100_reg;
        buf0[0] -= _cg_beta_i100_reg * _cg_x000_reg;
        buf0[0] -= _cg_beta_j010_reg * _cg_x000_reg;
      }
      {
        // New offset [-1, 1, 0]
        bElem _cg_beta_j100_reg;
        bElem _cg_x100_reg;
        {
          _cg_beta_j000_vecbuf = _cg_beta_j010_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x000_vecbuf
          dev_shl(_cg_x000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_x100_reg = _cg_x000_vecbuf;
        }
        buf0[0] += _cg_beta_j100_reg * _cg_x100_reg;
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 6; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, -1, 1]
            bElem _cg_beta_j010_reg;
            bElem _cg_x000_reg;
            {
              _cg_beta_j010_vecbuf = beta_j.dat[neighbor13 * beta_j.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor10 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x000_vecbuf
              dev_shl(_cg_x000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              _cg_beta_j010_reg = _cg_beta_j010_vecbuf;
              _cg_x000_reg = _cg_x000_vecbuf;
            }
            buf0[1 + rel] += _cg_beta_j010_reg * _cg_x000_reg;
          }
          {
            // New offset [-1, 0, 1]
            bElem _cg_beta_i100_reg;
            bElem _cg_x000_reg;
            bElem _cg_x100_reg;
            bElem _cg_beta_j100_reg;
            bElem _cg_beta_k101_reg;
            bElem _cg_beta_k100_reg;
            {
              _cg_beta_j000_vecbuf = _cg_beta_j010_vecbuf;
              _cg_beta_k000_vecbuf = _cg_beta_k001_vecbuf;
              _cg_beta_i000_vecbuf = beta_i.dat[neighbor13 * beta_i.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_x000_vecbuf = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_k001_vecbuf = beta_k.dat[neighbor13 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_i100_reg = _cg_beta_i000_vecbuf;
              bElem _cg_vectmp0;
              // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_x000_reg = _cg_vectmp0;
              _cg_x100_reg = _cg_x000_vecbuf;
              _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
              _cg_beta_k101_reg = _cg_beta_k001_vecbuf;
              _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
            }
            buf0[1 + rel] += _cg_beta_i100_reg * _cg_x000_reg;
            buf0[1 + rel] -= _cg_beta_i100_reg * _cg_x100_reg;
            buf0[1 + rel] -= _cg_beta_j100_reg * _cg_x100_reg;
            buf0[1 + rel] -= _cg_beta_k101_reg * _cg_x100_reg;
            buf0[1 + rel] -= _cg_beta_k100_reg * _cg_x100_reg;
            buf0[0 + rel] += _cg_beta_k100_reg * _cg_x100_reg;
            buf0[2 + rel] += _cg_beta_k101_reg * _cg_x100_reg;
          }
          {
            // New offset [0, 0, 1]
            bElem _cg_beta_i100_reg;
            bElem _cg_x100_reg;
            bElem _cg_x000_reg;
            bElem _cg_beta_j010_reg;
            {
              _cg_beta_i100_vecbuf = beta_i.dat[neighbor14 * beta_i.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_x100_vecbuf = x.dat[neighbor14 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              _cg_vectmp0 = beta_j.dat[neighbor16 * beta_j.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = beta_j.dat[neighbor13 * beta_j.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j010_vecbuf
              dev_shl(_cg_beta_j010_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_beta_i000_vecbuf ,_cg_beta_i100_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_beta_i000_vecbuf, _cg_beta_i100_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_i100_reg = _cg_vectmp2;
              bElem _cg_vectmp3;
              // merge0 _cg_x000_vecbuf ,_cg_x100_vecbuf, 1 -> _cg_vectmp3
              dev_shl(_cg_vectmp3, _cg_x000_vecbuf, _cg_x100_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_x100_reg = _cg_vectmp3;
              _cg_x000_reg = _cg_x000_vecbuf;
              _cg_beta_j010_reg = _cg_beta_j010_vecbuf;
            }
            buf0[1 + rel] += _cg_beta_i100_reg * _cg_x100_reg;
            buf0[1 + rel] -= _cg_beta_i100_reg * _cg_x000_reg;
            buf0[1 + rel] -= _cg_beta_j010_reg * _cg_x000_reg;
          }
          {
            // New offset [-1, 1, 1]
            bElem _cg_beta_j100_reg;
            bElem _cg_x100_reg;
            {
              _cg_beta_j000_vecbuf = _cg_beta_j010_vecbuf;
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor16 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x000_vecbuf
              dev_shl(_cg_x000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
              _cg_x100_reg = _cg_x000_vecbuf;
            }
            buf0[1 + rel] += _cg_beta_j100_reg * _cg_x100_reg;
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [0, -1, 7]
        bElem _cg_beta_j010_reg;
        bElem _cg_x000_reg;
        {
          _cg_beta_j010_vecbuf = beta_j.dat[neighbor13 * beta_j.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x000_vecbuf
          dev_shl(_cg_x000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_beta_j010_reg = _cg_beta_j010_vecbuf;
          _cg_x000_reg = _cg_x000_vecbuf;
        }
        buf0[7] += _cg_beta_j010_reg * _cg_x000_reg;
      }
      {
        // New offset [-1, 0, 7]
        bElem _cg_beta_i100_reg;
        bElem _cg_x000_reg;
        bElem _cg_x100_reg;
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_k101_reg;
        bElem _cg_beta_k100_reg;
        {
          _cg_beta_j000_vecbuf = _cg_beta_j010_vecbuf;
          _cg_beta_k000_vecbuf = _cg_beta_k001_vecbuf;
          _cg_beta_i000_vecbuf = beta_i.dat[neighbor13 * beta_i.step + 448 + hipThreadIdx_x];
          _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + 448 + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          _cg_beta_k001_vecbuf = beta_k.dat[neighbor22 * beta_k.step + hipThreadIdx_x];
          _cg_beta_i100_reg = _cg_beta_i000_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x000_reg = _cg_vectmp0;
          _cg_x100_reg = _cg_x000_vecbuf;
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_beta_k101_reg = _cg_beta_k001_vecbuf;
          _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
        }
        buf0[7] += _cg_beta_i100_reg * _cg_x000_reg;
        buf0[7] -= _cg_beta_i100_reg * _cg_x100_reg;
        buf0[7] -= _cg_beta_j100_reg * _cg_x100_reg;
        buf0[7] -= _cg_beta_k101_reg * _cg_x100_reg;
        buf0[7] -= _cg_beta_k100_reg * _cg_x100_reg;
        buf0[6] += _cg_beta_k100_reg * _cg_x100_reg;
      }
      {
        // New offset [0, 0, 7]
        bElem _cg_beta_i100_reg;
        bElem _cg_x100_reg;
        bElem _cg_x000_reg;
        bElem _cg_beta_j010_reg;
        {
          _cg_beta_i100_vecbuf = beta_i.dat[neighbor14 * beta_i.step + 448 + hipThreadIdx_x];
          _cg_x100_vecbuf = x.dat[neighbor14 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_j.dat[neighbor16 * beta_j.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_j.dat[neighbor13 * beta_j.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j010_vecbuf
          dev_shl(_cg_beta_j010_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_beta_i000_vecbuf ,_cg_beta_i100_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_beta_i000_vecbuf, _cg_beta_i100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i100_reg = _cg_vectmp2;
          bElem _cg_vectmp3;
          // merge0 _cg_x000_vecbuf ,_cg_x100_vecbuf, 1 -> _cg_vectmp3
          dev_shl(_cg_vectmp3, _cg_x000_vecbuf, _cg_x100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x100_reg = _cg_vectmp3;
          _cg_x000_reg = _cg_x000_vecbuf;
          _cg_beta_j010_reg = _cg_beta_j010_vecbuf;
        }
        buf0[7] += _cg_beta_i100_reg * _cg_x100_reg;
        buf0[7] -= _cg_beta_i100_reg * _cg_x000_reg;
        buf0[7] -= _cg_beta_j010_reg * _cg_x000_reg;
      }
      {
        // New offset [-1, 1, 7]
        bElem _cg_beta_j100_reg;
        bElem _cg_x100_reg;
        {
          _cg_beta_j000_vecbuf = _cg_beta_j010_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x000_vecbuf
          dev_shl(_cg_x000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_x100_reg = _cg_x000_vecbuf;
        }
        buf0[7] += _cg_beta_j100_reg * _cg_x100_reg;
      }
      {
        // New offset [-1, 0, 8]
        bElem _cg_beta_k100_reg;
        bElem _cg_x100_reg;
        {
          _cg_beta_k000_vecbuf = _cg_beta_k001_vecbuf;
          _cg_x000_vecbuf = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
          _cg_x100_reg = _cg_x000_vecbuf;
        }
        buf0[7] += _cg_beta_k100_reg * _cg_x100_reg;
      }
    }
  }
  bElem buf1[8];
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
            buf1[0 + rel] = c[0] * _cg_alpha000_reg * _cg_x000_reg - c[1] * c[2] * buf0[0 + rel];
          }
          _cg_rel2 += 1;
        }
      }
    }
    bElem *out_ref = &out.dat[neighbor13 * out.step];
    for (long sti = 0; sti < 8; ++sti)
    {
      out_ref[sti * 64 + hipThreadIdx_x] = buf1[sti];
    }
  }
}
# 120 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/helmholtz2/intermediate_gen/helmholtz2.cu" 2

}
__global__ void helmholtz2_naive5(bElem (*x)[STRIDE1][STRIDE0], bElem (*alpha)[STRIDE1][STRIDE0], bElem (*beta_i)[STRIDE1][STRIDE0], bElem (*beta_j)[STRIDE1][STRIDE0], bElem (*beta_k)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0], bElem *c) {
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;
    bElem c1 = c[0];
    bElem c2 = c[1];
    bElem h2inv = c[2];
    out[k][j][i] = c1 * alpha[k][j][i] * x[k][j][i] - 
        c2 * h2inv * (
            beta_i[k][j][i + 1] * (x[k][j][i + 1] - x[k][j][i]) + 
            beta_i[k][j][i]     * (x[k][j][i - 1] - x[k][j][i]) +
            beta_j[k][j + 1][i] * (x[k][j + 1][i] - x[k][j][i]) +
            beta_j[k][j - 1][i] * (x[k][j - 1][i] - x[k][j][i]) +
            beta_k[k + 1][j][i] * (x[k + 1][j][i] - x[k][j][i]) +
            beta_k[k - 1][j][i] * (x[k - 1][j][i] - x[k][j][i])
        );
}
#define x(a, b, c) x_arr[c][b][a]
#define alpha(a, b, c) alpha_arr[c][b][a]
#define beta_i(a, b, c) beta_i_arr[c][b][a]
#define beta_j(a, b, c) beta_j_arr[c][b][a]
#define beta_k(a, b, c) beta_k_arr[c][b][a] 
#define out(a, b, c) out_arr[c][b][a]
__global__ void helmholtz2_codegen5(bElem (*x_arr)[STRIDE1][STRIDE0], bElem (*alpha_arr)[STRIDE1][STRIDE0], bElem (*beta_i_arr)[STRIDE1][STRIDE0], bElem (*beta_j_arr)[STRIDE1][STRIDE0], bElem (*beta_k_arr)[STRIDE1][STRIDE0], bElem (*out_arr)[STRIDE1][STRIDE0], bElem *c) {
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;
# 1 "VSTile-helmholtz25.py-HIP-8x8x64" 1
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
                buf0[0 + rel] = c[0] * alpha(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) - c[1] * c[2] * (beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2) - beta_i(i + hipThreadIdx_x + 1, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) + beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x + -1, j + _cg_idx1, k + _cg_idx2) - beta_i(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) + beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2) - beta_j(i + hipThreadIdx_x, j + _cg_idx1 + 1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) + beta_j(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1 + -1, k + _cg_idx2) - beta_j(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) + beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1) - beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + 1) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) + beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2 + -1) - beta_k(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2) * x(i + hipThreadIdx_x, j + _cg_idx1, k + _cg_idx2));
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
            out(i + _cg_idx0, j + _cg_idx1, k + _cg_idx2) = buf0[rel];
          }
        }
      }
    }
  }
}
# 149 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/helmholtz2/intermediate_gen/helmholtz2.cu" 2

}
#undef x
#undef alpha
#undef beta_i
#undef beta_j
#undef beta_k
#undef out
__global__ void helmholtz2_naive_bricks5(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType x, BType alpha, BType beta_i, BType beta_j, BType beta_k, BType out, bElem *c) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;
    bElem c1 = c[0];
    bElem c2 = c[1];
    bElem h2inv = c[2];
    out[b][k][j][i] = c1 * alpha[b][k][j][i] * x[b][k][j][i] - 
        c2 * h2inv * (
            beta_i[b][k][j][i + 1] * (x[b][k][j][i + 1] - x[b][k][j][i]) + 
            beta_i[b][k][j][i]     * (x[b][k][j][i - 1] - x[b][k][j][i]) +
            beta_j[b][k][j + 1][i] * (x[b][k][j + 1][i] - x[b][k][j][i]) +
            beta_j[b][k][j - 1][i] * (x[b][k][j - 1][i] - x[b][k][j][i]) +
            beta_k[b][k + 1][j][i] * (x[b][k + 1][j][i] - x[b][k][j][i]) +
            beta_k[b][k - 1][j][i] * (x[b][k - 1][j][i] - x[b][k][j][i])
        );
}
__global__ void helmholtz2_codegen_bricks5(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType x, BType alpha, BType beta_i, BType beta_j, BType beta_k, BType out, bElem *c) {
  unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
# 1 "VSBrick-helmholtz25.py-HIP-8x8x8-8x8" 1
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
      bElem _cg_beta_i000_vecbuf;
      bElem _cg_beta_i100_vecbuf;
      bElem _cg_x000_vecbuf;
      bElem _cg_x100_vecbuf;
      bElem _cg_x_100_vecbuf;
      bElem _cg_beta_j000_vecbuf;
      bElem _cg_beta_j010_vecbuf;
      bElem _cg_beta_k000_vecbuf;
      bElem _cg_beta_k001_vecbuf;
      {
        // New offset [-1, 0, -1]
        bElem _cg_beta_k101_reg;
        bElem _cg_x100_reg;
        {
          _cg_beta_k001_vecbuf = beta_k.dat[neighbor13 * beta_k.step + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor4 * x.step + 448 + hipThreadIdx_x];
          _cg_beta_k101_reg = _cg_beta_k001_vecbuf;
          _cg_x100_reg = _cg_x000_vecbuf;
        }
        buf0[0] += _cg_beta_k101_reg * _cg_x100_reg;
      }
      {
        // New offset [0, -1, 0]
        bElem _cg_beta_j010_reg;
        bElem _cg_x000_reg;
        {
          _cg_beta_j010_vecbuf = beta_j.dat[neighbor13 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x000_vecbuf
          dev_shl(_cg_x000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_beta_j010_reg = _cg_beta_j010_vecbuf;
          _cg_x000_reg = _cg_x000_vecbuf;
        }
        buf0[0] += _cg_beta_j010_reg * _cg_x000_reg;
      }
      {
        // New offset [-1, 0, 0]
        bElem _cg_beta_i100_reg;
        bElem _cg_x000_reg;
        bElem _cg_x100_reg;
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_k101_reg;
        bElem _cg_beta_k100_reg;
        {
          _cg_beta_j000_vecbuf = _cg_beta_j010_vecbuf;
          _cg_beta_k000_vecbuf = _cg_beta_k001_vecbuf;
          _cg_beta_i000_vecbuf = beta_i.dat[neighbor13 * beta_i.step + hipThreadIdx_x];
          _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          _cg_beta_k001_vecbuf = beta_k.dat[neighbor13 * beta_k.step + 64 + hipThreadIdx_x];
          _cg_beta_i100_reg = _cg_beta_i000_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x000_reg = _cg_vectmp0;
          _cg_x100_reg = _cg_x000_vecbuf;
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_beta_k101_reg = _cg_beta_k001_vecbuf;
          _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
        }
        buf0[0] += _cg_beta_i100_reg * _cg_x000_reg;
        buf0[0] -= _cg_beta_i100_reg * _cg_x100_reg;
        buf0[0] -= _cg_beta_j100_reg * _cg_x100_reg;
        buf0[0] -= _cg_beta_k101_reg * _cg_x100_reg;
        buf0[0] -= _cg_beta_k100_reg * _cg_x100_reg;
        buf0[1] += _cg_beta_k101_reg * _cg_x100_reg;
      }
      {
        // New offset [0, 0, 0]
        bElem _cg_beta_i100_reg;
        bElem _cg_x100_reg;
        bElem _cg_x000_reg;
        bElem _cg_beta_j010_reg;
        {
          _cg_beta_i100_vecbuf = beta_i.dat[neighbor14 * beta_i.step + hipThreadIdx_x];
          _cg_x100_vecbuf = x.dat[neighbor14 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_j.dat[neighbor16 * beta_j.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_j.dat[neighbor13 * beta_j.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j010_vecbuf
          dev_shl(_cg_beta_j010_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_beta_i000_vecbuf ,_cg_beta_i100_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_beta_i000_vecbuf, _cg_beta_i100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i100_reg = _cg_vectmp2;
          bElem _cg_vectmp3;
          // merge0 _cg_x000_vecbuf ,_cg_x100_vecbuf, 1 -> _cg_vectmp3
          dev_shl(_cg_vectmp3, _cg_x000_vecbuf, _cg_x100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x100_reg = _cg_vectmp3;
          _cg_x000_reg = _cg_x000_vecbuf;
          _cg_beta_j010_reg = _cg_beta_j010_vecbuf;
        }
        buf0[0] += _cg_beta_i100_reg * _cg_x100_reg;
        buf0[0] -= _cg_beta_i100_reg * _cg_x000_reg;
        buf0[0] -= _cg_beta_j010_reg * _cg_x000_reg;
      }
      {
        // New offset [-1, 1, 0]
        bElem _cg_beta_j100_reg;
        bElem _cg_x100_reg;
        {
          _cg_beta_j000_vecbuf = _cg_beta_j010_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x000_vecbuf
          dev_shl(_cg_x000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_x100_reg = _cg_x000_vecbuf;
        }
        buf0[0] += _cg_beta_j100_reg * _cg_x100_reg;
      }
      {
        long _cg_rel2 = 0;
        for (long _cg_idx2 = 0; _cg_idx2 < 6; _cg_idx2 += 1)
        {
          long rel = _cg_rel2;
          {
            // New offset [0, -1, 1]
            bElem _cg_beta_j010_reg;
            bElem _cg_x000_reg;
            {
              _cg_beta_j010_vecbuf = beta_j.dat[neighbor13 * beta_j.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor10 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x000_vecbuf
              dev_shl(_cg_x000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
              _cg_beta_j010_reg = _cg_beta_j010_vecbuf;
              _cg_x000_reg = _cg_x000_vecbuf;
            }
            buf0[1 + rel] += _cg_beta_j010_reg * _cg_x000_reg;
          }
          {
            // New offset [-1, 0, 1]
            bElem _cg_beta_i100_reg;
            bElem _cg_x000_reg;
            bElem _cg_x100_reg;
            bElem _cg_beta_j100_reg;
            bElem _cg_beta_k101_reg;
            bElem _cg_beta_k100_reg;
            {
              _cg_beta_j000_vecbuf = _cg_beta_j010_vecbuf;
              _cg_beta_k000_vecbuf = _cg_beta_k001_vecbuf;
              _cg_beta_i000_vecbuf = beta_i.dat[neighbor13 * beta_i.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_x000_vecbuf = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_k001_vecbuf = beta_k.dat[neighbor13 * beta_k.step + 128 + (hipThreadIdx_x + rel * 64)];
              _cg_beta_i100_reg = _cg_beta_i000_vecbuf;
              bElem _cg_vectmp0;
              // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
              dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
              _cg_x000_reg = _cg_vectmp0;
              _cg_x100_reg = _cg_x000_vecbuf;
              _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
              _cg_beta_k101_reg = _cg_beta_k001_vecbuf;
              _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
            }
            buf0[1 + rel] += _cg_beta_i100_reg * _cg_x000_reg;
            buf0[1 + rel] -= _cg_beta_i100_reg * _cg_x100_reg;
            buf0[1 + rel] -= _cg_beta_j100_reg * _cg_x100_reg;
            buf0[1 + rel] -= _cg_beta_k101_reg * _cg_x100_reg;
            buf0[1 + rel] -= _cg_beta_k100_reg * _cg_x100_reg;
            buf0[0 + rel] += _cg_beta_k100_reg * _cg_x100_reg;
            buf0[2 + rel] += _cg_beta_k101_reg * _cg_x100_reg;
          }
          {
            // New offset [0, 0, 1]
            bElem _cg_beta_i100_reg;
            bElem _cg_x100_reg;
            bElem _cg_x000_reg;
            bElem _cg_beta_j010_reg;
            {
              _cg_beta_i100_vecbuf = beta_i.dat[neighbor14 * beta_i.step + 64 + (hipThreadIdx_x + rel * 64)];
              _cg_x100_vecbuf = x.dat[neighbor14 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp0;
              _cg_vectmp0 = beta_j.dat[neighbor16 * beta_j.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = beta_j.dat[neighbor13 * beta_j.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j010_vecbuf
              dev_shl(_cg_beta_j010_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              bElem _cg_vectmp2;
              // merge0 _cg_beta_i000_vecbuf ,_cg_beta_i100_vecbuf, 1 -> _cg_vectmp2
              dev_shl(_cg_vectmp2, _cg_beta_i000_vecbuf, _cg_beta_i100_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_beta_i100_reg = _cg_vectmp2;
              bElem _cg_vectmp3;
              // merge0 _cg_x000_vecbuf ,_cg_x100_vecbuf, 1 -> _cg_vectmp3
              dev_shl(_cg_vectmp3, _cg_x000_vecbuf, _cg_x100_vecbuf, 7, 8, hipThreadIdx_x & 7);
              _cg_x100_reg = _cg_vectmp3;
              _cg_x000_reg = _cg_x000_vecbuf;
              _cg_beta_j010_reg = _cg_beta_j010_vecbuf;
            }
            buf0[1 + rel] += _cg_beta_i100_reg * _cg_x100_reg;
            buf0[1 + rel] -= _cg_beta_i100_reg * _cg_x000_reg;
            buf0[1 + rel] -= _cg_beta_j010_reg * _cg_x000_reg;
          }
          {
            // New offset [-1, 1, 1]
            bElem _cg_beta_j100_reg;
            bElem _cg_x100_reg;
            {
              _cg_beta_j000_vecbuf = _cg_beta_j010_vecbuf;
              bElem _cg_vectmp0;
              _cg_vectmp0 = x.dat[neighbor16 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              bElem _cg_vectmp1;
              _cg_vectmp1 = x.dat[neighbor13 * x.step + 64 + (hipThreadIdx_x + rel * 64)];
              // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x000_vecbuf
              dev_shl(_cg_x000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
              _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
              _cg_x100_reg = _cg_x000_vecbuf;
            }
            buf0[1 + rel] += _cg_beta_j100_reg * _cg_x100_reg;
          }
          _cg_rel2 += 1;
        }
      }
      {
        // New offset [0, -1, 7]
        bElem _cg_beta_j010_reg;
        bElem _cg_x000_reg;
        {
          _cg_beta_j010_vecbuf = beta_j.dat[neighbor13 * beta_j.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor10 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 7 -> _cg_x000_vecbuf
          dev_shl(_cg_x000_vecbuf, _cg_vectmp1, _cg_vectmp0, 8, 64, hipThreadIdx_x);
          _cg_beta_j010_reg = _cg_beta_j010_vecbuf;
          _cg_x000_reg = _cg_x000_vecbuf;
        }
        buf0[7] += _cg_beta_j010_reg * _cg_x000_reg;
      }
      {
        // New offset [-1, 0, 7]
        bElem _cg_beta_i100_reg;
        bElem _cg_x000_reg;
        bElem _cg_x100_reg;
        bElem _cg_beta_j100_reg;
        bElem _cg_beta_k101_reg;
        bElem _cg_beta_k100_reg;
        {
          _cg_beta_j000_vecbuf = _cg_beta_j010_vecbuf;
          _cg_beta_k000_vecbuf = _cg_beta_k001_vecbuf;
          _cg_beta_i000_vecbuf = beta_i.dat[neighbor13 * beta_i.step + 448 + hipThreadIdx_x];
          _cg_x_100_vecbuf = x.dat[neighbor12 * x.step + 448 + hipThreadIdx_x];
          _cg_x000_vecbuf = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          _cg_beta_k001_vecbuf = beta_k.dat[neighbor22 * beta_k.step + hipThreadIdx_x];
          _cg_beta_i100_reg = _cg_beta_i000_vecbuf;
          bElem _cg_vectmp0;
          // merge0 _cg_x_100_vecbuf ,_cg_x000_vecbuf, 7 -> _cg_vectmp0
          dev_shl(_cg_vectmp0, _cg_x_100_vecbuf, _cg_x000_vecbuf, 1, 8, hipThreadIdx_x & 7);
          _cg_x000_reg = _cg_vectmp0;
          _cg_x100_reg = _cg_x000_vecbuf;
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_beta_k101_reg = _cg_beta_k001_vecbuf;
          _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
        }
        buf0[7] += _cg_beta_i100_reg * _cg_x000_reg;
        buf0[7] -= _cg_beta_i100_reg * _cg_x100_reg;
        buf0[7] -= _cg_beta_j100_reg * _cg_x100_reg;
        buf0[7] -= _cg_beta_k101_reg * _cg_x100_reg;
        buf0[7] -= _cg_beta_k100_reg * _cg_x100_reg;
        buf0[6] += _cg_beta_k100_reg * _cg_x100_reg;
      }
      {
        // New offset [0, 0, 7]
        bElem _cg_beta_i100_reg;
        bElem _cg_x100_reg;
        bElem _cg_x000_reg;
        bElem _cg_beta_j010_reg;
        {
          _cg_beta_i100_vecbuf = beta_i.dat[neighbor14 * beta_i.step + 448 + hipThreadIdx_x];
          _cg_x100_vecbuf = x.dat[neighbor14 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp0;
          _cg_vectmp0 = beta_j.dat[neighbor16 * beta_j.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = beta_j.dat[neighbor13 * beta_j.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_beta_j010_vecbuf
          dev_shl(_cg_beta_j010_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          bElem _cg_vectmp2;
          // merge0 _cg_beta_i000_vecbuf ,_cg_beta_i100_vecbuf, 1 -> _cg_vectmp2
          dev_shl(_cg_vectmp2, _cg_beta_i000_vecbuf, _cg_beta_i100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_beta_i100_reg = _cg_vectmp2;
          bElem _cg_vectmp3;
          // merge0 _cg_x000_vecbuf ,_cg_x100_vecbuf, 1 -> _cg_vectmp3
          dev_shl(_cg_vectmp3, _cg_x000_vecbuf, _cg_x100_vecbuf, 7, 8, hipThreadIdx_x & 7);
          _cg_x100_reg = _cg_vectmp3;
          _cg_x000_reg = _cg_x000_vecbuf;
          _cg_beta_j010_reg = _cg_beta_j010_vecbuf;
        }
        buf0[7] += _cg_beta_i100_reg * _cg_x100_reg;
        buf0[7] -= _cg_beta_i100_reg * _cg_x000_reg;
        buf0[7] -= _cg_beta_j010_reg * _cg_x000_reg;
      }
      {
        // New offset [-1, 1, 7]
        bElem _cg_beta_j100_reg;
        bElem _cg_x100_reg;
        {
          _cg_beta_j000_vecbuf = _cg_beta_j010_vecbuf;
          bElem _cg_vectmp0;
          _cg_vectmp0 = x.dat[neighbor16 * x.step + 448 + hipThreadIdx_x];
          bElem _cg_vectmp1;
          _cg_vectmp1 = x.dat[neighbor13 * x.step + 448 + hipThreadIdx_x];
          // merge1 _cg_vectmp1 ,_cg_vectmp0, 1 -> _cg_x000_vecbuf
          dev_shl(_cg_x000_vecbuf, _cg_vectmp1, _cg_vectmp0, 56, 64, hipThreadIdx_x);
          _cg_beta_j100_reg = _cg_beta_j000_vecbuf;
          _cg_x100_reg = _cg_x000_vecbuf;
        }
        buf0[7] += _cg_beta_j100_reg * _cg_x100_reg;
      }
      {
        // New offset [-1, 0, 8]
        bElem _cg_beta_k100_reg;
        bElem _cg_x100_reg;
        {
          _cg_beta_k000_vecbuf = _cg_beta_k001_vecbuf;
          _cg_x000_vecbuf = x.dat[neighbor22 * x.step + hipThreadIdx_x];
          _cg_beta_k100_reg = _cg_beta_k000_vecbuf;
          _cg_x100_reg = _cg_x000_vecbuf;
        }
        buf0[7] += _cg_beta_k100_reg * _cg_x100_reg;
      }
    }
  }
  bElem buf1[8];
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
            buf1[0 + rel] = c[0] * _cg_alpha000_reg * _cg_x000_reg - c[1] * c[2] * buf0[0 + rel];
          }
          _cg_rel2 += 1;
        }
      }
    }
    bElem *out_ref = &out.dat[neighbor13 * out.step];
    for (long sti = 0; sti < 8; ++sti)
    {
      out_ref[sti * 64 + hipThreadIdx_x] = buf1[sti];
    }
  }
}
# 177 "/autofs/nccs-svm1_home1/priyagoyal/bricklib-testing-framework/kernels/helmholtz2/intermediate_gen/helmholtz2.cu" 2

}
