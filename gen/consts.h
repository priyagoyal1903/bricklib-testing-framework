// This is a generated file and should not be edited
#define VSVEC "HIP"
#define N0 384
#define N1 384
#define N2 384
#define TILE0 8
#define TILE1 8
#define TILE2 8
#define PADDING0 32
#define PADDING1 32
#define PADDING2 32
#define VECSIZE 64
#define FOLD 8,8
#define GZ0 (8)
#define GZ1 (8)
#define GZ2 (8)
#define OFF0 (GZ0 + 32)
#define OFF1 (GZ1 + 32)
#define OFF2 (GZ2 + 32)
#define STRIDE0 (384 + 2 * (OFF0))
#define STRIDE1 (384 + 2 * (OFF1))
#define STRIDE2 (384 + 2 * (OFF2))
#define GB0 (GZ0 / (8))
#define GB1 (GZ1 / (8))
#define GB2 (GZ2 / (8))
#define BLOCK0 (384 / 8)
#define BLOCK1 (384 / 8)
#define BLOCK2 (384 / 8)
#define NAIVE_BSTRIDE0 ((384 + 2 * GZ0) / (8))
#define NAIVE_BSTRIDE1 ((384 + 2 * GZ1) / (8))
#define NAIVE_BSTRIDE2 ((384 + 2 * GZ2) / (8))
#define BRICK_SIZE TILE2, TILE1, TILE0
#define BType Brick<Dim<BRICK_SIZE>, Dim<FOLD>>
