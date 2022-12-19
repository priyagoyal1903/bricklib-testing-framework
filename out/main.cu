#include "./gen/consts.h"
#include <brick-hip.h>
#include <iostream>
#include "bricksetup.h"
#include "multiarray.h"
#include "brickcompare.h"
#include <omp.h>
#include <cmath>
#include <cassert>
#include <string.h>
#include "brick.h"
#include "vecscatter.h"
#include "./kernels/f3d/gen/f3d.h"
typedef void (*kernel)();
int main(void) {
	 kernel kernels[] = {[]() {
	bElem *arg0 = randomArray({STRIDE0,STRIDE1,STRIDE2});
	bElem *dev_arg0;
	{
		unsigned size = STRIDE0*STRIDE1*STRIDE2 * sizeof(bElem);
		gpuMalloc(&dev_arg0, size);
		gpuMemcpy(dev_arg0, arg0, size, gpuMemcpyHostToDevice);
	}
	bElem *arg1 = zeroArray({STRIDE0,STRIDE1,STRIDE2});
	bElem *dev_arg1;
	{
		unsigned size = STRIDE0*STRIDE1*STRIDE2 * sizeof(bElem);
		gpuMalloc(&dev_arg1, size);
		gpuMemcpy(dev_arg1, arg1, size, gpuMemcpyHostToDevice);
	}
	bElem *arg2 = randomArray({8,8,8});
	bElem *dev_arg2;
	{
		unsigned size = 8*8*8 * sizeof(bElem);
		gpuMalloc(&dev_arg2, size);
		gpuMemcpy(dev_arg2, arg2, size, gpuMemcpyHostToDevice);
	}
	printf("Executing naive f3d, size 1\n");
	for (int iter = 0; iter < 100; ++iter)
	gpuExecKernel(f3d_naive1, dim3(BLOCK0,BLOCK1,BLOCK2), dim3(TILE0,TILE1,TILE2), (bElem(*)[STRIDE1][STRIDE0]) dev_arg0, (bElem(*)[STRIDE1][STRIDE0]) dev_arg1, (bElem(*)[8][8]) dev_arg2);
	gpuDeviceSynchronize();
	gpuFree(dev_arg0);
	gpuFree(dev_arg1);
	gpuFree(dev_arg2);
},[]() {
	bElem *arg0 = randomArray({STRIDE0,STRIDE1,STRIDE2});
	bElem *dev_arg0;
	{
		unsigned size = STRIDE0*STRIDE1*STRIDE2 * sizeof(bElem);
		gpuMalloc(&dev_arg0, size);
		gpuMemcpy(dev_arg0, arg0, size, gpuMemcpyHostToDevice);
	}
	bElem *arg1 = zeroArray({STRIDE0,STRIDE1,STRIDE2});
	bElem *dev_arg1;
	{
		unsigned size = STRIDE0*STRIDE1*STRIDE2 * sizeof(bElem);
		gpuMalloc(&dev_arg1, size);
		gpuMemcpy(dev_arg1, arg1, size, gpuMemcpyHostToDevice);
	}
	bElem *arg2 = randomArray({8,8,8});
	bElem *dev_arg2;
	{
		unsigned size = 8*8*8 * sizeof(bElem);
		gpuMalloc(&dev_arg2, size);
		gpuMemcpy(dev_arg2, arg2, size, gpuMemcpyHostToDevice);
	}
	printf("Executing naive f3d, size 2\n");
	for (int iter = 0; iter < 100; ++iter)
	gpuExecKernel(f3d_naive2, dim3(BLOCK0,BLOCK1,BLOCK2), dim3(TILE0,TILE1,TILE2), (bElem(*)[STRIDE1][STRIDE0]) dev_arg0, (bElem(*)[STRIDE1][STRIDE0]) dev_arg1, (bElem(*)[8][8]) dev_arg2);
	gpuDeviceSynchronize();
	gpuFree(dev_arg0);
	gpuFree(dev_arg1);
	gpuFree(dev_arg2);
},[]() {
	unsigned *bgrid;
	auto binfo = init_grid<3>(bgrid, {NAIVE_BSTRIDE2,NAIVE_BSTRIDE1,NAIVE_BSTRIDE0}); 
	unsigned *device_bgrid;
	unsigned grid_size = (NAIVE_BSTRIDE2 * NAIVE_BSTRIDE1 * NAIVE_BSTRIDE0) * sizeof(unsigned); 
	gpuMalloc(&device_bgrid, grid_size);
	gpuMemcpy(device_bgrid, bgrid, grid_size, gpuMemcpyHostToDevice);
	BrickInfo<3> _binfo = movBrickInfo(binfo, gpuMemcpyHostToDevice);
	BrickInfo<3> *device_binfo;
	unsigned binfo_size = sizeof(BrickInfo<3>);
	gpuMalloc(&device_binfo, binfo_size);
	gpuMemcpy(device_binfo, &_binfo, binfo_size, gpuMemcpyHostToDevice);
	auto brick_size = cal_size<BRICK_SIZE>::value;
	auto brick_storage = BrickStorage::allocate(binfo.nbricks, brick_size * 2);
	BType brick0(&binfo, brick_storage, brick_size * 0);
	bElem *temp_arr0 = randomArray({STRIDE0,STRIDE1,STRIDE2});
	copyToBrick<3>({N0 + 2 * GZ0, N1 + 2 * GZ1, N2 + 2 * GZ2}, {PADDING0, PADDING1, PADDING2}, {0,0,0}, temp_arr0, bgrid, brick0);
	
	BType brick1(&binfo, brick_storage, brick_size * 1);
	BrickStorage device_bstorage = movBrickStorage(brick_storage, gpuMemcpyHostToDevice);
	brick0 = BType(device_binfo, device_bstorage, brick_size * 0);
	brick1 = BType(device_binfo, device_bstorage, brick_size * 1);
	bElem *local_arr0 = randomArray({8,8,8});
	bElem *device_arr0;
	{
		unsigned size = 8*8*8 * sizeof(bElem);
		gpuMalloc(&device_arr0, size);
		gpuMemcpy(device_arr0, local_arr0, size, gpuMemcpyHostToDevice);
	}
	printf("Executing naive-bricks f3d, size 1\n");
	for (int iter = 0; iter < 100; ++iter)
	gpuExecKernel(f3d_naive_bricks1, dim3(BLOCK0, BLOCK1, BLOCK2), dim3(TILE0, TILE1, TILE2),(unsigned(*)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0])device_bgrid, brick0, brick1, (bElem(*)[8][8]) device_arr0);
	gpuDeviceSynchronize();
	free(bgrid);
	free(binfo.adj);
	gpuFree(device_binfo);
	gpuFree(device_bgrid);
},[]() {
	unsigned *bgrid;
	auto binfo = init_grid<3>(bgrid, {NAIVE_BSTRIDE2,NAIVE_BSTRIDE1,NAIVE_BSTRIDE0}); 
	unsigned *device_bgrid;
	unsigned grid_size = (NAIVE_BSTRIDE2 * NAIVE_BSTRIDE1 * NAIVE_BSTRIDE0) * sizeof(unsigned); 
	gpuMalloc(&device_bgrid, grid_size);
	gpuMemcpy(device_bgrid, bgrid, grid_size, gpuMemcpyHostToDevice);
	BrickInfo<3> _binfo = movBrickInfo(binfo, gpuMemcpyHostToDevice);
	BrickInfo<3> *device_binfo;
	unsigned binfo_size = sizeof(BrickInfo<3>);
	gpuMalloc(&device_binfo, binfo_size);
	gpuMemcpy(device_binfo, &_binfo, binfo_size, gpuMemcpyHostToDevice);
	auto brick_size = cal_size<BRICK_SIZE>::value;
	auto brick_storage = BrickStorage::allocate(binfo.nbricks, brick_size * 2);
	BType brick0(&binfo, brick_storage, brick_size * 0);
	bElem *temp_arr0 = randomArray({STRIDE0,STRIDE1,STRIDE2});
	copyToBrick<3>({N0 + 2 * GZ0, N1 + 2 * GZ1, N2 + 2 * GZ2}, {PADDING0, PADDING1, PADDING2}, {0,0,0}, temp_arr0, bgrid, brick0);
	
	BType brick1(&binfo, brick_storage, brick_size * 1);
	BrickStorage device_bstorage = movBrickStorage(brick_storage, gpuMemcpyHostToDevice);
	brick0 = BType(device_binfo, device_bstorage, brick_size * 0);
	brick1 = BType(device_binfo, device_bstorage, brick_size * 1);
	bElem *local_arr0 = randomArray({8,8,8});
	bElem *device_arr0;
	{
		unsigned size = 8*8*8 * sizeof(bElem);
		gpuMalloc(&device_arr0, size);
		gpuMemcpy(device_arr0, local_arr0, size, gpuMemcpyHostToDevice);
	}
	printf("Executing naive-bricks f3d, size 2\n");
	for (int iter = 0; iter < 100; ++iter)
	gpuExecKernel(f3d_naive_bricks2, dim3(BLOCK0, BLOCK1, BLOCK2), dim3(TILE0, TILE1, TILE2),(unsigned(*)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0])device_bgrid, brick0, brick1, (bElem(*)[8][8]) device_arr0);
	gpuDeviceSynchronize();
	free(bgrid);
	free(binfo.adj);
	gpuFree(device_binfo);
	gpuFree(device_bgrid);
},[]() {
	bElem *arg0 = randomArray({STRIDE0,STRIDE1,STRIDE2});
	bElem *dev_arg0;
	{
		unsigned size = STRIDE0*STRIDE1*STRIDE2 * sizeof(bElem);
		gpuMalloc(&dev_arg0, size);
		gpuMemcpy(dev_arg0, arg0, size, gpuMemcpyHostToDevice);
	}
	bElem *arg1 = zeroArray({STRIDE0,STRIDE1,STRIDE2});
	bElem *dev_arg1;
	{
		unsigned size = STRIDE0*STRIDE1*STRIDE2 * sizeof(bElem);
		gpuMalloc(&dev_arg1, size);
		gpuMemcpy(dev_arg1, arg1, size, gpuMemcpyHostToDevice);
	}
	bElem *arg2 = randomArray({8,8,8});
	bElem *dev_arg2;
	{
		unsigned size = 8*8*8 * sizeof(bElem);
		gpuMalloc(&dev_arg2, size);
		gpuMemcpy(dev_arg2, arg2, size, gpuMemcpyHostToDevice);
	}
	printf("Executing codegen f3d, size 1\n");
	for (int iter = 0; iter < 100; ++iter)
	gpuExecKernel(f3d_codegen1, dim3(N0/VECSIZE,BLOCK1,BLOCK2), VECSIZE, (bElem(*)[STRIDE1][STRIDE0]) dev_arg0, (bElem(*)[STRIDE1][STRIDE0]) dev_arg1, (bElem(*)[8][8]) dev_arg2);
	gpuDeviceSynchronize();
	gpuFree(dev_arg0);
	gpuFree(dev_arg1);
	gpuFree(dev_arg2);
},[]() {
	bElem *arg0 = randomArray({STRIDE0,STRIDE1,STRIDE2});
	bElem *dev_arg0;
	{
		unsigned size = STRIDE0*STRIDE1*STRIDE2 * sizeof(bElem);
		gpuMalloc(&dev_arg0, size);
		gpuMemcpy(dev_arg0, arg0, size, gpuMemcpyHostToDevice);
	}
	bElem *arg1 = zeroArray({STRIDE0,STRIDE1,STRIDE2});
	bElem *dev_arg1;
	{
		unsigned size = STRIDE0*STRIDE1*STRIDE2 * sizeof(bElem);
		gpuMalloc(&dev_arg1, size);
		gpuMemcpy(dev_arg1, arg1, size, gpuMemcpyHostToDevice);
	}
	bElem *arg2 = randomArray({8,8,8});
	bElem *dev_arg2;
	{
		unsigned size = 8*8*8 * sizeof(bElem);
		gpuMalloc(&dev_arg2, size);
		gpuMemcpy(dev_arg2, arg2, size, gpuMemcpyHostToDevice);
	}
	printf("Executing codegen f3d, size 2\n");
	for (int iter = 0; iter < 100; ++iter)
	gpuExecKernel(f3d_codegen2, dim3(N0/VECSIZE,BLOCK1,BLOCK2), VECSIZE, (bElem(*)[STRIDE1][STRIDE0]) dev_arg0, (bElem(*)[STRIDE1][STRIDE0]) dev_arg1, (bElem(*)[8][8]) dev_arg2);
	gpuDeviceSynchronize();
	gpuFree(dev_arg0);
	gpuFree(dev_arg1);
	gpuFree(dev_arg2);
},[]() {
	unsigned *bgrid;
	auto binfo = init_grid<3>(bgrid, {NAIVE_BSTRIDE2,NAIVE_BSTRIDE1,NAIVE_BSTRIDE0}); 
	unsigned *device_bgrid;
	unsigned grid_size = (NAIVE_BSTRIDE2 * NAIVE_BSTRIDE1 * NAIVE_BSTRIDE0) * sizeof(unsigned); 
	gpuMalloc(&device_bgrid, grid_size);
	gpuMemcpy(device_bgrid, bgrid, grid_size, gpuMemcpyHostToDevice);
	BrickInfo<3> _binfo = movBrickInfo(binfo, gpuMemcpyHostToDevice);
	BrickInfo<3> *device_binfo;
	unsigned binfo_size = sizeof(BrickInfo<3>);
	gpuMalloc(&device_binfo, binfo_size);
	gpuMemcpy(device_binfo, &_binfo, binfo_size, gpuMemcpyHostToDevice);
	auto brick_size = cal_size<BRICK_SIZE>::value;
	auto brick_storage = BrickStorage::allocate(binfo.nbricks, brick_size * 2);
	BType brick0(&binfo, brick_storage, brick_size * 0);
	bElem *temp_arr0 = randomArray({STRIDE0,STRIDE1,STRIDE2});
	copyToBrick<3>({N0 + 2 * GZ0, N1 + 2 * GZ1, N2 + 2 * GZ2}, {PADDING0, PADDING1, PADDING2}, {0,0,0}, temp_arr0, bgrid, brick0);
	
	BType brick1(&binfo, brick_storage, brick_size * 1);
	BrickStorage device_bstorage = movBrickStorage(brick_storage, gpuMemcpyHostToDevice);
	brick0 = BType(device_binfo, device_bstorage, brick_size * 0);
	brick1 = BType(device_binfo, device_bstorage, brick_size * 1);
	bElem *local_arr0 = randomArray({8,8,8});
	bElem *device_arr0;
	{
		unsigned size = 8*8*8 * sizeof(bElem);
		gpuMalloc(&device_arr0, size);
		gpuMemcpy(device_arr0, local_arr0, size, gpuMemcpyHostToDevice);
	}
	printf("Executing codegen-bricks f3d, size 1\n");
	for (int iter = 0; iter < 100; ++iter)
	gpuExecKernel(f3d_codegen_bricks1, dim3(BLOCK0, BLOCK1, BLOCK2), VECSIZE,(unsigned(*)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0])device_bgrid, brick0, brick1, (bElem(*)[8][8]) device_arr0);
	gpuDeviceSynchronize();
	free(bgrid);
	free(binfo.adj);
	gpuFree(device_binfo);
	gpuFree(device_bgrid);
},[]() {
	unsigned *bgrid;
	auto binfo = init_grid<3>(bgrid, {NAIVE_BSTRIDE2,NAIVE_BSTRIDE1,NAIVE_BSTRIDE0}); 
	unsigned *device_bgrid;
	unsigned grid_size = (NAIVE_BSTRIDE2 * NAIVE_BSTRIDE1 * NAIVE_BSTRIDE0) * sizeof(unsigned); 
	gpuMalloc(&device_bgrid, grid_size);
	gpuMemcpy(device_bgrid, bgrid, grid_size, gpuMemcpyHostToDevice);
	BrickInfo<3> _binfo = movBrickInfo(binfo, gpuMemcpyHostToDevice);
	BrickInfo<3> *device_binfo;
	unsigned binfo_size = sizeof(BrickInfo<3>);
	gpuMalloc(&device_binfo, binfo_size);
	gpuMemcpy(device_binfo, &_binfo, binfo_size, gpuMemcpyHostToDevice);
	auto brick_size = cal_size<BRICK_SIZE>::value;
	auto brick_storage = BrickStorage::allocate(binfo.nbricks, brick_size * 2);
	BType brick0(&binfo, brick_storage, brick_size * 0);
	bElem *temp_arr0 = randomArray({STRIDE0,STRIDE1,STRIDE2});
	copyToBrick<3>({N0 + 2 * GZ0, N1 + 2 * GZ1, N2 + 2 * GZ2}, {PADDING0, PADDING1, PADDING2}, {0,0,0}, temp_arr0, bgrid, brick0);
	
	BType brick1(&binfo, brick_storage, brick_size * 1);
	BrickStorage device_bstorage = movBrickStorage(brick_storage, gpuMemcpyHostToDevice);
	brick0 = BType(device_binfo, device_bstorage, brick_size * 0);
	brick1 = BType(device_binfo, device_bstorage, brick_size * 1);
	bElem *local_arr0 = randomArray({8,8,8});
	bElem *device_arr0;
	{
		unsigned size = 8*8*8 * sizeof(bElem);
		gpuMalloc(&device_arr0, size);
		gpuMemcpy(device_arr0, local_arr0, size, gpuMemcpyHostToDevice);
	}
	printf("Executing codegen-bricks f3d, size 2\n");
	for (int iter = 0; iter < 100; ++iter)
	gpuExecKernel(f3d_codegen_bricks2, dim3(BLOCK0, BLOCK1, BLOCK2), VECSIZE,(unsigned(*)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0])device_bgrid, brick0, brick1, (bElem(*)[8][8]) device_arr0);
	gpuDeviceSynchronize();
	free(bgrid);
	free(binfo.adj);
	gpuFree(device_binfo);
	gpuFree(device_bgrid);
}};
	for (int i = 0; i < 8; i++) {
		kernels[i]();
	}
}
