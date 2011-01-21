#ifndef __DEVICE_EMULATION__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <openssl/blowfish.h>
#include <cuda_runtime_api.h>
#include "cuda_common.h"
#include "common.h"
//#include "lib/cuPrintf.cu"

__constant__ BF_KEY bf_constant_schedule;
//__shared__ BF_KEY bf_schedule;

__device__ uint64_t *bf_device_data;
uint8_t  *bf_host_data;

float bf_elapsed;
cudaEvent_t bf_start,bf_stop;

#define BF_M  (0xFF<<2)
#define BF_0  (24-2)
#define BF_1  (16-2)
#define BF_2  ( 8-2)
#define BF_3  2
#define BF_ENC(LL,R,S,P) ( \
	LL^=P, \
	LL^= (((*(BF_LONG *)((unsigned char *)&(S[  0])+((R>>BF_0)&BF_M))+ \
		*(BF_LONG *)((unsigned char *)&(S[256])+((R>>BF_1)&BF_M)))^ \
		*(BF_LONG *)((unsigned char *)&(S[512])+((R>>BF_2)&BF_M)))+ \
		*(BF_LONG *)((unsigned char *)&(S[768])+((R<<BF_3)&BF_M))) \
	)

__global__ void BFencKernel(uint64_t *data) {
	register uint32_t l, r;
	register uint64_t block = data[TX];

	// TODO: Let's see if this is really faster than constant memory!
	/*
	if(threadIdx.x < 18)
		bf_schedule.P[threadIdx.x] = bf_constant_schedule.P[threadIdx.x];

	bf_schedule.S[threadIdx.x] = bf_constant_schedule.S[threadIdx.x];
	bf_schedule.S[threadIdx.x+128] = bf_constant_schedule.S[threadIdx.x+128];
	bf_schedule.S[threadIdx.x+256] = bf_constant_schedule.S[threadIdx.x+256];
	bf_schedule.S[threadIdx.x+384] = bf_constant_schedule.S[threadIdx.x+384];
	bf_schedule.S[threadIdx.x+512] = bf_constant_schedule.S[threadIdx.x+512];
	bf_schedule.S[threadIdx.x+640] = bf_constant_schedule.S[threadIdx.x+640];
	bf_schedule.S[threadIdx.x+768] = bf_constant_schedule.S[threadIdx.x+768];
	bf_schedule.S[threadIdx.x+896] = bf_constant_schedule.S[threadIdx.x+896];
	// TODO: Let's see if synching isn't needed
	//__syncthreads();
	*/

	n2l((unsigned char *)&block,l);
	n2l(((unsigned char *)&block)+4,r);

	register const uint32_t *p,*s;

	p=&(bf_constant_schedule.P[0]);
	s=&(bf_constant_schedule.S[0]);

	l^=p[0];
	BF_ENC(r,l,s,p[ 1]);
	BF_ENC(l,r,s,p[ 2]);
	BF_ENC(r,l,s,p[ 3]);
	BF_ENC(l,r,s,p[ 4]);
	BF_ENC(r,l,s,p[ 5]);
	BF_ENC(l,r,s,p[ 6]);
	BF_ENC(r,l,s,p[ 7]);
	BF_ENC(l,r,s,p[ 8]);
	BF_ENC(r,l,s,p[ 9]);
	BF_ENC(l,r,s,p[10]);
	BF_ENC(r,l,s,p[11]);
	BF_ENC(l,r,s,p[12]);
	BF_ENC(r,l,s,p[13]);
	BF_ENC(l,r,s,p[14]);
	BF_ENC(r,l,s,p[15]);
	BF_ENC(l,r,s,p[16]);
	r^=p[BF_ROUNDS+1];

	block = ((uint64_t)r) << 32 | l;
	flip64(block);
	data[TX] = block;

}

__global__ void BFdecKernel(uint64_t *data) {
	
}

extern "C" void BF_cuda_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, int enc, uint8_t **host_data, uint64_t **device_data) {
	assert(in && out && nbytes);
	cudaError_t cudaerrno;
	int gridSize;
	dim3 dimBlock(MAX_THREAD, 1, 1);

	transferHostToDevice(&in, (uint32_t **)device_data, host_data, &nbytes);

	if ((nbytes%(MAX_THREAD*BF_BLOCK_SIZE))==0) {
		gridSize = nbytes/(MAX_THREAD*BF_BLOCK_SIZE);
	} else {
		if (nbytes < MAX_THREAD*BF_BLOCK_SIZE)
			dimBlock.x = nbytes / 8;
		gridSize = nbytes/(MAX_THREAD*BF_BLOCK_SIZE)+1;
	}

	if (output_verbosity==OUTPUT_VERBOSE)
		fprintf(stdout,"Starting BF kernel for %zu bytes with (%d, (%d, %d))...\n", nbytes, gridSize, dimBlock.x, dimBlock.y);

	if(enc == BF_ENCRYPT) {
		BFencKernel<<<gridSize,dimBlock>>>(*device_data);
		_CUDA_N("BF encryption kernel could not be launched!");
	} else {
		BFdecKernel<<<gridSize,dimBlock>>>(*device_data);
		_CUDA_N("BF decryption kernel could not be launched!");
	}

	transferDeviceToHost(&out, (uint32_t **)device_data, host_data, host_data, &nbytes);
}

extern "C" void BF_cuda_transfer_key_schedule(BF_KEY *ks) {
	assert(ks);
	cudaError_t cudaerrno;
	size_t ks_size = sizeof(BF_KEY);
	_CUDA(cudaMemcpyToSymbolAsync(bf_constant_schedule,ks,ks_size,0,cudaMemcpyHostToDevice));
}

//
// CBC parallel decrypt
//

__global__ void BFdecKernel_cbc(uint32_t in[],uint32_t out[],uint32_t iv[]) {
}

extern "C" void BF_cuda_transfer_iv(const unsigned char *iv) {
}

extern "C" void BF_cuda_decrypt_cbc(const unsigned char *in, unsigned char *out, size_t nbytes) {
}

#ifndef CBC_ENC_CPU
//
// CBC  encrypt
//

__global__ void BFencKernel_cbc(uint32_t state[],uint32_t iv[],size_t length) {
}

#endif

extern "C" void BF_cuda_encrypt_cbc(const unsigned char *in, unsigned char *out, size_t nbytes) {
}
#else
#error "ERROR: DEVICE EMULATION is NOT supported."
#endif
