#ifndef __DEVICE_EMULATION__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <openssl/idea.h>
#include <openssl/evp.h>
#include <cuda_runtime_api.h>
#include "cuda_common.h"
#include "common.h"

__constant__ uint64_t idea_constant_schedule[27];
__shared__ uint64_t idea_schedule[27];

#define idea_mul(r,a,b,ul) \
	ul=__umul24(a,b); \
	if (ul != 0) { \
		r=(ul&0xffff)-(ul>>16); \
		r-=((r)>>16); \
	} \
	else \
		r=(-(int)a-b+1); /* assuming a or b is 0 and in range */ 

#define E_IDEA(num) \
	x1&=0xffff; \
	idea_mul(x1,x1,*p,ul); p++; \
	x2+= *(p++); \
	x3+= *(p++); \
	x4&=0xffff; \
	idea_mul(x4,x4,*p,ul); p++; \
	t0=(x1^x3)&0xffff; \
	idea_mul(t0,t0,*p,ul); p++; \
	t1=(t0+(x2^x4))&0xffff; \
	idea_mul(t1,t1,*p,ul); p++; \
	t0+=t1; \
	x1^=t1; \
	x4^=t0; \
	ul=x2^t0; /* do the swap to x3 */ \
	x2=x3^t1; \
	x3=ul;


__global__ void IDEAencKernel(uint64_t *data) {
	
	if(threadIdx.x < 27)
		idea_schedule[threadIdx.x] = idea_constant_schedule[threadIdx.x];

	unsigned int *p = (unsigned int *)&idea_schedule;
	uint32_t x1,x2,x3,x4,t0,t1,ul;

	register uint64_t block = data[TX];

	nl2i(block,x2,x4);

	x1=(x2>>16);
	x3=(x4>>16);

	__syncthreads();
	E_IDEA(0);
	E_IDEA(1);
	E_IDEA(2);
	E_IDEA(3);
	E_IDEA(4);
	E_IDEA(5);
	E_IDEA(6);
	E_IDEA(7);

	x1&=0xffff;
	idea_mul(x1,x1,*p,ul); p++;

	t0= x3+ *(p++);
	t1= x2+ *(p++);

	x4&=0xffff;
	idea_mul(x4,x4,*p,ul);

	x2=(t0&0xffff)|((x1&0xffff)<<16);
	x3=(x4&0xffff)|((t1&0xffff)<<16);

	block = ((uint64_t)x2) << 32 | x3;
	flip64(block);
	data[TX] = block;

	p = (unsigned int *)&idea_schedule;

	block = data[__umul24(gridDim.x,blockDim.x) + TX];

	nl2i(block,x2,x4);

	x1=(x2>>16);
	x3=(x4>>16);

	E_IDEA(0);
	E_IDEA(1);
	E_IDEA(2);
	E_IDEA(3);
	E_IDEA(4);
	E_IDEA(5);
	E_IDEA(6);
	E_IDEA(7);

	x1&=0xffff;
	idea_mul(x1,x1,*p,ul); p++;

	t0= x3+ *(p++);
	t1= x2+ *(p++);

	x4&=0xffff;
	idea_mul(x4,x4,*p,ul);

	x2=(t0&0xffff)|((x1&0xffff)<<16);
	x3=(x4&0xffff)|((t1&0xffff)<<16);

	block = ((uint64_t)x2) << 32 | x3;
	flip64(block);
	data[__umul24(gridDim.x,blockDim.x) + TX] = block;
	p = (unsigned int *)&idea_schedule;

	block = data[__umul24(gridDim.x,blockDim.x)*2 + TX];

	nl2i(block,x2,x4);

	x1=(x2>>16);
	x3=(x4>>16);

	E_IDEA(0);
	E_IDEA(1);
	E_IDEA(2);
	E_IDEA(3);
	E_IDEA(4);
	E_IDEA(5);
	E_IDEA(6);
	E_IDEA(7);

	x1&=0xffff;
	idea_mul(x1,x1,*p,ul); p++;

	t0= x3+ *(p++);
	t1= x2+ *(p++);

	x4&=0xffff;
	idea_mul(x4,x4,*p,ul);

	x2=(t0&0xffff)|((x1&0xffff)<<16);
	x3=(x4&0xffff)|((t1&0xffff)<<16);

	block = ((uint64_t)x2) << 32 | x3;
	flip64(block);
	data[__umul24(gridDim.x,blockDim.x)*2 + TX] = block;
	p = (unsigned int *)&idea_schedule;

	block = data[__umul24(gridDim.x,blockDim.x)*3 + TX];

	nl2i(block,x2,x4);

	x1=(x2>>16);
	x3=(x4>>16);

	E_IDEA(0);
	E_IDEA(1);
	E_IDEA(2);
	E_IDEA(3);
	E_IDEA(4);
	E_IDEA(5);
	E_IDEA(6);
	E_IDEA(7);

	x1&=0xffff;
	idea_mul(x1,x1,*p,ul); p++;

	t0= x3+ *(p++);
	t1= x2+ *(p++);

	x4&=0xffff;
	idea_mul(x4,x4,*p,ul);

	x2=(t0&0xffff)|((x1&0xffff)<<16);
	x3=(x4&0xffff)|((t1&0xffff)<<16);

	block = ((uint64_t)x2) << 32 | x3;
	flip64(block);
	data[__umul24(gridDim.x,blockDim.x)*3 + TX] = block;
}

/*
__global__ void IDEAdecKernel(uint64_t *data) {
	
}
*/

extern "C" void IDEA_cuda_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, EVP_CIPHER_CTX *ctx, uint8_t **host_data, uint64_t **device_data) {
	assert(in && out && nbytes);
	int gridSize;

	transferHostToDevice(&in, (uint32_t **)device_data, host_data, &nbytes);

	if ((nbytes%(MAX_THREAD*IDEA_BLOCK_SIZE*4))==0) {
		gridSize = nbytes/(MAX_THREAD*IDEA_BLOCK_SIZE*4);
	} else {
		gridSize = nbytes/(MAX_THREAD*IDEA_BLOCK_SIZE*4)+1;
	}

	#ifdef DEBUG
		cudaError_t cudaerrno;
		fprintf(stdout,"Starting IDEA kernel for %zu bytes with (%d, (%d))...\n", nbytes, gridSize, MAX_THREAD);
	#endif

	if(ctx->encrypt == IDEA_ENCRYPT) {
		IDEAencKernel<<<gridSize,MAX_THREAD>>>(*device_data);
	} else {
		//IDEAdecKernel<<<gridSize,MAX_THREAD>>>(*device_data);
	}
	
	#ifdef DEBUG
		_CUDA_N("IDEA kernel could not be launched!");
	#endif

	transferDeviceToHost(&out, (uint32_t **)device_data, host_data, host_data, &nbytes);
}

extern "C" void IDEA_cuda_transfer_key_schedule(IDEA_KEY_SCHEDULE *ks) {
	//assert(ks);
	cudaError_t cudaerrno;
	_CUDA(cudaMemcpyToSymbolAsync(idea_constant_schedule,ks,sizeof(IDEA_KEY_SCHEDULE),0,cudaMemcpyHostToDevice));
}

#else
#error "ERROR: DEVICE EMULATION is NOT supported."
#endif
