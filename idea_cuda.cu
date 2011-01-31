#ifndef __DEVICE_EMULATION__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <openssl/idea.h>
#include <cuda_runtime_api.h>
#include "cuda_common.h"
#include "common.h"
//#include "lib/cuPrintf.cu"

__constant__ uint64_t idea_constant_schedule[27];
__shared__ IDEA_KEY_SCHEDULE idea_schedule;

float idea_elapsed;
cudaEvent_t idea_start,idea_stop;

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
	
	*((uint64_t *)&idea_schedule+threadIdx.x) = idea_constant_schedule[threadIdx.x];

	unsigned int *p = (unsigned int *)&idea_schedule;
	uint32_t x1,x2,x3,x4,t0,t1,ul,l0,l1;

	register uint64_t block = data[TX];

	// TODO: One split64 op!
	n2l((unsigned char *)&block,x2);
	n2l(((unsigned char *)&block)+4,x4);

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

	l0=(t0&0xffff)|((x1&0xffff)<<16);
	l1=(x4&0xffff)|((t1&0xffff)<<16);

	block = ((uint64_t)l0) << 32 | l1;
	flip64(block);
	data[TX] = block;
}

__global__ void IDEAdecKernel(uint64_t *data) {
	
}

extern "C" void IDEA_cuda_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, int enc, uint8_t **host_data, uint64_t **device_data, cudaStream_t stream) {
	assert(in && out && nbytes);
	cudaError_t cudaerrno;
	int gridSize;
	dim3 dimBlock(MAX_THREAD, 1, 1);

	transferHostToDevice(&in, (uint32_t **)device_data, host_data, &nbytes, stream);

	if ((nbytes%(MAX_THREAD*IDEA_BLOCK_SIZE))==0) {
		gridSize = nbytes/(MAX_THREAD*IDEA_BLOCK_SIZE);
	} else {
		if (nbytes < MAX_THREAD*IDEA_BLOCK_SIZE)
			dimBlock.x = nbytes / 8;
		gridSize = nbytes/(MAX_THREAD*IDEA_BLOCK_SIZE)+1;
	}

	if (output_verbosity==OUTPUT_VERBOSE)
		fprintf(stdout,"Starting IDEA kernel for %zu bytes with (%d, (%d, %d))...\n", nbytes, gridSize, dimBlock.x, dimBlock.y);

	if(enc == IDEA_ENCRYPT) {
		IDEAencKernel<<<gridSize,dimBlock,0,stream>>>(*device_data);
		_CUDA_N("IDEA encryption kernel could not be launched!");
	} else {
		IDEAdecKernel<<<gridSize,dimBlock,0,stream>>>(*device_data);
		_CUDA_N("IDEA decryption kernel could not be launched!");
	}

	transferDeviceToHost(&out, (uint32_t **)device_data, host_data, host_data, &nbytes, stream);
}

extern "C" void IDEA_cuda_transfer_key_schedule(IDEA_KEY_SCHEDULE *ks) {
	assert(ks);
	cudaError_t cudaerrno;
	size_t ks_size = sizeof(IDEA_KEY_SCHEDULE);
	_CUDA(cudaMemcpyToSymbolAsync(idea_constant_schedule,ks,ks_size,0,cudaMemcpyHostToDevice));
}

//
// CBC parallel decrypt
//

__global__ void IDEAdecKernel_cbc(uint32_t in[],uint32_t out[],uint32_t iv[]) {
}

extern "C" void IDEA_cuda_transfer_iv(const unsigned char *iv) {
}

extern "C" void IDEA_cuda_decrypt_cbc(const unsigned char *in, unsigned char *out, size_t nbytes) {
}

#ifndef CBC_ENC_CPU
//
// CBC  encrypt
//

__global__ void IDEAencKernel_cbc(uint32_t state[],uint32_t iv[],size_t length) {
}

#endif

extern "C" void IDEA_cuda_encrypt_cbc(const unsigned char *in, unsigned char *out, size_t nbytes) {
}
#else
#error "ERROR: DEVICE EMULATION is NOT supported."
#endif
