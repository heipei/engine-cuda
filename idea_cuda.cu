#ifndef __DEVICE_EMULATION__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <openssl/idea.h>
#include <cuda_runtime_api.h>
#include "cuda_common.h"
#include "common.h"
#include "lib/cuPrintf.cu"

__constant__ uint64_t idea_constant_schedule[27];
__shared__ IDEA_KEY_SCHEDULE idea_schedule;

__device__ __constant__ uint64_t *idea_device_data;
uint8_t  *idea_host_data;

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

#define n2l(c,l)        (l =((unsigned long)(*(c)))<<24L, \
                         l|=((unsigned long)(*(c+1)))<<16L, \
                         l|=((unsigned long)(*(c+2)))<< 8L, \
                         l|=((unsigned long)(*(c+3))))

#define flip64(a)	(a= \
			((a & 0x00000000000000FF) << 56) | \
			((a & 0x000000000000FF00) << 40) | \
			((a & 0x0000000000FF0000) << 24) | \
			((a & 0x00000000FF000000) << 8)  | \
			((a & 0x000000FF00000000) >> 8)  | \
			((a & 0x0000FF0000000000) >> 24) | \
			((a & 0x00FF000000000000) >> 40) | \
			((a & 0xFF00000000000000) >> 56))

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

extern "C" void IDEA_cuda_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, int enc) {
	assert(in && out && nbytes);
	cudaError_t cudaerrno;
	int gridSize;
	dim3 dimBlock(MAX_THREAD, 1, 1);

	transferHostToDevice(&in, (uint32_t **)&idea_device_data, &idea_host_data, &nbytes);

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
		IDEAencKernel<<<gridSize,dimBlock>>>(idea_device_data);
		_CUDA_N("IDEA encryption kernel could not be launched!");
	} else {
		IDEAdecKernel<<<gridSize,dimBlock>>>(idea_device_data);
		_CUDA_N("IDEA decryption kernel could not be launched!");
	}

	transferDeviceToHost(&out, (uint32_t **)&idea_device_data, &idea_host_data, &idea_host_data, &nbytes);
}

extern "C" void IDEA_cuda_transfer_key_schedule(IDEA_KEY_SCHEDULE *ks) {
	assert(ks);
	cudaError_t cudaerrno;
	size_t ks_size = sizeof(IDEA_KEY_SCHEDULE);
	_CUDA(cudaMemcpyToSymbolAsync(idea_constant_schedule,ks,ks_size,0,cudaMemcpyHostToDevice));
}

extern "C" void IDEA_cuda_finish() {
	cudaError_t cudaerrno;

	if (output_verbosity>=OUTPUT_NORMAL) fprintf(stdout, "\nDone. Finishing up IDEA\n");

#ifndef PAGEABLE 
#if CUDART_VERSION >= 2020
	if(isIntegrated) {
		_CUDA(cudaFreeHost(idea_host_data));
		//_CUDA(cudaFreeHost(idea_h_iv));
	} else {
		_CUDA(cudaFree(idea_device_data));
		//_CUDA(cudaFree(idea_d_iv));
	}
#else	
	_CUDA(cudaFree(idea_device_data));
	//_CUDA(cudaFree(idea_d_iv));
#endif
#else
	_CUDA(cudaFree(idea_device_data));
	//_CUDA(cudaFree(idea_d_iv));
#endif	

	_CUDA(cudaEventRecord(idea_stop,0));
	_CUDA(cudaEventSynchronize(idea_stop));
	_CUDA(cudaEventElapsedTime(&idea_elapsed,idea_start,idea_stop));

	if (output_verbosity>=OUTPUT_NORMAL) fprintf(stdout,"\nTotal time: %f milliseconds\n",idea_elapsed);	
}

extern "C" void IDEA_cuda_init(int *nm, int buffer_size_engine, int output_kind) {
	assert(nm);
	cudaError_t cudaerrno;
   	int buffer_size;
	cudaDeviceProp deviceProp;
    	
	output_verbosity=output_kind;

	checkCUDADevice(&deviceProp, output_verbosity);
	
	if(buffer_size_engine==0)
		buffer_size=MAX_CHUNK_SIZE;
	else 
		buffer_size=buffer_size_engine;
	
#if CUDART_VERSION >= 2000
	*nm=deviceProp.multiProcessorCount;
#endif

#ifndef PAGEABLE 
#if CUDART_VERSION >= 2020
	isIntegrated=deviceProp.integrated;
	if(isIntegrated) {
        	//zero-copy memory mode - use special function to get OS-pinned memory
		_CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));
        	if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"Using zero-copy memory.\n");
        	_CUDA(cudaHostAlloc((void**)&idea_host_data,buffer_size,cudaHostAllocMapped));
		transferHostToDevice = transferHostToDevice_ZEROCOPY;		// set memory transfer function
		transferDeviceToHost = transferDeviceToHost_ZEROCOPY;		// set memory transfer function
		_CUDA(cudaHostGetDevicePointer(&idea_device_data,idea_host_data, 0));
	} else {
		//pinned memory mode - use special function to get OS-pinned memory
		_CUDA(cudaHostAlloc( (void**)&idea_host_data, buffer_size, cudaHostAllocDefault));
		if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"Using pinned memory: cudaHostAllocDefault.\n");
		transferHostToDevice = transferHostToDevice_PINNED;	// set memory transfer function
		transferDeviceToHost = transferDeviceToHost_PINNED;	// set memory transfer function
		_CUDA(cudaMalloc((void **)&idea_device_data,buffer_size));
	}
#else
        //pinned memory mode - use special function to get OS-pinned memory
        _CUDA(cudaMallocHost((void**)&h_s, buffer_size));
        if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"Using pinned memory: cudaHostAllocDefault.\n");
	transferHostToDevice = transferHostToDevice_PINNED;			// set memory transfer function
	transferDeviceToHost = transferDeviceToHost_PINNED;			// set memory transfer function
	_CUDA(cudaMalloc((void **)&idea_device_data,buffer_size));
#endif
#else
        if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"Using pageable memory.\n");
	transferHostToDevice = transferHostToDevice_PAGEABLE;			// set memory transfer function
	transferDeviceToHost = transferDeviceToHost_PAGEABLE;			// set memory transfer function
	_CUDA(cudaMalloc((void **)&idea_device_data,buffer_size));
#endif

	if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"The current buffer size is %d.\n\n", buffer_size);

	_CUDA(cudaEventCreate(&idea_start));
	_CUDA(cudaEventCreate(&idea_stop));
	_CUDA(cudaEventRecord(idea_start,0));

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
