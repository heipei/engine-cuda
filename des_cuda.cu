#ifndef __DEVICE_EMULATION__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <openssl/des.h>
#include "cuda_common.h"
#include "common.h"
#include "lib/cuPrintf.cu"

__constant__ uint32_t des_d_sp_c[8][64]={
{
/* nibble 0 */
0x02080800L, 0x00080000L, 0x02000002L, 0x02080802L,
0x02000000L, 0x00080802L, 0x00080002L, 0x02000002L,
0x00080802L, 0x02080800L, 0x02080000L, 0x00000802L,
0x02000802L, 0x02000000L, 0x00000000L, 0x00080002L,
0x00080000L, 0x00000002L, 0x02000800L, 0x00080800L,
0x02080802L, 0x02080000L, 0x00000802L, 0x02000800L,
0x00000002L, 0x00000800L, 0x00080800L, 0x02080002L,
0x00000800L, 0x02000802L, 0x02080002L, 0x00000000L,
0x00000000L, 0x02080802L, 0x02000800L, 0x00080002L,
0x02080800L, 0x00080000L, 0x00000802L, 0x02000800L,
0x02080002L, 0x00000800L, 0x00080800L, 0x02000002L,
0x00080802L, 0x00000002L, 0x02000002L, 0x02080000L,
0x02080802L, 0x00080800L, 0x02080000L, 0x02000802L,
0x02000000L, 0x00000802L, 0x00080002L, 0x00000000L,
0x00080000L, 0x02000000L, 0x02000802L, 0x02080800L,
0x00000002L, 0x02080002L, 0x00000800L, 0x00080802L,
},{
/* nibble 1 */
0x40108010L, 0x00000000L, 0x00108000L, 0x40100000L,
0x40000010L, 0x00008010L, 0x40008000L, 0x00108000L,
0x00008000L, 0x40100010L, 0x00000010L, 0x40008000L,
0x00100010L, 0x40108000L, 0x40100000L, 0x00000010L,
0x00100000L, 0x40008010L, 0x40100010L, 0x00008000L,
0x00108010L, 0x40000000L, 0x00000000L, 0x00100010L,
0x40008010L, 0x00108010L, 0x40108000L, 0x40000010L,
0x40000000L, 0x00100000L, 0x00008010L, 0x40108010L,
0x00100010L, 0x40108000L, 0x40008000L, 0x00108010L,
0x40108010L, 0x00100010L, 0x40000010L, 0x00000000L,
0x40000000L, 0x00008010L, 0x00100000L, 0x40100010L,
0x00008000L, 0x40000000L, 0x00108010L, 0x40008010L,
0x40108000L, 0x00008000L, 0x00000000L, 0x40000010L,
0x00000010L, 0x40108010L, 0x00108000L, 0x40100000L,
0x40100010L, 0x00100000L, 0x00008010L, 0x40008000L,
0x40008010L, 0x00000010L, 0x40100000L, 0x00108000L,
},{
/* nibble 2 */
0x04000001L, 0x04040100L, 0x00000100L, 0x04000101L,
0x00040001L, 0x04000000L, 0x04000101L, 0x00040100L,
0x04000100L, 0x00040000L, 0x04040000L, 0x00000001L,
0x04040101L, 0x00000101L, 0x00000001L, 0x04040001L,
0x00000000L, 0x00040001L, 0x04040100L, 0x00000100L,
0x00000101L, 0x04040101L, 0x00040000L, 0x04000001L,
0x04040001L, 0x04000100L, 0x00040101L, 0x04040000L,
0x00040100L, 0x00000000L, 0x04000000L, 0x00040101L,
0x04040100L, 0x00000100L, 0x00000001L, 0x00040000L,
0x00000101L, 0x00040001L, 0x04040000L, 0x04000101L,
0x00000000L, 0x04040100L, 0x00040100L, 0x04040001L,
0x00040001L, 0x04000000L, 0x04040101L, 0x00000001L,
0x00040101L, 0x04000001L, 0x04000000L, 0x04040101L,
0x00040000L, 0x04000100L, 0x04000101L, 0x00040100L,
0x04000100L, 0x00000000L, 0x04040001L, 0x00000101L,
0x04000001L, 0x00040101L, 0x00000100L, 0x04040000L,
},{
/* nibble 3 */
0x00401008L, 0x10001000L, 0x00000008L, 0x10401008L,
0x00000000L, 0x10400000L, 0x10001008L, 0x00400008L,
0x10401000L, 0x10000008L, 0x10000000L, 0x00001008L,
0x10000008L, 0x00401008L, 0x00400000L, 0x10000000L,
0x10400008L, 0x00401000L, 0x00001000L, 0x00000008L,
0x00401000L, 0x10001008L, 0x10400000L, 0x00001000L,
0x00001008L, 0x00000000L, 0x00400008L, 0x10401000L,
0x10001000L, 0x10400008L, 0x10401008L, 0x00400000L,
0x10400008L, 0x00001008L, 0x00400000L, 0x10000008L,
0x00401000L, 0x10001000L, 0x00000008L, 0x10400000L,
0x10001008L, 0x00000000L, 0x00001000L, 0x00400008L,
0x00000000L, 0x10400008L, 0x10401000L, 0x00001000L,
0x10000000L, 0x10401008L, 0x00401008L, 0x00400000L,
0x10401008L, 0x00000008L, 0x10001000L, 0x00401008L,
0x00400008L, 0x00401000L, 0x10400000L, 0x10001008L,
0x00001008L, 0x10000000L, 0x10000008L, 0x10401000L,
},{
/* nibble 4 */
0x08000000L, 0x00010000L, 0x00000400L, 0x08010420L,
0x08010020L, 0x08000400L, 0x00010420L, 0x08010000L,
0x00010000L, 0x00000020L, 0x08000020L, 0x00010400L,
0x08000420L, 0x08010020L, 0x08010400L, 0x00000000L,
0x00010400L, 0x08000000L, 0x00010020L, 0x00000420L,
0x08000400L, 0x00010420L, 0x00000000L, 0x08000020L,
0x00000020L, 0x08000420L, 0x08010420L, 0x00010020L,
0x08010000L, 0x00000400L, 0x00000420L, 0x08010400L,
0x08010400L, 0x08000420L, 0x00010020L, 0x08010000L,
0x00010000L, 0x00000020L, 0x08000020L, 0x08000400L,
0x08000000L, 0x00010400L, 0x08010420L, 0x00000000L,
0x00010420L, 0x08000000L, 0x00000400L, 0x00010020L,
0x08000420L, 0x00000400L, 0x00000000L, 0x08010420L,
0x08010020L, 0x08010400L, 0x00000420L, 0x00010000L,
0x00010400L, 0x08010020L, 0x08000400L, 0x00000420L,
0x00000020L, 0x00010420L, 0x08010000L, 0x08000020L,
},{
/* nibble 5 */
0x80000040L, 0x00200040L, 0x00000000L, 0x80202000L,
0x00200040L, 0x00002000L, 0x80002040L, 0x00200000L,
0x00002040L, 0x80202040L, 0x00202000L, 0x80000000L,
0x80002000L, 0x80000040L, 0x80200000L, 0x00202040L,
0x00200000L, 0x80002040L, 0x80200040L, 0x00000000L,
0x00002000L, 0x00000040L, 0x80202000L, 0x80200040L,
0x80202040L, 0x80200000L, 0x80000000L, 0x00002040L,
0x00000040L, 0x00202000L, 0x00202040L, 0x80002000L,
0x00002040L, 0x80000000L, 0x80002000L, 0x00202040L,
0x80202000L, 0x00200040L, 0x00000000L, 0x80002000L,
0x80000000L, 0x00002000L, 0x80200040L, 0x00200000L,
0x00200040L, 0x80202040L, 0x00202000L, 0x00000040L,
0x80202040L, 0x00202000L, 0x00200000L, 0x80002040L,
0x80000040L, 0x80200000L, 0x00202040L, 0x00000000L,
0x00002000L, 0x80000040L, 0x80002040L, 0x80202000L,
0x80200000L, 0x00002040L, 0x00000040L, 0x80200040L,
},{
/* nibble 6 */
0x00004000L, 0x00000200L, 0x01000200L, 0x01000004L,
0x01004204L, 0x00004004L, 0x00004200L, 0x00000000L,
0x01000000L, 0x01000204L, 0x00000204L, 0x01004000L,
0x00000004L, 0x01004200L, 0x01004000L, 0x00000204L,
0x01000204L, 0x00004000L, 0x00004004L, 0x01004204L,
0x00000000L, 0x01000200L, 0x01000004L, 0x00004200L,
0x01004004L, 0x00004204L, 0x01004200L, 0x00000004L,
0x00004204L, 0x01004004L, 0x00000200L, 0x01000000L,
0x00004204L, 0x01004000L, 0x01004004L, 0x00000204L,
0x00004000L, 0x00000200L, 0x01000000L, 0x01004004L,
0x01000204L, 0x00004204L, 0x00004200L, 0x00000000L,
0x00000200L, 0x01000004L, 0x00000004L, 0x01000200L,
0x00000000L, 0x01000204L, 0x01000200L, 0x00004200L,
0x00000204L, 0x00004000L, 0x01004204L, 0x01000000L,
0x01004200L, 0x00000004L, 0x00004004L, 0x01004204L,
0x01000004L, 0x01004200L, 0x01004000L, 0x00004004L,
},{
/* nibble 7 */
0x20800080L, 0x20820000L, 0x00020080L, 0x00000000L,
0x20020000L, 0x00800080L, 0x20800000L, 0x20820080L,
0x00000080L, 0x20000000L, 0x00820000L, 0x00020080L,
0x00820080L, 0x20020080L, 0x20000080L, 0x20800000L,
0x00020000L, 0x00820080L, 0x00800080L, 0x20020000L,
0x20820080L, 0x20000080L, 0x00000000L, 0x00820000L,
0x20000000L, 0x00800000L, 0x20020080L, 0x20800080L,
0x00800000L, 0x00020000L, 0x20820000L, 0x00000080L,
0x00800000L, 0x00020000L, 0x20000080L, 0x20820080L,
0x00020080L, 0x20000000L, 0x00000000L, 0x00820000L,
0x20800080L, 0x20020080L, 0x20020000L, 0x00800080L,
0x20820000L, 0x00000080L, 0x00800080L, 0x20020000L,
0x20820080L, 0x00800000L, 0x20800000L, 0x20000080L,
0x00820000L, 0x00020080L, 0x20020080L, 0x20800000L,
0x00000080L, 0x20820000L, 0x00820080L, 0x00000000L,
0x20000000L, 0x20800080L, 0x00020000L, 0x00820080L,
}};

__shared__ uint32_t des_d_sp[8][64];

__constant__ uint64_t cs[16];
__shared__ uint64_t s[16];

__device__ __constant__ uint64_t *des_d_s;
uint8_t  *des_h_s;
//__device__ uint32_t *des_d_iv;

float des_elapsed;
cudaEvent_t des_start,des_stop;

#define IP(left,right) \
	{ \
	register uint32_t tt; \
	PERM_OP(right,left,tt, 4,0x0f0f0f0fL); \
	PERM_OP(left,right,tt,16,0x0000ffffL); \
	PERM_OP(right,left,tt, 2,0x33333333L); \
	PERM_OP(left,right,tt, 8,0x00ff00ffL); \
	PERM_OP(right,left,tt, 1,0x55555555L); \
	}

#define FP(left,right) \
	{ \
	register uint32_t tt; \
	PERM_OP(left,right,tt, 1,0x55555555L); \
	PERM_OP(right,left,tt, 8,0x00ff00ffL); \
	PERM_OP(left,right,tt, 2,0x33333333L); \
	PERM_OP(right,left,tt,16,0x0000ffffL); \
	PERM_OP(left,right,tt, 4,0x0f0f0f0fL); \
	}

#define	ROTATE(a,n)	(((a)>>(n))|((a)<<(32-(n))))

#define PERM_OP(a,b,t,n,m) ((t)=((((a)>>(n))^(b))&(m)),\
	(b)^=(t),\
	(a)^=((t)<<(n)))

#define D_ENCRYPT(LL,R,S) { \
	register uint64_t ss = s[S]; \
	u=R^ss; \
	t=R^ss>>32; \
	t=ROTATE(t,4); \
	LL^= \
	*(const uint32_t *)(des_SP      +((u     )&0xfc))^ \
	*(const uint32_t *)(des_SP+0x200+((u>> 8L)&0xfc))^ \
	*(const uint32_t *)(des_SP+0x400+((u>>16L)&0xfc))^ \
	*(const uint32_t *)(des_SP+0x600+((u>>24L)&0xfc))^ \
	*(const uint32_t *)(des_SP+0x100+((t     )&0xfc))^ \
	*(const uint32_t *)(des_SP+0x300+((t>> 8L)&0xfc))^ \
	*(const uint32_t *)(des_SP+0x500+((t>>16L)&0xfc))^ \
	*(const uint32_t *)(des_SP+0x700+((t>>24L)&0xfc)); }

__global__ void DESencKernel(uint64_t *data) {
	
	if(threadIdx.x < 16)
		s[threadIdx.x] = cs[threadIdx.x];

	// Careful: Based on the assumption of a constant 128 threads!
	((uint32_t *)des_d_sp)[threadIdx.x] = ((uint32_t *)des_d_sp_c)[threadIdx.x];
	((uint32_t *)des_d_sp)[threadIdx.x+128] = ((uint32_t *)des_d_sp_c)[threadIdx.x+128];
	((uint32_t *)des_d_sp)[threadIdx.x+256] = ((uint32_t *)des_d_sp_c)[threadIdx.x+256];
	((uint32_t *)des_d_sp)[threadIdx.x+384] = ((uint32_t *)des_d_sp_c)[threadIdx.x+384];

	uint64_t load = data[TX];
	uint32_t right = load;
	uint32_t left = load>>32;
	
	unsigned int t,u;
	unsigned char *des_SP = (unsigned char *) (&des_d_sp);

	IP(right,left);

	left=ROTATE(left,29);
	right=ROTATE(right,29);

	D_ENCRYPT(left,right, 0);
	D_ENCRYPT(right,left, 1);
	D_ENCRYPT(left,right, 2);
	D_ENCRYPT(right,left, 3);
	D_ENCRYPT(left,right, 4);
	D_ENCRYPT(right,left, 5);
	D_ENCRYPT(left,right, 6);
	D_ENCRYPT(right,left, 7);
	D_ENCRYPT(left,right, 8);
	D_ENCRYPT(right,left, 9);
	D_ENCRYPT(left,right,10);
	D_ENCRYPT(right,left,11);
	D_ENCRYPT(left,right,12);
	D_ENCRYPT(right,left,13);
	D_ENCRYPT(left,right,14);
	D_ENCRYPT(right,left,15);

	left=ROTATE(left,3);
	right=ROTATE(right,3);

	FP(right,left);
	data[TX]=left|((uint64_t)right)<<32;
}

__global__ void DESdecKernel(uint64_t *data) {
	
	if(threadIdx.x < 16)
		s[threadIdx.x] = cs[threadIdx.x];

	((uint32_t *)des_d_sp)[threadIdx.x] = ((uint32_t *)des_d_sp_c)[threadIdx.x];
	((uint32_t *)des_d_sp)[threadIdx.x+128] = ((uint32_t *)des_d_sp_c)[threadIdx.x+128];
	((uint32_t *)des_d_sp)[threadIdx.x+256] = ((uint32_t *)des_d_sp_c)[threadIdx.x+256];
	((uint32_t *)des_d_sp)[threadIdx.x+384] = ((uint32_t *)des_d_sp_c)[threadIdx.x+384];

	uint64_t load = data[TX];
	uint32_t right = load;
	uint32_t left = load>>32;

	unsigned int t,u;
	unsigned char *des_SP = (unsigned char *) (&des_d_sp);

	IP(right,left);

	left=ROTATE(left,29);
	right=ROTATE(right,29);

	D_ENCRYPT(left,right,15); /*  16 */
	D_ENCRYPT(right,left,14); /*  15 */
	D_ENCRYPT(left,right,13); /*  14 */
	D_ENCRYPT(right,left,12); /*  13 */
	D_ENCRYPT(left,right,11); /*  12 */
	D_ENCRYPT(right,left,10); /*  11 */
	D_ENCRYPT(left,right, 9); /*  10 */
	D_ENCRYPT(right,left, 8); /*  9 */
	D_ENCRYPT(left,right, 7); /*  8 */
	D_ENCRYPT(right,left, 6); /*  7 */
	D_ENCRYPT(left,right, 5); /*  6 */
	D_ENCRYPT(right,left, 4); /*  5 */
	D_ENCRYPT(left,right, 3); /*  4 */
	D_ENCRYPT(right,left, 2); /*  3 */
	D_ENCRYPT(left,right, 1); /*  2 */
	D_ENCRYPT(right,left, 0); /*  1 */

	left=ROTATE(left,3);
	right=ROTATE(right,3);

	FP(right,left);
	data[TX]=left|((uint64_t)right)<<32;
}

extern "C" void DES_cuda_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, int enc) {
	assert(in && out && nbytes);
	cudaError_t cudaerrno;
	int gridSize;
	dim3 dimBlock(MAX_THREAD, 1, 1);

	transferHostToDevice(&in, (uint32_t **)&des_d_s, &des_h_s, &nbytes);

	if ((nbytes%(MAX_THREAD*DES_BLOCK_SIZE))==0) {
		gridSize = nbytes/(MAX_THREAD*2);
	} else {
		gridSize = nbytes/(MAX_THREAD*2)+1;
	}

	if (output_verbosity==OUTPUT_VERBOSE)
		fprintf(stdout,"Starting DES kernel with (%d, (%d, %d))...\n", gridSize, dimBlock.x, dimBlock.y);

	if(enc == DES_ENCRYPT) {
		DESencKernel<<<gridSize,dimBlock>>>(des_d_s);
		_CUDA_N("DES encryption kernel could not be launched!");
	} else {
		DESdecKernel<<<gridSize,dimBlock>>>(des_d_s);
		_CUDA_N("DES decryption kernel could not be launched!");
	}

	transferDeviceToHost(&out, (uint32_t **)&des_d_s, &des_h_s, &nbytes);
}

extern "C" void DES_cuda_transfer_key_schedule(DES_key_schedule *ks) {
	assert(ks);
	cudaError_t cudaerrno;
	size_t ks_size = sizeof(DES_key_schedule);
	_CUDA(cudaMemcpyToSymbolAsync(cs,ks,ks_size,0,cudaMemcpyHostToDevice));
}

extern "C" void DES_cuda_finish() {
	cudaError_t cudaerrno;

	if (output_verbosity>=OUTPUT_NORMAL) fprintf(stdout, "\nDone. Finishing up DES\n");

#ifndef PAGEABLE 
#if CUDART_VERSION >= 2020
	if(isIntegrated) {
		_CUDA(cudaFreeHost(des_h_s));
		//_CUDA(cudaFreeHost(des_h_iv));
	} else {
		_CUDA(cudaFree(des_d_s));
		//_CUDA(cudaFree(des_d_iv));
	}
#else	
	_CUDA(cudaFree(des_d_s));
	//_CUDA(cudaFree(des_d_iv));
#endif
#else
	_CUDA(cudaFree(des_d_s));
	//_CUDA(cudaFree(des_d_iv));
#endif	

	_CUDA(cudaEventRecord(des_stop,0));
	_CUDA(cudaEventSynchronize(des_stop));
	_CUDA(cudaEventElapsedTime(&des_elapsed,des_start,des_stop));

	if (output_verbosity>=OUTPUT_NORMAL) fprintf(stdout,"\nTotal time: %f milliseconds\n",des_elapsed);	
}

extern "C" void DES_cuda_init(int *nm, int buffer_size_engine, int output_kind) {
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
        	_CUDA(cudaHostAlloc((void**)&des_h_s,buffer_size,cudaHostAllocMapped));
		transferHostToDevice = transferHostToDevice_ZEROCOPY;		// set memory transfer function
		transferDeviceToHost = transferDeviceToHost_ZEROCOPY;		// set memory transfer function
		_CUDA(cudaHostGetDevicePointer(&des_d_s,des_h_s, 0));
	} else {
		//pinned memory mode - use special function to get OS-pinned memory
		_CUDA(cudaHostAlloc( (void**)&des_h_s, buffer_size, cudaHostAllocDefault));
		if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"Using pinned memory: cudaHostAllocDefault.\n");
		transferHostToDevice = transferHostToDevice_PINNED;	// set memory transfer function
		transferDeviceToHost = transferDeviceToHost_PINNED;	// set memory transfer function
		_CUDA(cudaMalloc((void **)&des_d_s,buffer_size));
	}
#else
        //pinned memory mode - use special function to get OS-pinned memory
        _CUDA(cudaMallocHost((void**)&h_s, buffer_size));
        if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"Using pinned memory: cudaHostAllocDefault.\n");
	transferHostToDevice = transferHostToDevice_PINNED;			// set memory transfer function
	transferDeviceToHost = transferDeviceToHost_PINNED;			// set memory transfer function
	_CUDA(cudaMalloc((void **)&des_d_s,buffer_size));
#endif
#else
        if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"Using pageable memory.\n");
	transferHostToDevice = transferHostToDevice_PAGEABLE;			// set memory transfer function
	transferDeviceToHost = transferDeviceToHost_PAGEABLE;			// set memory transfer function
	_CUDA(cudaMalloc((void **)&des_d_s,buffer_size));
#endif

	if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"The current buffer size is %d.\n\n", buffer_size);

	_CUDA(cudaEventCreate(&des_start));
	_CUDA(cudaEventCreate(&des_stop));
	_CUDA(cudaEventRecord(des_start,0));

}
//
// CBC parallel decrypt
//

__global__ void DESdecKernel_cbc(uint32_t in[],uint32_t out[],uint32_t iv[]) {
}

extern "C" void DES_cuda_transfer_iv(const unsigned char *iv) {
}

extern "C" void DES_cuda_decrypt_cbc(const unsigned char *in, unsigned char *out, size_t nbytes) {
}

#ifndef CBC_ENC_CPU
//
// CBC  encrypt
//

__global__ void DESencKernel_cbc(uint32_t state[],uint32_t iv[],size_t length) {
}

#endif

extern "C" void DES_cuda_encrypt_cbc(const unsigned char *in, unsigned char *out, size_t nbytes) {
}
#else
#error "ERROR: DEVICE EMULATION is NOT supported."
#endif
