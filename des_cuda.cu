#ifndef __DEVICE_EMULATION__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <openssl/des.h>
#include "cuda_common.h"
#include "common.h"

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
	
	//if(threadIdx.x < 16)
		s[threadIdx.x%16] = cs[threadIdx.x%16];
	//__syncthreads();

	// Careful: Based on the assumption of a constant 128 threads!
	// What happens for kernel calls with less than 128 threads, like the final padding 8-byte call?
	// It seems to work, but might be because of a strange race condition. Watch out!
	((uint64_t *)des_d_sp)[threadIdx.x] = ((uint64_t *)des_d_sp_c)[threadIdx.x];
	((uint64_t *)des_d_sp)[threadIdx.x+128] = ((uint64_t *)des_d_sp_c)[threadIdx.x+128];

	register uint64_t load = data[TX];
	register uint32_t right = load;
	register uint32_t left = load >> 32;
	
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
	load = left|((uint64_t)right)<<32;
	data[TX]=load;
}

__global__ void DESdecKernel(uint64_t *data) {
	
	if(threadIdx.x < 16)
		s[threadIdx.x] = cs[threadIdx.x];

	((uint64_t *)des_d_sp)[threadIdx.x] = ((uint64_t *)des_d_sp_c)[threadIdx.x];
	((uint64_t *)des_d_sp)[threadIdx.x+128] = ((uint64_t *)des_d_sp_c)[threadIdx.x+128];

	//uint64_t load = data[TX];
	uint32_t right = data[TX];
	uint32_t left = data[TX]>>32;

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

extern "C" void DES_cuda_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, int enc, uint8_t **host_data, uint64_t **device_data) {
	assert(in && out && nbytes);
	cudaError_t cudaerrno;
	int gridSize;

	transferHostToDevice(&in, (uint32_t **)device_data, host_data, &nbytes);
	
	if ((nbytes%(MAX_THREAD*DES_BLOCK_SIZE))==0) {
		gridSize = nbytes/(MAX_THREAD*DES_BLOCK_SIZE);
	} else {
		gridSize = nbytes/(MAX_THREAD*DES_BLOCK_SIZE)+1;
	}

	#ifdef DEBUG
		fprintf(stdout,"Starting DES kernel for %zu bytes with (%d, (%d))...\n", nbytes, gridSize, MAX_THREAD);
	#endif

	if(enc == DES_ENCRYPT) {
		DESencKernel<<<gridSize,MAX_THREAD>>>(*device_data);
		_CUDA_N("DES encryption kernel could not be launched!");
	} else {
		DESdecKernel<<<gridSize,MAX_THREAD>>>(*device_data);
		_CUDA_N("DES decryption kernel could not be launched!");
	}

	transferDeviceToHost(&out, (uint32_t **)device_data, host_data, host_data, &nbytes);
}

extern "C" void DES_cuda_transfer_key_schedule(DES_key_schedule *ks) {
	cudaError_t cudaerrno;
	_CUDA(cudaMemcpyToSymbolAsync(cs,ks,sizeof(DES_key_schedule),0,cudaMemcpyHostToDevice));
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
