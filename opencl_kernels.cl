// vim:ft=opencl
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define n2l(c,l)        (l =((uint)(*(c)))<<24L, \
                         l|=((uint)(*(c+1)))<<16L, \
                         l|=((uint)(*(c+2)))<< 8L, \
                         l|=((uint)(*(c+3))))

#define flip64(a)	(a= \
			((a & 0x00000000000000FF) << 56) | \
			((a & 0x000000000000FF00) << 40) | \
			((a & 0x0000000000FF0000) << 24) | \
			((a & 0x00000000FF000000) << 8)  | \
			((a & 0x000000FF00000000) >> 8)  | \
			((a & 0x0000FF0000000000) >> 24) | \
			((a & 0x00FF000000000000) >> 40) | \
			((a & 0xFF00000000000000) >> 56))

#include <openssl/blowfish.h>

#define BF_ENC(LL,R,S,P) ( \
	LL^=P, \
	LL^=(((	S[       ((uint)(R>>24)&0xff)] + \
		S[0x0100+((uint)(R>>16)&0xff)])^ \
		S[0x0200+((uint)(R>> 8)&0xff)])+ \
		S[0x0300+((uint)(R    )&0xff)]) \
	)


// TODO: When using __constant memory for the key-schedule, each call to BF_ENC
// increases the compilation-time by at least factor 2. Bug "filed" in NVIDIA
// forums
__kernel void BFencKernel(__global unsigned long *data, __global unsigned int *bf_constant_schedule) {
	__private unsigned int l, r;
	__private unsigned long block = data[get_global_id(0)];
	
	n2l((unsigned char *)&block,l);
	n2l(((unsigned char *)&block)+4,r);

	__global unsigned int *p=bf_constant_schedule;
	__global unsigned int *s=bf_constant_schedule+18;

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
	r^=p[17];

	block = ((unsigned long)r) << 32 | l;
	flip64(block);
	data[get_global_id(0)] = block;
}

#define IP(left,right) \
	{ \
	register unsigned int tt; \
	PERM_OP(right,left,tt, 4,0x0f0f0f0fL); \
	PERM_OP(left,right,tt,16,0x0000ffffL); \
	PERM_OP(right,left,tt, 2,0x33333333L); \
	PERM_OP(left,right,tt, 8,0x00ff00ffL); \
	PERM_OP(right,left,tt, 1,0x55555555L); \
	}

#define FP(left,right) \
	{ \
	register unsigned int tt; \
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
	__private unsigned long ss = s[S]; \
	u=R^ss; \
	t=R^ss>>32; \
	t=ROTATE(t,4); \
	LL^= \
	*(__local unsigned int *)(des_SP      +((u     )&0xfc))^ \
	*(__local unsigned int *)(des_SP+0x200+((u>> 8L)&0xfc))^ \
	*(__local unsigned int *)(des_SP+0x400+((u>>16L)&0xfc))^ \
	*(__local unsigned int *)(des_SP+0x600+((u>>24L)&0xfc))^ \
	*(__local unsigned int *)(des_SP+0x100+((t     )&0xfc))^ \
	*(__local unsigned int *)(des_SP+0x300+((t>> 8L)&0xfc))^ \
	*(__local unsigned int *)(des_SP+0x500+((t>>16L)&0xfc))^ \
	*(__local unsigned int *)(des_SP+0x700+((t>>24L)&0xfc)); }

__kernel void DESencKernel(__global unsigned long *data, __local unsigned int *des_d_sp, __local unsigned long *s, __global unsigned int *des_d_sp_c, __global unsigned long *cs) {
	
	//if(get_local_id(0) < 16)
		s[get_local_id(0)%16] = cs[get_local_id(0)%16];

	// Careful: Based on the assumption of a constant 128 threads!
	// What happens for kernel calls with less than 128 threads, like the final padding 8-byte call?
	// It seems to work, but might be because of a strange race condition. Watch out!
	des_d_sp[get_local_id(0)] = des_d_sp_c[get_local_id(0)];
	des_d_sp[get_local_id(0)+128] = des_d_sp_c[get_local_id(0)+128];
	des_d_sp[get_local_id(0)+256] = des_d_sp_c[get_local_id(0)+256];
	des_d_sp[get_local_id(0)+384] = des_d_sp_c[get_local_id(0)+384];

	barrier(CLK_LOCAL_MEM_FENCE);
	__private unsigned long load = data[get_global_id(0)];
	__private unsigned int right = load;
	__private unsigned int left = load>>32;
	
	__private unsigned int t,u;
	__local unsigned char *des_SP = (__local unsigned char *) des_d_sp;

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
	data[get_global_id(0)]=left|((unsigned long)right)<<32;
}

//__global void DESdecKernel(__global unsigned long *data) {
//	
//	if(get_local_id(0) < 16)
//		s[get_local_id(0)] = cs[get_local_id(0)];
//
//	((unsigned int *)des_d_sp)[get_local_id(0)] = ((unsigned int *)des_d_sp_c)[get_local_id(0)];
//	((unsigned int *)des_d_sp)[get_local_id(0)+128] = ((unsigned int *)des_d_sp_c)[get_local_id(0)+128];
//	((unsigned int *)des_d_sp)[get_local_id(0)+256] = ((unsigned int *)des_d_sp_c)[get_local_id(0)+256];
//	((unsigned int *)des_d_sp)[get_local_id(0)+384] = ((unsigned int *)des_d_sp_c)[get_local_id(0)+384];
//
//	unsigned long load = data[get_global_id(0)];
//	unsigned int right = load;
//	unsigned int left = load>>32;
//
//	unsigned int t,u;
//	unsigned char *des_SP = (unsigned char *) (&des_d_sp);
//
//	IP(right,left);
//
//	left=ROTATE(left,29);
//	right=ROTATE(right,29);
//
//	D_ENCRYPT(left,right,15); /*  16 */
//	D_ENCRYPT(right,left,14); /*  15 */
//	D_ENCRYPT(left,right,13); /*  14 */
//	D_ENCRYPT(right,left,12); /*  13 */
//	D_ENCRYPT(left,right,11); /*  12 */
//	D_ENCRYPT(right,left,10); /*  11 */
//	D_ENCRYPT(left,right, 9); /*  10 */
//	D_ENCRYPT(right,left, 8); /*  9 */
//	D_ENCRYPT(left,right, 7); /*  8 */
//	D_ENCRYPT(right,left, 6); /*  7 */
//	D_ENCRYPT(left,right, 5); /*  6 */
//	D_ENCRYPT(right,left, 4); /*  5 */
//	D_ENCRYPT(left,right, 3); /*  4 */
//	D_ENCRYPT(right,left, 2); /*  3 */
//	D_ENCRYPT(left,right, 1); /*  2 */
//	D_ENCRYPT(right,left, 0); /*  1 */
//
//	left=ROTATE(left,3);
//	right=ROTATE(right,3);
//
//	FP(right,left);
//	data[get_global_id(0)]=left|((unsigned long)right)<<32;
//}
