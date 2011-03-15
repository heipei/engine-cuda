// vim:ft=opencl
#ifndef MAX_THREAD
	#define MAX_THREAD	256
#endif
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

// Split uint64_t into two uint32_t and convert each from BE to LE
#define nl2i(s,a,b)      a = ((s >> 24L) & 0x000000ff) | \
			     ((s >> 8L ) & 0x0000ff00) | \
			     ((s << 8L ) & 0x00ff0000) | \
			     ((s << 24L) & 0xff000000),   \
			 b = ((s >> 56L) & 0x000000ff) | \
			     ((s >> 40L) & 0x0000ff00) | \
			     ((s >> 24L) & 0x00ff0000) | \
			     ((s >> 8L) & 0xff000000)

// Convert uint64_t endianness
#define flip64(a)	(a = \
			a << 56 | \
			((a & 0x000000000000FF00) << 40) | \
			((a & 0x0000000000FF0000) << 24) | \
			((a & 0x00000000FF000000) << 8)  | \
			((a & 0x000000FF00000000) >> 8)  | \
			((a & 0x0000FF0000000000) >> 24) | \
			((a & 0x00FF000000000000) >> 40) | \
			a >> 56)

// ###########
// # BF ECB #
// ###########

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
__kernel void BFencKernel(__global unsigned long *data, __global unsigned int *p) {
	__private unsigned int l, r;
	__private unsigned long block = data[get_global_id(0)];
	
	nl2i(block,l,r);

	__global unsigned int *s=p+18;

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

// ###########
// # DES ECB #
// ###########
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

__kernel void DESencKernel(__global unsigned long *data, __local unsigned char *des_SP, __local unsigned long *s, __global unsigned int *des_d_sp_c, __global unsigned long *cs) {
	
	if(get_local_id(0) < 16)
		s[get_local_id(0)] = cs[get_local_id(0)];

	// Careful: Based on the assumption of a constant 128 threads!
	// What happens for kernel calls with less than 128 threads, like the final padding 8-byte call?
	// It seems to work, but might be because of a strange race condition. Watch out!
	((__local ulong *)des_SP)[get_local_id(0)] = ((__global ulong *)des_d_sp_c)[get_local_id(0)];
	#if MAX_THREAD == 128
		((__local ulong *)des_SP)[get_local_id(0)+128] = ((__global ulong *)des_d_sp_c)[get_local_id(0)+128];
	#endif

	__private unsigned long load = data[get_global_id(0)];
	__private unsigned int right = load;
	__private unsigned int left = load>>32;
	
	__private unsigned int t,u;

	IP(right,left);

	left=ROTATE(left,29);
	right=ROTATE(right,29);

	barrier(CLK_LOCAL_MEM_FENCE);

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

// #############
// # CAST5 ECB #
// #############
#define ROTL(a,n)     ((((a)<<(n))&0xffffffffL)|((a)>>(32-(n))))

#define C_M    0x3fc
#define C_0    22L
#define C_1    14L
#define C_2     6L
#define C_3     2L /* left shift */

#define E_CAST(n,key,L,R,OP1,OP2,OP3) \
	{ \
	unsigned int a,b,c,d; \
	t=(key[n*2] OP1 R)&0xffffffff; \
	t=ROTL(t,(key[n*2+1])); \
	a=CAST_S_table0[(t>> 8)&0xff]; \
	b=CAST_S_table1[(t    )&0xff]; \
	c=CAST_S_table2[(t>>24)&0xff]; \
	d=CAST_S_table3[(t>>16)&0xff]; \
	L^=(((((a OP2 b)&0xffffffffL) OP3 c)&0xffffffffL) OP1 d)&0xffffffffL; \
	}

__kernel void CASTencKernel(__global unsigned long *data, __constant unsigned int *k, __global unsigned int *CAST_S_table) {
	__local unsigned int CAST_S_table0[256], CAST_S_table1[256], CAST_S_table2[256], CAST_S_table3[256];
	__private unsigned int l,r,t;

	__private unsigned long block = data[get_global_id(0)];

	nl2i(block,l,r);

	#if MAX_THREAD == 128
		((__local ulong *)CAST_S_table0)[get_local_id(0)] = ((__global ulong *)CAST_S_table)[get_local_id(0)];
		((__local ulong *)CAST_S_table2)[get_local_id(0)] = ((__global ulong *)CAST_S_table)[get_local_id(0)+128];
	#elif MAX_THREAD == 256
		((__local uint *)CAST_S_table0)[get_local_id(0)] = ((__global uint *)CAST_S_table)[get_local_id(0)];
		((__local uint *)CAST_S_table1)[get_local_id(0)] = ((__global uint *)CAST_S_table)[get_local_id(0)+256];
		((__local uint *)CAST_S_table2)[get_local_id(0)] = ((__global uint *)CAST_S_table)[get_local_id(0)+512];
		((__local uint *)CAST_S_table3)[get_local_id(0)] = ((__global uint *)CAST_S_table)[get_local_id(0)+768];
	#endif

	barrier(CLK_LOCAL_MEM_FENCE);

	E_CAST( 0,k,l,r,+,^,-);
	E_CAST( 1,k,r,l,^,-,+);
	E_CAST( 2,k,l,r,-,+,^);
	E_CAST( 3,k,r,l,+,^,-);
	E_CAST( 4,k,l,r,^,-,+);
	E_CAST( 5,k,r,l,-,+,^);
	E_CAST( 6,k,l,r,+,^,-);
	E_CAST( 7,k,r,l,^,-,+);
	E_CAST( 8,k,l,r,-,+,^);
	E_CAST( 9,k,r,l,+,^,-);
	E_CAST(10,k,l,r,^,-,+);
	E_CAST(11,k,r,l,-,+,^);
	E_CAST(12,k,l,r,+,^,-);
	E_CAST(13,k,r,l,^,-,+);
	E_CAST(14,k,l,r,-,+,^);
	E_CAST(15,k,r,l,+,^,-);

	block = ((unsigned long)r) << 32 | l;

	flip64(block);
	data[get_global_id(0)] = block;
}

#define SBOX1_1110 Camellia_SBOX[0]
#define SBOX4_4404 Camellia_SBOX[1]
#define SBOX2_0222 Camellia_SBOX[2]
#define SBOX3_3033 Camellia_SBOX[3]

#define SWAP(x) ((LeftRotate(x,8) & 0x00ff00ff) | (RightRotate(x,8) & 0xff00ff00))
#define RightRotate(x, s) ( ((x) >> (s)) + ((x) << (32 - s)) )
#define LeftRotate(x, s)  ( ((x) << (s)) + ((x) >> (32 - s)) )
#define GETU32(p)   SWAP(*((unsigned int *)(p)))
#define PUTU32(p,v) (*((unsigned int *)(p)) = SWAP((v)))

#define Camellia_Feistel(_s0,_s1,_s2,_s3,_key) do {\
	unsigned int _t0,_t1,_t2,_t3;\
\
	_t0  = _s0 ^ (_key)[1];\
	_t3  = SBOX4_4404[_t0&0xff];\
	_t1  = _s1 ^ (_key)[0];\
	_t3 ^= SBOX3_3033[(_t0 >> 8)&0xff];\
	_t2  = SBOX1_1110[_t1&0xff];\
	_t3 ^= SBOX2_0222[(_t0 >> 16)&0xff];\
	_t2 ^= SBOX4_4404[(_t1 >> 8)&0xff];\
	_t3 ^= SBOX1_1110[(_t0 >> 24)];\
	_t2 ^= _t3;\
	_t3  = RightRotate(_t3,8);\
	_t2 ^= SBOX3_3033[(_t1 >> 16)&0xff];\
	_s3 ^= _t3;\
	_t2 ^= SBOX2_0222[(_t1 >> 24)];\
	_s2 ^= _t2; \
	_s3 ^= _t2;\
} while(0)

__kernel void CMLLencKernel(__global unsigned long *data, __constant unsigned int *k, __global unsigned int *Camellia_global_SBOX) {
	__local unsigned int Camellia_SBOX[4][256];

	#if MAX_THREAD == 128
		((__local ulong *)Camellia_SBOX[0])[get_local_id(0)] = ((__global ulong *)Camellia_global_SBOX)[get_local_id(0)];
		((__local ulong *)Camellia_SBOX[1])[get_local_id(0)] = ((__global ulong *)Camellia_global_SBOX)[get_local_id(0)+128];
		((__local ulong *)Camellia_SBOX[2])[get_local_id(0)] = ((__global ulong *)Camellia_global_SBOX)[get_local_id(0)+256];
		((__local ulong *)Camellia_SBOX[3])[get_local_id(0)] = ((__global ulong *)Camellia_global_SBOX)[get_local_id(0)+384];
	#elif MAX_THREAD == 256
		((__local uint *)Camellia_SBOX[0])[get_local_id(0)] = ((__global uint *)Camellia_global_SBOX)[get_local_id(0)];
		((__local uint *)Camellia_SBOX[1])[get_local_id(0)] = ((__global uint *)Camellia_global_SBOX)[get_local_id(0)+256];
		((__local uint *)Camellia_SBOX[2])[get_local_id(0)] = ((__global uint *)Camellia_global_SBOX)[get_local_id(0)+512];
		((__local uint *)Camellia_SBOX[3])[get_local_id(0)] = ((__global uint *)Camellia_global_SBOX)[get_local_id(0)+768];
	#endif

	__private unsigned int s0,s1,s2,s3; 

	__private unsigned long block = data[get_global_id(0)*2];
	nl2i(block,s0,s1);
	s1 ^= k[0];
	s0 ^= k[1];

	block = data[(get_global_id(0)*2)+1];
	nl2i(block,s2,s3);
	s3 ^= k[2];
	s2 ^= k[3];

	k += 4;

	barrier(CLK_LOCAL_MEM_FENCE);

	Camellia_Feistel(s0,s1,s2,s3,k+0);
	Camellia_Feistel(s2,s3,s0,s1,k+2);
	Camellia_Feistel(s0,s1,s2,s3,k+4);
	Camellia_Feistel(s2,s3,s0,s1,k+6);
	Camellia_Feistel(s0,s1,s2,s3,k+8);
	Camellia_Feistel(s2,s3,s0,s1,k+10);
	k += 12;

	s1 ^= LeftRotate(s0 & k[1], 1);
	s2 ^= s3 | k[2];
	s0 ^= s1 | k[0];
	s3 ^= LeftRotate(s2 & k[3], 1);
	k += 4;

	Camellia_Feistel(s0,s1,s2,s3,k+0);
	Camellia_Feistel(s2,s3,s0,s1,k+2);
	Camellia_Feistel(s0,s1,s2,s3,k+4);
	Camellia_Feistel(s2,s3,s0,s1,k+6);
	Camellia_Feistel(s0,s1,s2,s3,k+8);
	Camellia_Feistel(s2,s3,s0,s1,k+10);
	k += 12;

	s1 ^= LeftRotate(s0 & k[1], 1);
	s2 ^= s3 | k[2];
	s0 ^= s1 | k[0];
	s3 ^= LeftRotate(s2 & k[3], 1);
	k += 4;

	Camellia_Feistel(s0,s1,s2,s3,k+0);
	Camellia_Feistel(s2,s3,s0,s1,k+2);
	Camellia_Feistel(s0,s1,s2,s3,k+4);
	Camellia_Feistel(s2,s3,s0,s1,k+6);
	Camellia_Feistel(s0,s1,s2,s3,k+8);
	Camellia_Feistel(s2,s3,s0,s1,k+10);
	k += 12;

	s2 ^= k[1], s3 ^= k[0], s0 ^= k[3], s1 ^= k[2];

	block = ((unsigned long)s2) << 32 | s3;
	flip64(block);
	data[get_global_id(0)*2] = block;

	block = ((unsigned long)s0) << 32 | s1;
	flip64(block);
	data[(get_global_id(0)*2)+1] = block;
}

#define idea_mul(r,a,b,ul) \
	ul = mul24(a,b); \
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

// ############
// # IDEA ECB #
// ############

__kernel void IDEAencKernel(__global unsigned long *data, __constant unsigned long *idea_constant_schedule) {
	
	__local unsigned long idea_schedule[27];
	if(get_local_id(0) < 27)
		idea_schedule[get_local_id(0)] = idea_constant_schedule[get_local_id(0)];

	__local unsigned int *p = (__local unsigned int *)&idea_schedule;

	__private unsigned int x1,x2,x3,x4,t0,t1,ul,l0,l1;

	__private unsigned long block = data[get_global_id(0)];

	nl2i(block,x2,x4);

	x1=(x2>>16);
	x3=(x4>>16);

	barrier(CLK_LOCAL_MEM_FENCE);

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

	block = ((unsigned long)l0) << 32 | l1;
	flip64(block);
	data[get_global_id(0)] = block;
}
