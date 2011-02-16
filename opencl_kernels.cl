// vim:ft=opencl
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#include <openssl/blowfish.h>

#define BF_ENC(LL,R,S,P) ( \
	LL^=P, \
	LL^=(((	S[       ((uint)(R>>24)&0xff)] + \
		S[0x0100+((uint)(R>>16)&0xff)])^ \
		S[0x0200+((uint)(R>> 8)&0xff)])+ \
		S[0x0300+((uint)(R    )&0xff)]) \
	)

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

