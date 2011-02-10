// vim:ft=opencl

#define BF_M  (0xFF<<2)

// TODO: This will issue a lot of ld.global.u8 instructions!
#define BF_ENC(LL,R,S,P) ( \
	LL^=P, \
	LL^=(((*(S+   0+((R>>22)&BF_M)))<<0 | (*(S+   1+((R>>22)&BF_M)))<<8 | (*(S+   2+((R>>22)&BF_M)))<<16 | (*(S+   3+((R>>22)&BF_M)))<<24) + \
	    ((*(S+1024+((R>>14)&BF_M)))<<0 | (*(S+1025+((R>>14)&BF_M)))<<8 | (*(S+1026+((R>>14)&BF_M)))<<16 | (*(S+1027+((R>>14)&BF_M)))<<24) ^ \
	    ((*(S+2048+((R>>6)&BF_M)))<<0 | (*(S+2049+((R>>6)&BF_M)))<<8 | (*(S+2050+((R>>6)&BF_M)))<<16 | (*(S+2051+((R>>6)&BF_M)))<<24)) + \
	    ((*(S+3072+((R<<2)&BF_M)))<<0 | (*(S+3073+((R<<2)&BF_M)))<<8 | (*(S+3074+((R<<2)&BF_M)))<<16 | (*(S+3075+((R<<2)&BF_M)))<<24)   \
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


__kernel void BFencKernel(__global unsigned long *data, __global unsigned int *bf_constant_schedule) {
	__private unsigned int l, r;
	__private unsigned long block = data[get_global_id(0)];
	
	n2l((unsigned char *)&block,l);
	n2l(((unsigned char *)&block)+4,r);

	__global unsigned int *p;
	__global unsigned char *s;

	p=bf_constant_schedule;
	s=bf_constant_schedule+18;

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

