// vim:foldenable:foldmethod=marker:foldmarker=[[,]]
/**
 * @version 0.1.3 (2011)
 * @author Johannes Gilger <heipei@hackvalue.de>
 * 
 * Copyright 2011 Johannes Gilger
 *
 * This file is part of engine-cuda
 *
 * engine-cuda is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License or
 * any later version.
 * 
 * engine-cuda is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with engine-cuda. If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef __DEVICE_EMULATION__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <openssl/idea.h>
#include <openssl/evp.h>
#include <cuda_runtime_api.h>
#include "cuda_common.h"
#include "common.h"

__constant__ uint64_t idea_constant_schedule[27];
__shared__ uint64_t idea_schedule[27];

#define idea_mul(r,a,b,ul) \
	ul=__umul24(a,b); \
	if (ul) { \
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
	uint32_t x1,x2,x3,x4,t0,t1,ul,l0,l1;

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

	l0=(t0&0xffff);
	l0|=((x1&0xffff)<<16);
	l1=(x4&0xffff);
	l1|=((t1&0xffff)<<16);

	block = ((uint64_t)l0) << 32 | l1;
	flip64(block);
	data[TX] = block;
}

extern "C" void IDEA_cuda_crypt(cuda_crypt_parameters *c) {
	int gridSize;

	transferHostToDevice(c->in, (uint32_t *)c->d_in, c->host_data, c->nbytes);

	if ((c->nbytes%(MAX_THREAD*IDEA_BLOCK_SIZE))==0) {
		gridSize = c->nbytes/(MAX_THREAD*IDEA_BLOCK_SIZE);
	} else {
		gridSize = c->nbytes/(MAX_THREAD*IDEA_BLOCK_SIZE)+1;
	}

	if(output_verbosity == OUTPUT_VERBOSE)
		fprintf(stdout,"Starting IDEA kernel for %zu bytes with (%d, (%d))...\n", c->nbytes, gridSize, MAX_THREAD);

	CUDA_START_TIME

	IDEAencKernel<<<gridSize,MAX_THREAD>>>(c->d_in);
	
	CUDA_STOP_TIME("IDEA       ")

	transferDeviceToHost(c->out, (uint32_t *)c->d_in, c->host_data, c->host_data, c->nbytes);
}

extern "C" void IDEA_cuda_transfer_key_schedule(IDEA_KEY_SCHEDULE *ks) {
	cudaError_t cudaerrno;
	_CUDA(cudaMemcpyToSymbolAsync(idea_constant_schedule,ks,sizeof(IDEA_KEY_SCHEDULE),0,cudaMemcpyHostToDevice));
}

#else
#error "ERROR: DEVICE EMULATION is NOT supported."
#endif
