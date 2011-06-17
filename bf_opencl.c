// vim:ft=opencl:
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
#include <CL/opencl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <openssl/blowfish.h>
#include <sys/time.h>
#include "common.h"
#include "opencl_common.h"

static cl_mem bf_iv = NULL;

void BF_opencl_crypt(opencl_crypt_parameters *c) {

	size_t gridSize[3] = {1, 0, 0};
	size_t blockSize[3] = {MAX_THREAD, 0, 0};
	
	if ((c->nbytes%BF_BLOCK_SIZE)==0) {
		gridSize[0] = c->nbytes/BF_BLOCK_SIZE;
	} else {
		gridSize[0] = c->nbytes/BF_BLOCK_SIZE+1;
	}

	if(gridSize[0] < MAX_THREAD) {
		blockSize[0] = gridSize[0];
	}

	//fprintf(stdout, "\nMax buffer size: %d, BF_KEY size: %zu\n", CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(BF_KEY));
	clSetKernelArg(*c->d_kernel, 0, sizeof(cl_mem), *c->d_in);
	clSetKernelArg(*c->d_kernel, 1, sizeof(cl_mem), *c->d_schedule);

	cl_uint args;
	clGetKernelInfo(*c->d_kernel,CL_KERNEL_NUM_ARGS,4,&args,NULL);
	if(args > 2 && bf_iv) {
		clSetKernelArg(*c->d_kernel, 2, sizeof(cl_mem), &bf_iv);
	}

	clEnqueueWriteBuffer(*c->queue,*c->d_in,CL_TRUE,0,c->nbytes,c->in,0,NULL,NULL);

	OPENCL_TIME_KERNEL("BF      ",1)

	clEnqueueReadBuffer(*c->queue,*c->d_in,CL_TRUE,0,c->nbytes,c->out,0,NULL,NULL);
}

void BF_opencl_transfer_key_schedule(BF_KEY *ks,cl_mem *device_schedule,cl_command_queue queue) {
	assert(ks);
	clEnqueueWriteBuffer(queue,*device_schedule,CL_TRUE,0,sizeof(BF_KEY),ks,0,NULL,NULL);
}

void BF_opencl_transfer_iv(cl_context context, const unsigned char *iv,cl_command_queue queue) {
	cl_int error;
	CL_ASSIGN(bf_iv = clCreateBuffer(context,CL_MEM_READ_ONLY,BF_BLOCK_SIZE,NULL,&error));
	CL_WRAPPER(clEnqueueWriteBuffer(queue,bf_iv,CL_TRUE,0,BF_BLOCK_SIZE,iv,0,NULL,NULL));
}
