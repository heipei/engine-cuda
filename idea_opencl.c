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
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <openssl/idea.h>
#include <CL/opencl.h>
#include "common.h"
#include "opencl_common.h"

void IDEA_opencl_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, int enc, cl_mem *device_buffer, cl_mem *device_schedule, cl_command_queue queue, cl_kernel device_kernel, cl_context context) {
	size_t gridSize[3] = {1, 0, 0};
	size_t blockSize[3] = {MAX_THREAD, 0, 0};
	
	if ((nbytes%IDEA_BLOCK_SIZE)==0) {
		gridSize[0] = nbytes/IDEA_BLOCK_SIZE;
	} else {
		gridSize[0] = nbytes/IDEA_BLOCK_SIZE+1;
	}

	if(gridSize[0] < MAX_THREAD) {
		blockSize[0] = gridSize[0] = MAX_THREAD;
	}

	clSetKernelArg(device_kernel, 0, sizeof(cl_mem), device_buffer);
	clSetKernelArg(device_kernel, 1, sizeof(cl_mem), device_schedule);

	clEnqueueWriteBuffer(queue,*device_buffer,CL_TRUE,0,nbytes,in,0,NULL,NULL);

	OPENCL_TIME_KERNEL("IDEA    ",1)

	clEnqueueReadBuffer(queue,*device_buffer,CL_TRUE,0,nbytes,out,0,NULL,NULL);
}

void IDEA_opencl_transfer_key_schedule(IDEA_KEY_SCHEDULE *ks, cl_mem *device_schedule, cl_command_queue queue) {
	clEnqueueWriteBuffer(queue,*device_schedule,CL_TRUE,0,sizeof(IDEA_KEY_SCHEDULE),ks,0,NULL,NULL);
	#ifdef DEBUG
		clFinish(queue);
	#endif
}
