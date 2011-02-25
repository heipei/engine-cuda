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
		blockSize[0] = gridSize[0] = 128;
	}


	clSetKernelArg(device_kernel, 0, sizeof(cl_mem), device_buffer);
	clSetKernelArg(device_kernel, 1, sizeof(cl_mem), device_schedule);

	#ifdef DEBUG
		struct timeval starttime,curtime,difference;
		gettimeofday(&starttime, NULL);
		fprintf(stdout, "nbytes: %zu, gridsize: %zu, blocksize: %zu\n", nbytes, gridSize[0], blockSize[0]);
	#endif
	
	clEnqueueWriteBuffer(queue,*device_buffer,CL_TRUE,0,nbytes,in,0,NULL,NULL);
	clEnqueueNDRangeKernel(queue,device_kernel, 1, NULL,gridSize, blockSize, 0, NULL, NULL);
	clEnqueueReadBuffer(queue,*device_buffer,CL_TRUE,0,nbytes,out,0,NULL,NULL);

	#ifdef DEBUG
		gettimeofday(&curtime, NULL);
		timeval_subtract(&difference,&curtime,&starttime);
		fprintf(stdout, "OpenCL kernel: %d.%06d\n", (int)difference.tv_sec, (int)difference.tv_usec);
	#endif
}

void IDEA_opencl_transfer_key_schedule(IDEA_KEY_SCHEDULE *ks, cl_mem *device_schedule, cl_command_queue queue) {
	clEnqueueWriteBuffer(queue,*device_schedule,CL_TRUE,0,sizeof(IDEA_KEY_SCHEDULE),ks,0,NULL,NULL);
}
