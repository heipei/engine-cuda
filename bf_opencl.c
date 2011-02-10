// vim:ft=opencl:
#include <CL/opencl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <assert.h>
#include <openssl/blowfish.h>
#include <sys/time.h>
#include "common.h"
#include "opencl_common.h"

void BF_opencl_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, int enc, cl_mem *device_buffer, cl_mem *device_schedule, cl_command_queue queue, cl_kernel device_kernel) {
	assert(in && out && nbytes);

	size_t gridSize[1] = {1};
	size_t blockSize[1] = {MAX_THREAD};
	//struct timeval starttime,curtime,difference;
	//cl_int error;

	if ((nbytes%BF_BLOCK_SIZE)==0) {
		gridSize[0] = nbytes/BF_BLOCK_SIZE;
	} else {
		gridSize[0] = nbytes/BF_BLOCK_SIZE+1;
	}

	if(gridSize[0] < MAX_THREAD) {
		blockSize[0] = gridSize[0];
	}

	//fprintf(stdout, "nbytes: %zu, gridsize: %zu, blocksize: %zu\n", nbytes, gridSize[0], blockSize[0]);

	clSetKernelArg(device_kernel, 0, sizeof(cl_mem), device_buffer);
	//check_opencl_error(error);
	clSetKernelArg(device_kernel, 1, sizeof(cl_mem), device_schedule);
	//check_opencl_error(error);

	//gettimeofday(&starttime, NULL);

	clEnqueueWriteBuffer(queue,*device_buffer,CL_TRUE,0,nbytes,in,0,NULL,NULL);
	//check_opencl_error(error);
	clEnqueueNDRangeKernel(queue,device_kernel, 1, NULL,gridSize, blockSize, 0, NULL, NULL);
	//check_opencl_error(error);
	clEnqueueReadBuffer(queue,*device_buffer,CL_TRUE,0,nbytes,out,0,NULL,NULL);
	//check_opencl_error(error);

	/*
	gettimeofday(&curtime, NULL);
	timeval_subtract(&difference,&curtime,&starttime);
	fprintf(stdout, "OpenCL kernel: %d.%06d\n", (int)difference.tv_sec, (int)difference.tv_usec);
	*/

}

void BF_opencl_transfer_key_schedule(BF_KEY *ks,cl_mem *device_schedule,cl_command_queue queue) {
	assert(ks);
	clEnqueueWriteBuffer(queue,*device_schedule,CL_TRUE,0,sizeof(BF_KEY),ks,0,NULL,NULL);
}
