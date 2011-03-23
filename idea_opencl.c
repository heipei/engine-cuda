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

	#ifdef DEBUG
		cl_event event;
		clFinish(queue);
		struct timeval starttime,curtime,difference;
		gettimeofday(&starttime, NULL);
		fprintf(stdout, "nbytes: %zu, gridsize: %zu, blocksize: %zu\n", nbytes, gridSize[0], blockSize[0]);
		
		clEnqueueNDRangeKernel(queue,device_kernel, 1, NULL,gridSize, blockSize, 0, NULL, &event);

		clFinish(queue);
		cl_ulong start, end;
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,sizeof(cl_ulong), &start, NULL);
		gettimeofday(&curtime, NULL);
		timeval_subtract(&difference,&curtime,&starttime);
		unsigned long opencl_time = (end - start) / 1000;
		fprintf(stdout, "IDEA     OpenCi %zu bytes, %06lu usecs, %lu Mb/s\n", nbytes, opencl_time, (1000000/opencl_time * 8 * ((unsigned int)nbytes/1024)/1024));
		fprintf(stdout, "IDEA     OpenCL %zu bytes, %06d usecs, %u Mb/s\n", nbytes, (int)difference.tv_usec, (1000000/(unsigned int)difference.tv_usec * 8 * ((unsigned int)nbytes/1024)/1024));
	#else
		clEnqueueNDRangeKernel(queue,device_kernel, 1, NULL,gridSize, blockSize, 0, NULL, NULL);
	#endif

	clEnqueueReadBuffer(queue,*device_buffer,CL_TRUE,0,nbytes,out,0,NULL,NULL);
}

void IDEA_opencl_transfer_key_schedule(IDEA_KEY_SCHEDULE *ks, cl_mem *device_schedule, cl_command_queue queue) {
	clEnqueueWriteBuffer(queue,*device_schedule,CL_TRUE,0,sizeof(IDEA_KEY_SCHEDULE),ks,0,NULL,NULL);
	#ifdef DEBUG
		clFinish(queue);
	#endif
}
