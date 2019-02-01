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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#include "cuda_common.h"
#include "common.h"

#ifndef PAGEABLE
extern "C" void transferHostToDevice_PINNED   (const unsigned char *input, uint32_t *deviceMem, uint8_t *hostMem, size_t size) {
	cudaError_t cudaerrno;
	if(size <= 1048576) {
		memcpy(hostMem,input,size);
        	_CUDA(cudaMemcpyAsync(deviceMem, hostMem, size, cudaMemcpyHostToDevice, 0));
	} else {
		//fprintf(stdout, "Now trying cudaMemcpy\n");
		_CUDA(cudaMemcpyAsync(deviceMem, input, size, cudaMemcpyHostToDevice,0));
	}
}
#if CUDART_VERSION >= 2020
extern "C" void transferHostToDevice_ZEROCOPY (const unsigned char *input, uint32_t *deviceMem, uint8_t *hostMem, size_t size) {
	//cudaError_t cudaerrno;
	memcpy(hostMem,input,size);
	//_CUDA(cudaHostGetDevicePointer(&d_s,h_s, 0));
}
#endif
#else
extern "C" void transferHostToDevice_PAGEABLE (const unsigned char *input, uint32_t *deviceMem, uint8_t *hostMem, size_t size) {
	cudaError_t cudaerrno;
	_CUDA(cudaMemcpy(deviceMem, input, size, cudaMemcpyHostToDevice));
}
#endif

#ifndef PAGEABLE
extern "C" void transferDeviceToHost_PINNED   (unsigned char *output, uint32_t *deviceMem, uint8_t *hostMemS, uint8_t *hostMemOUT, size_t size) {
	cudaError_t cudaerrno;
	if(size <= 1048576) {
        	_CUDA(cudaMemcpyAsync(hostMemS, deviceMem, size, cudaMemcpyDeviceToHost, 0));
		_CUDA(cudaThreadSynchronize());
		memcpy(output,hostMemS,size);
	} else {
		_CUDA(cudaMemcpyAsync(output, deviceMem, size, cudaMemcpyDeviceToHost, 0));
	}
}
#if CUDART_VERSION >= 2020
extern "C" void transferDeviceToHost_ZEROCOPY (unsigned char *output, uint32_t *deviceMem, uint8_t *hostMemS, uint8_t *hostMemOUT, size_t size) {
	cudaError_t cudaerrno;
	_CUDA(cudaThreadSynchronize());
	memcpy(output,hostMemOUT,size);
}
#endif
#else
extern "C" void transferDeviceToHost_PAGEABLE (unsigned char *output, uint32_t *deviceMem, uint8_t *hostMemS, uint8_t *hostMemOUT, size_t size) {
	cudaError_t cudaerrno;
	_CUDA(cudaMemcpy(output,deviceMem,size, cudaMemcpyDeviceToHost));
}
#endif


float time_elapsed;
cudaEvent_t time_start,time_stop;

#ifdef DEBUG
#include <sys/time.h>
int timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y) {
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (x->tv_usec - y->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_usec = x->tv_usec - y->tv_usec;

  return x->tv_sec < y->tv_sec;
}
#endif

void checkCUDADevice(struct cudaDeviceProp *deviceProp, int output_verbosity) {
	int deviceCount;
	cudaError_t cudaerrno;

	_CUDA(cudaGetDeviceCount(&deviceCount));

	if (!deviceCount) {
		if (output_verbosity!=OUTPUT_QUIET) 
			fprintf(stderr,"There is no device supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}

	if (output_verbosity>=OUTPUT_NORMAL) 
		fprintf(stdout,"Successfully found %d CUDA devices (CUDART_VERSION %d).\n",deviceCount, CUDART_VERSION);

	/* added by abu naser(an16e@my.fsu.edu) to solve deviceid problem */	
	int deviceId;
	_CUDA(cudaGetDevice(&deviceId)) ;
	if(!deviceId)
		fprintf(stdout,"Successfully found  CUDA deviceId = %d.\n",deviceId);
	else
		fprintf(stdout,"CUDA deviceId not found.\n");
	
	_CUDA(cudaSetDevice(deviceId));
	_CUDA(cudaGetDeviceProperties(deviceProp, deviceId));
	
	// _CUDA(cudaSetDevice(6));
	// _CUDA(cudaGetDeviceProperties(deviceProp, 6));
	/* end of add */
	

	
	if (output_verbosity==OUTPUT_VERBOSE) {
			fprintf(stdout,"\nDevice %d: \"%s\"\n", deviceId, deviceProp->name);
			// fprintf(stdout,"\nDevice %d: \"%s\"\n", 6, deviceProp->name);
      	 	fprintf(stdout,"  CUDA Compute Capability:                       %d.%d\n", deviceProp->major,deviceProp->minor);
#if CUDART_VERSION >= 2000
        	fprintf(stdout,"  Number of multiprocessors (SM):                %d\n", deviceProp->multiProcessorCount);
#endif
#if CUDART_VERSION >= 2020
		fprintf(stdout,"  Integrated:                                    %s\n", deviceProp->integrated ? "Yes" : "No");
        	fprintf(stdout,"  Support host page-locked memory mapping:       %s\n", deviceProp->canMapHostMemory ? "Yes" : "No");
#endif
		fprintf(stdout,"\n");
	}
}

extern "C" void cuda_device_init(int *nm, int buffer_size, int output_verbosity, uint8_t **host_data, uint64_t **device_data, uint64_t **device_data_out) {
	assert(nm);
	cudaError_t cudaerrno;
	cudaDeviceProp deviceProp;
    	
	checkCUDADevice(&deviceProp, output_verbosity);
	
	if(buffer_size==0)
		buffer_size=MAX_CHUNK_SIZE;
	
	//_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleYield));
	//_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
	//_CUDA(cudaSetDeviceFlags(cudaDeviceBlockingSync));
	//_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleYield|cudaDeviceBlockingSync));
#if CUDART_VERSION >= 2000
	*nm=deviceProp.multiProcessorCount;
#endif

#ifndef PAGEABLE 
#if CUDART_VERSION >= 2020
	isIntegrated=deviceProp.integrated;
	if(isIntegrated) {
        	//zero-copy memory mode - use special function to get OS-pinned memory
		_CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));
        	if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"Using zero-copy memory.\n");
        	_CUDA(cudaHostAlloc((void**)host_data,buffer_size,cudaHostAllocMapped));
		transferHostToDevice = transferHostToDevice_ZEROCOPY;		// set memory transfer function
		transferDeviceToHost = transferDeviceToHost_ZEROCOPY;		// set memory transfer function
		_CUDA(cudaHostGetDevicePointer(device_data,host_data, 0));
	} else {
		//pinned memory mode - use special function to get OS-pinned memory
		_CUDA(cudaHostAlloc( (void**)host_data, buffer_size, cudaHostAllocDefault));
		if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"Using pinned memory: cudaHostAllocDefault.\n");
		transferHostToDevice = transferHostToDevice_PINNED;	// set memory transfer function
		transferDeviceToHost = transferDeviceToHost_PINNED;	// set memory transfer function
		_CUDA(cudaMalloc((void **)device_data,buffer_size));
		_CUDA(cudaMalloc((void **)device_data_out,buffer_size));
	}
#else
        //pinned memory mode - use special function to get OS-pinned memory
        _CUDA(cudaMallocHost((void**)&h_s, buffer_size));
        if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"Using pinned memory: cudaHostAllocDefault.\n");
	transferHostToDevice = transferHostToDevice_PINNED;			// set memory transfer function
	transferDeviceToHost = transferDeviceToHost_PINNED;			// set memory transfer function
	_CUDA(cudaMalloc((void **)device_data,buffer_size));
	_CUDA(cudaMalloc((void **)device_data_out,buffer_size));
#endif
#else
        if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"Using pageable memory.\n");
	transferHostToDevice = transferHostToDevice_PAGEABLE;			// set memory transfer function
	transferDeviceToHost = transferDeviceToHost_PAGEABLE;			// set memory transfer function
	_CUDA(cudaMalloc((void **)device_data,buffer_size));
	_CUDA(cudaMalloc((void **)device_data_out,buffer_size));
#endif

	if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"The current buffer size is %d.\n\n", buffer_size);

	if(output_verbosity>=OUTPUT_NORMAL) {
		_CUDA(cudaEventCreate(&time_start));
		_CUDA(cudaEventCreate(&time_stop));
		_CUDA(cudaEventRecord(time_start,0));
	}

}

extern "C" void cuda_device_finish(uint8_t *host_data, uint64_t *device_data) {
	cudaError_t cudaerrno;

	if (output_verbosity>=OUTPUT_NORMAL) fprintf(stdout, "\nDone. Finishing up...\n");

#ifndef PAGEABLE 
#if CUDART_VERSION >= 2020
	if(isIntegrated) {
		_CUDA(cudaFreeHost(host_data));
	} else {
		_CUDA(cudaFree(device_data));
	}
#else	
	_CUDA(cudaFree(device_data));
#endif
#else
	_CUDA(cudaFree(device_data));
#endif	

	if(output_verbosity>=OUTPUT_NORMAL) {
		_CUDA(cudaEventRecord(time_stop,0));
		_CUDA(cudaEventSynchronize(time_stop));
		_CUDA(cudaEventElapsedTime(&time_elapsed,time_start,time_stop));
		fprintf(stdout,"\nTotal time: %f milliseconds\n",time_elapsed);	
	}
}
