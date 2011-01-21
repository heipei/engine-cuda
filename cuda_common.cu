#include <cuda_common.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <common.h>
#include <assert.h>

#ifndef PAGEABLE
extern "C" void transferHostToDevice_PINNED   (const unsigned char **input, uint32_t **deviceMem, uint8_t **hostMem, size_t *size) {
	cudaError_t cudaerrno;
	memcpy(*hostMem,*input,*size);
        _CUDA(cudaMemcpyAsync(*deviceMem, *hostMem, *size, cudaMemcpyHostToDevice, 0));
}
#if CUDART_VERSION >= 2020
extern "C" void transferHostToDevice_ZEROCOPY (const unsigned char **input, uint32_t **deviceMem, uint8_t **hostMem, size_t *size) {
	//cudaError_t cudaerrno;
	memcpy(*hostMem,*input,*size);
	//_CUDA(cudaHostGetDevicePointer(&d_s,h_s, 0));
}
#endif
#else
extern "C" void transferHostToDevice_PAGEABLE (const unsigned char **input, uint32_t **deviceMem, uint8_t **hostMem, size_t *size) {
	cudaError_t cudaerrno;
	_CUDA(cudaMemcpy(*deviceMem, *input, *size, cudaMemcpyHostToDevice));
}
#endif

#ifndef PAGEABLE
extern "C" void transferDeviceToHost_PINNED   (unsigned char **output, uint32_t **deviceMem, uint8_t **hostMemS, uint8_t **hostMemOUT, size_t *size) {
	cudaError_t cudaerrno;
        _CUDA(cudaMemcpyAsync(*hostMemS, *deviceMem, *size, cudaMemcpyDeviceToHost, 0));
	_CUDA(cudaThreadSynchronize());
	memcpy(*output,*hostMemS,*size);
}
#if CUDART_VERSION >= 2020
extern "C" void transferDeviceToHost_ZEROCOPY (unsigned char **output, uint32_t **deviceMem, uint8_t **hostMemS, uint8_t **hostMemOUT, size_t *size) {
	cudaError_t cudaerrno;
	_CUDA(cudaThreadSynchronize());
	memcpy(*output,*hostMemOUT,*size);
}
#endif
#else
extern "C" void transferDeviceToHost_PAGEABLE (unsigned char **output, uint32_t **deviceMem, uint8_t **hostMemS, uint8_t **hostMemOUT, size_t *size) {
	cudaError_t cudaerrno;
	_CUDA(cudaMemcpy(*output,*deviceMem,*size, cudaMemcpyDeviceToHost));
}
#endif


float time_elapsed;
cudaEvent_t time_start,time_stop;

void checkCUDADevice(struct cudaDeviceProp *deviceProp, int output_verbosity) {
	int deviceCount;
	cudaError_t cudaerrno;

	_CUDA(cudaGetDeviceCount(&deviceCount));

	if (!deviceCount) {
		if (output_verbosity!=OUTPUT_QUIET) 
			fprintf(stderr,"There is no device supporting CUDA.\n");
		exit(EXIT_FAILURE);
	} else {
		if (output_verbosity>=OUTPUT_NORMAL) 
			fprintf(stdout,"Successfully found %d CUDA devices (CUDART_VERSION %d).\n",deviceCount, CUDART_VERSION);
	}
	_CUDA(cudaSetDevice(0));
	_CUDA(cudaGetDeviceProperties(deviceProp, 0));
	
	if (output_verbosity==OUTPUT_VERBOSE) {
        	fprintf(stdout,"\nDevice %d: \"%s\"\n", 0, deviceProp->name);
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

extern "C" void cuda_device_init(int *nm, int buffer_size, int output_verbosity, uint8_t **host_data, uint64_t **device_data) {
	assert(nm);
	cudaError_t cudaerrno;
	cudaDeviceProp deviceProp;
    	
	checkCUDADevice(&deviceProp, output_verbosity);
	
	if(buffer_size==0)
		buffer_size=MAX_CHUNK_SIZE;
	
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
	}
#else
        //pinned memory mode - use special function to get OS-pinned memory
        _CUDA(cudaMallocHost((void**)&h_s, buffer_size));
        if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"Using pinned memory: cudaHostAllocDefault.\n");
	transferHostToDevice = transferHostToDevice_PINNED;			// set memory transfer function
	transferDeviceToHost = transferDeviceToHost_PINNED;			// set memory transfer function
	_CUDA(cudaMalloc((void **)device_data,buffer_size));
#endif
#else
        if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"Using pageable memory.\n");
	transferHostToDevice = transferHostToDevice_PAGEABLE;			// set memory transfer function
	transferDeviceToHost = transferDeviceToHost_PAGEABLE;			// set memory transfer function
	_CUDA(cudaMalloc((void **)device_data,buffer_size));
#endif

	if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"The current buffer size is %d.\n\n", buffer_size);

	_CUDA(cudaEventCreate(&time_start));
	_CUDA(cudaEventCreate(&time_stop));
	_CUDA(cudaEventRecord(time_start,0));

}

extern "C" void cuda_device_finish(uint8_t *host_data, uint64_t *device_data) {
	cudaError_t cudaerrno;

	if (output_verbosity>=OUTPUT_NORMAL) fprintf(stdout, "\nDone. Finishing up...\n");

#ifndef PAGEABLE 
#if CUDART_VERSION >= 2020
	if(isIntegrated) {
		_CUDA(cudaFreeHost(host_data));
		//_CUDA(cudaFreeHost(h_iv));
	} else {
		_CUDA(cudaFree(device_data));
		//_CUDA(cudaFree(d_iv));
	}
#else	
	_CUDA(cudaFree(device_data));
	//_CUDA(cudaFree(d_iv));
#endif
#else
	_CUDA(cudaFree(device_data));
	//_CUDA(cudaFree(d_iv));
#endif	

	_CUDA(cudaEventRecord(time_stop,0));
	_CUDA(cudaEventSynchronize(time_stop));
	_CUDA(cudaEventElapsedTime(&time_elapsed,time_start,time_stop));

	if (output_verbosity>=OUTPUT_NORMAL) fprintf(stdout,"\nTotal time: %f milliseconds\n",time_elapsed);	
}
