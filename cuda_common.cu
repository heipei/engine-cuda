#include <cuda_common.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <common.h>
#include <assert.h>

#ifndef PAGEABLE
extern "C" void transferHostToDevice_PINNED   (const unsigned char **input, uint32_t **deviceMem, uint8_t **hostMem, size_t *size, cudaStream_t stream) {
	cudaError_t cudaerrno;
	//memcpy(*hostMem,*input,*size);
        cudaMemcpyAsync(*deviceMem, *hostMem, *size, cudaMemcpyHostToDevice, stream);
}
#if CUDART_VERSION >= 2020
extern "C" void transferHostToDevice_ZEROCOPY (const unsigned char **input, uint32_t **deviceMem, uint8_t **hostMem, size_t *size, cudaStream_t stream) {
	//cudaError_t cudaerrno;
	memcpy(*hostMem,*input,*size);
	//_CUDA(cudaHostGetDevicePointer(&d_s,h_s, 0));
}
#endif
#else
extern "C" void transferHostToDevice_PAGEABLE (const unsigned char **input, uint32_t **deviceMem, uint8_t **hostMem, size_t *size, cudaStream_t stream) {
	cudaError_t cudaerrno;
	_CUDA(cudaMemcpy(*deviceMem, *input, *size, cudaMemcpyHostToDevice));
}
#endif

#ifndef PAGEABLE
extern "C" void transferDeviceToHost_PINNED   (unsigned char **output, uint32_t **deviceMem, uint8_t **hostMemS, uint8_t **hostMemOUT, size_t *size, cudaStream_t stream) {
	cudaError_t cudaerrno;
        cudaMemcpyAsync(*hostMemS, *deviceMem, *size, cudaMemcpyDeviceToHost, stream);
	//_CUDA(cudaThreadSynchronize());
	//memcpy(*output,*hostMemS,*size);
}
#if CUDART_VERSION >= 2020
extern "C" void transferDeviceToHost_ZEROCOPY (unsigned char **output, uint32_t **deviceMem, uint8_t **hostMemS, uint8_t **hostMemOUT, size_t *size, cudaStream_t stream) {
	cudaError_t cudaerrno;
	_CUDA(cudaThreadSynchronize());
	memcpy(*output,*hostMemOUT,*size);
}
#endif
#else
extern "C" void transferDeviceToHost_PAGEABLE (unsigned char **output, uint32_t **deviceMem, uint8_t **hostMemS, uint8_t **hostMemOUT, size_t *size, cudaStream_t stream) {
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
	}

	if (output_verbosity>=OUTPUT_NORMAL) 
		fprintf(stdout,"Successfully found %d CUDA devices (CUDART_VERSION %d).\n",deviceCount, CUDART_VERSION);

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

extern "C" void cuda_device_init(int *num_streams, cuda_streams_st *streams, int *nm, int buffer_size, int output_verbosity) {
	assert(nm);
	cudaError_t cudaerrno;
	cudaDeviceProp deviceProp;
	int i=0;
    	
	checkCUDADevice(&deviceProp, output_verbosity);
	
	if(buffer_size==0)
		buffer_size=MAX_CHUNK_SIZE;
	
#if CUDART_VERSION >= 2000
	*nm=deviceProp.multiProcessorCount;
#endif
	if(deviceProp.deviceOverlap) {
		*num_streams = 8;
		buffer_size=buffer_size/(*num_streams);
	}

	for(i=0; i<*num_streams; i++) {
		cudaStreamCreate(&(streams[i].stream));
	}
        if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"CUDA streams enabled (%d streams)\n", *num_streams);

#ifndef PAGEABLE 
#if CUDART_VERSION >= 2020
	isIntegrated=deviceProp.integrated;
	if(isIntegrated) {
        	//zero-copy memory mode - use special function to get OS-pinned memory
		_CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));
        	if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"Using zero-copy memory.\n");
		for(i=0; i<*num_streams; i++) {
			_CUDA(cudaHostAlloc((void**)&(streams[i].host_data),buffer_size,cudaHostAllocMapped));
			_CUDA(cudaHostGetDevicePointer(&(streams[i].device_data),&(streams[i].host_data), 0));
		}
		transferHostToDevice = transferHostToDevice_ZEROCOPY;		// set memory transfer function
		transferDeviceToHost = transferDeviceToHost_ZEROCOPY;		// set memory transfer function
	} else {
		//pinned memory mode - use special function to get OS-pinned memory
		for(i=0; i<*num_streams; i++) {
			_CUDA(cudaHostAlloc( (void**)&(streams[i].host_data), buffer_size, cudaHostAllocDefault));
			_CUDA(cudaMalloc((void **)&(streams[i].device_data),buffer_size));
		}
		if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"Using pinned memory: cudaHostAllocDefault.\n");
		transferHostToDevice = transferHostToDevice_PINNED;	// set memory transfer function
		transferDeviceToHost = transferDeviceToHost_PINNED;	// set memory transfer function
	}
#else
        //pinned memory mode - use special function to get OS-pinned memory
	for(i=0; i<*num_streams; i++) {
        	_CUDA(cudaMallocHost((void**)streams[i].host_data,buffer_size));
		_CUDA(cudaMalloc((void **)streams[i].device_data,buffer_size));
	}
        if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"Using pinned memory: cudaHostAllocDefault.\n");
	transferHostToDevice = transferHostToDevice_PINNED;			// set memory transfer function
	transferDeviceToHost = transferDeviceToHost_PINNED;			// set memory transfer function
#endif
#else
        if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"Using pageable memory.\n");
	for(i=0; i<*num_streams; i++) {
		_CUDA(cudaMalloc((void **)&(streams[i].device_data),buffer_size));
	}
	transferHostToDevice = transferHostToDevice_PAGEABLE;			// set memory transfer function
	transferDeviceToHost = transferDeviceToHost_PAGEABLE;			// set memory transfer function
#endif

	if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"The current buffer size is %d.\n\n", buffer_size);

	if(output_verbosity>=OUTPUT_NORMAL) {
		_CUDA(cudaEventCreate(&time_start));
		_CUDA(cudaEventCreate(&time_stop));
		_CUDA(cudaEventRecord(time_start,0));
	}

}

extern "C" void cuda_device_finish(int num_streams, cuda_streams_st *streams) {
	cudaError_t cudaerrno;

	if (output_verbosity>=OUTPUT_NORMAL) fprintf(stdout, "\nDone. Finishing up...\n");
	
	int i=0;
	for(i=0; i<num_streams; i++) {
	
#ifndef PAGEABLE 
#if CUDART_VERSION >= 2020
	if(isIntegrated) {
		_CUDA(cudaFreeHost(streams[i].host_data));
		//_CUDA(cudaFreeHost(h_iv));
	} else {
		_CUDA(cudaFree(streams[i].device_data));
		_CUDA(cudaFreeHost(streams[i].host_data));
		//_CUDA(cudaFree(d_iv));
	}
#else	
	_CUDA(cudaFree(streams[i].device_data));
	//_CUDA(cudaFree(d_iv));
#endif
#else
	_CUDA(cudaFree(streams[i].device_data));
	//_CUDA(cudaFree(d_iv));
#endif	
	
	_CUDA(cudaStreamDestroy(streams[i].stream));
	}
	if(output_verbosity>=OUTPUT_NORMAL) {
		_CUDA(cudaEventRecord(time_stop,0));
		_CUDA(cudaEventSynchronize(time_stop));
		_CUDA(cudaEventElapsedTime(&time_elapsed,time_start,time_stop));
		fprintf(stdout,"\nTotal time: %f milliseconds\n",time_elapsed);	
	}
}
