#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <CL/opencl.h>
#ifdef DEBUG
	#include <sys/time.h>
#endif

void transferHostToDevice_PINNED   (const unsigned char **input, uint32_t **deviceMem, uint8_t **hostMem, size_t *size);
void transferDeviceToHost_PINNED   (unsigned char **output, uint32_t **deviceMem, uint8_t **hostMemS, uint8_t **hostMemOUT, size_t *size);
void transferHostToDevice_ZEROCOPY (const unsigned char **input, uint32_t **deviceMem, uint8_t **hostMem, size_t *size);
void transferDeviceToHost_ZEROCOPY (unsigned char **output, uint32_t **deviceMem, uint8_t **hostMemS, uint8_t **hostMemOUT, size_t *size);
void transferHostToDevice_PAGEABLE (const unsigned char **input, uint32_t **deviceMem, uint8_t **hostMem, size_t *size);
void transferDeviceToHost_PAGEABLE (unsigned char **output, uint32_t **deviceMem, uint8_t **hostMemS, uint8_t **hostMemOUT, size_t *size);

void (*transferHostToDevice) (const unsigned char  **input, uint32_t **deviceMem, uint8_t **hostMem, size_t *size);
void (*transferDeviceToHost) (      unsigned char **output, uint32_t **deviceMem, uint8_t **hostMemS, uint8_t **hostMemOUT, size_t *size);

void check_opencl_error(cl_int error);
int timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y);
const char *cl_error_to_str(cl_int e);

#ifdef DEBUG
	#define OPENCL_TIME_KERNEL(NAME,DIM) \
		cl_event event; \
		clFinish(queue); \
		struct timeval starttime,curtime,difference; \
		gettimeofday(&starttime, NULL); \
		fprintf(stdout, "nbytes: %zu, gridsize: %zu, blocksize: %zu\n", nbytes, gridSize[0], blockSize[0]); \
		\
		clEnqueueNDRangeKernel(queue,device_kernel, DIM, NULL,gridSize, blockSize, 0, NULL, &event); \
		\
		clFinish(queue); \
		cl_ulong start, end; \
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL); \
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,sizeof(cl_ulong), &start, NULL); \
		gettimeofday(&curtime, NULL); \
		timeval_subtract(&difference,&curtime,&starttime); \
		unsigned long opencl_time = (end - start) / 1000; \
		fprintf(stdout, NAME " OpenCL %zu bytes, %06lu usecs, %lu Mb/s\n", nbytes, opencl_time, (1000000/opencl_time * 8 * (unsigned int)nbytes / 1024 / 1024)); \
		fprintf(stdout, NAME " OpenCs %zu bytes, %06d usecs, %u Mb/s\n", nbytes, (int)difference.tv_usec, (1000000/(unsigned int)difference.tv_usec * 8 * (unsigned int)nbytes / 1024 / 1024));
#else
	#define OPENCL_TIME_KERNEL(NAME,DIM) \
		clEnqueueNDRangeKernel(queue,device_kernel, DIM, NULL,gridSize, blockSize, 0, NULL, NULL); 
#endif

#ifndef __OPENCL_HELPER_MACROS__
#define __OPENCL_HELPER_MACROS__
 
#define CL_WRAPPER(FUNC) { \
        cl_int error = FUNC; \
        if (error != CL_SUCCESS) { \
            fprintf(stderr, "Error %d executing %s on %s:%d (%s)\n", \
                err, #FUNC, __FILE__, __LINE__, cl_error_to_str(error)); \
            abort(); \
        }; \
    }
 
int err; 
#define CL_ASSIGN(ASSIGNMENT) { \
        ASSIGNMENT; \
        if (error != CL_SUCCESS) { \
            fprintf(stderr, "Error %d executing %s on %s:%d (%s)\n", \
                error, #ASSIGNMENT, __FILE__, __LINE__, cl_error_to_str(error)); \
            abort(); \
        }; \
    }
 
 
#endif
