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
