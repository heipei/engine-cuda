#include <stdint.h>
#include <cuda_runtime_api.h>

static int __attribute__((unused)) output_verbosity;
static int __attribute__((unused)) isIntegrated;

#define n2l(c,l)        (l =((unsigned long)(*(c)))<<24L, \
                         l|=((unsigned long)(*(c+1)))<<16L, \
                         l|=((unsigned long)(*(c+2)))<< 8L, \
                         l|=((unsigned long)(*(c+3))))

#define flip64(a)	(a= \
			((a & 0x00000000000000FF) << 56) | \
			((a & 0x000000000000FF00) << 40) | \
			((a & 0x0000000000FF0000) << 24) | \
			((a & 0x00000000FF000000) << 8)  | \
			((a & 0x000000FF00000000) >> 8)  | \
			((a & 0x0000FF0000000000) >> 24) | \
			((a & 0x00FF000000000000) >> 40) | \
			((a & 0xFF00000000000000) >> 56))


#define _CUDA(call) {																	\
	call;				                                												\
	cudaerrno=cudaGetLastError();																	\
	if(cudaSuccess!=cudaerrno) {                                       					         						\
		if (output_verbosity!=OUTPUT_QUIET) fprintf(stderr, "Cuda error in file '%s' in line %i: %s.\n",__FILE__,__LINE__,cudaGetErrorString(cudaerrno));	\
		exit(EXIT_FAILURE);                                                  											\
    } }

#define _CUDA_N(msg) {                                    												\
	cudaerrno=cudaGetLastError();																	\
	if(cudaSuccess!=cudaerrno) {                                                											\
		if (output_verbosity!=OUTPUT_QUIET) fprintf(stderr, "Cuda error in file '%s' in line %i: %s.\n",__FILE__,__LINE__-3,cudaGetErrorString(cudaerrno));	\
		exit(EXIT_FAILURE);                                                  											\
    } }


#ifndef PAGEABLE
extern "C" void transferHostToDevice_PINNED   (const unsigned char **input, uint32_t **deviceMem, uint8_t **hostMem, size_t *size);
extern "C" void transferDeviceToHost_PINNED   (unsigned char **output, uint32_t **deviceMem, uint8_t **hostMemS, uint8_t **hostMemOUT, size_t *size);
#if CUDART_VERSION >= 2020
extern "C" void transferHostToDevice_ZEROCOPY (const unsigned char **input, uint32_t **deviceMem, uint8_t **hostMem, size_t *size);
extern "C" void transferDeviceToHost_ZEROCOPY (unsigned char **output, uint32_t **deviceMem, uint8_t **hostMemS, uint8_t **hostMemOUT, size_t *size);
#endif
#else
extern "C" void transferHostToDevice_PAGEABLE (const unsigned char **input, uint32_t **deviceMem, uint8_t **hostMem, size_t *size);
extern "C" void transferDeviceToHost_PAGEABLE (unsigned char **output, uint32_t **deviceMem, uint8_t **hostMemS, uint8_t **hostMemOUT, size_t *size);
#endif

extern "C" void (*transferHostToDevice) (const unsigned char  **input, uint32_t **deviceMem, uint8_t **hostMem, size_t *size);
extern "C" void (*transferDeviceToHost) (      unsigned char **output, uint32_t **deviceMem, uint8_t **hostMemS, uint8_t **hostMemOUT, size_t *size);

extern "C" void checkCUDADevice(struct cudaDeviceProp *deviceProp, int output_verbosity);

extern "C" void tobinary(unsigned char);
extern "C" const char *byte_to_binary(int);
extern "C" char *int2bin(unsigned int, char*, int);
