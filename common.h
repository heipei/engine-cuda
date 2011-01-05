#ifndef common_h
#define common_h

#include <cuda_runtime_api.h>

#define NUM_BLOCK_PER_MULTIPROCESSOR	3
#define SIZE_BLOCK_PER_MULTIPROCESSOR	256*1024
#define MAX_THREAD			128
#define STATE_THREAD_AES		4
#define STATE_THREAD_DES		2

#define AES_KEY_SIZE_128	16
#define AES_KEY_SIZE_192	24
#define AES_KEY_SIZE_256	32

#define OUTPUT_QUIET		0
#define OUTPUT_NORMAL		1
#define OUTPUT_VERBOSE		2

#define DES_MAXNR		8
#define DES_BLOCK_SIZE		8
#define DES_KEY_SIZE		8
#define DES_KEY_SIZE_64		8

#define IDEA_BLOCK_SIZE		8
#define IDEA_KEY_SIZE		8
#define IDEA_KEY_SIZE_64	8

#define TX blockIdx.x * (blockDim.x * blockDim.y) + (blockDim.y * threadIdx.x) + threadIdx.y

const char *byte_to_binary(int);
void tobinary(unsigned char);
char *int2bin(unsigned int, char*, int);
#endif
