#ifndef common_h
#define common_h

#include <cuda_runtime_api.h>

#define NUM_BLOCK_PER_MULTIPROCESSOR	3
#define SIZE_BLOCK_PER_MULTIPROCESSOR	256*1024
#define MAX_THREAD			256
#define STATE_THREAD			4
#define STATE_THREAD_DES		2

#define OUTPUT_QUIET		0
#define OUTPUT_NORMAL		1
#define OUTPUT_VERBOSE		2

const char *byte_to_binary(int);
void tobinary(unsigned char);
char *int2bin(unsigned int, char*, int);
#endif
