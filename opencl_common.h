#include <stdint.h>
#include <CL/opencl.h>

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
