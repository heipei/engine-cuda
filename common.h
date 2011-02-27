#ifndef common_h
#define common_h

#define OUTPUT_QUIET		0
#define OUTPUT_NORMAL		1
#define OUTPUT_VERBOSE		2

#define NUM_BLOCK_PER_MULTIPROCESSOR	3
#define SIZE_BLOCK_PER_MULTIPROCESSOR	256*1024
#define MAX_THREAD			128

#define STATE_THREAD_AES	4
#define AES_KEY_SIZE_128	16
#define AES_KEY_SIZE_192	24
#define AES_KEY_SIZE_256	32

#define STATE_THREAD_DES	2
#define DES_MAXNR		8
#define DES_BLOCK_SIZE		8
#define DES_KEY_SIZE		8
#define DES_KEY_SIZE_64		8

#define IDEA_BLOCK_SIZE		8
#define IDEA_KEY_SIZE		8
#define IDEA_KEY_SIZE_64	8

#define BF_BLOCK_SIZE		8
#define BF_KEY_SIZE_64		8
#define BF_KEY_SIZE_128		16

#define CAST_BLOCK_SIZE		8
#define CAST_KEY_SIZE_128	16

#define CMLL_BLOCK_SIZE		16
#define CMLL_KEY_SIZE_128	16
#define CMLL_KEY_SIZE_192	24
#define CMLL_KEY_SIZE_256	32

#define CAMELLIA_BLOCK_SIZE	16
#define CAMELLIA_KEY_SIZE_128	16
#define CAMELLIA_KEY_SIZE_192	24
#define CAMELLIA_KEY_SIZE_256	32

void cuda_device_init(int *nm, int buffer_size, int output_verbosity, uint8_t **host_data, uint64_t **device_data);
void cuda_device_finish(uint8_t *host_data, uint64_t *device_data);

// Split uint64_t into two uint32_t and convert each from BE to LE
#define nl2i(s,a,b)      a = ((s >> 24L) & 0x000000ff) | \
			     ((s >> 8L ) & 0x0000ff00) | \
			     ((s << 8L ) & 0x00ff0000) | \
			     ((s << 24L) & 0xff000000),   \
			 b = ((s >> 56L) & 0x000000ff) | \
			     ((s >> 40L) & 0x0000ff00) | \
			     ((s >> 24L) & 0x00ff0000) | \
			     ((s >> 8L) & 0xff000000)

// Convert uint64_t endianness
#define flip64(a)	(a= \
			((a & 0x00000000000000FF) << 56) | \
			((a & 0x000000000000FF00) << 40) | \
			((a & 0x0000000000FF0000) << 24) | \
			((a & 0x00000000FF000000) << 8)  | \
			((a & 0x000000FF00000000) >> 8)  | \
			((a & 0x0000FF0000000000) >> 24) | \
			((a & 0x00FF000000000000) >> 40) | \
			((a & 0xFF00000000000000) >> 56))

#endif
