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
#ifndef common_h
#define common_h

#define OUTPUT_QUIET		0
#define OUTPUT_NORMAL		1
#define OUTPUT_VERBOSE		2

#define NUM_BLOCK_PER_MULTIPROCESSOR	3
#define SIZE_BLOCK_PER_MULTIPROCESSOR	256*1024
#ifndef MAX_THREAD
	#define MAX_THREAD			256
#endif

#define STATE_THREAD_AES	4
#define AES_BLOCK_SIZE		16
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

void cuda_device_init(int *nm, int buffer_size, int output_verbosity, uint8_t **host_data, uint64_t **device_data, uint64_t**);
void cuda_device_finish(uint8_t *host_data, uint64_t *device_data);

#include <openssl/evp.h>
typedef struct cuda_crypt_parameters_st {
	const unsigned char *in;	// host input buffer 
	unsigned char *out;		// host output buffer
	size_t nbytes;			// number of bytes to be operated on
	EVP_CIPHER_CTX *ctx;		// EVP OpenSSL structure
	uint8_t **host_data;		// possible page-locked host memory
	uint64_t **d_in;		// Device memory (input)
	uint64_t **d_out;		// Device memory (output)
} cuda_crypt_parameters;


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
			(a << 56) | \
			((a & 0xff00) << 40) | \
			((a & 0xff0000) << 24) | \
			((a & 0xff000000) << 8)  | \
			((a & 0xff00000000) >> 8)  | \
			((a & 0xff0000000000) >> 24) | \
			((a & 0xff000000000000) >> 40) | \
			(a >> 56))

#endif
