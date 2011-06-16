/**
 * @version 0.1.3 (2011)
 * @author Paolo Margara <paolo.margara@gmail.com>
 * 
 * Copyright 2011 Paolo Margara
 *
 * This file is part of engine-cuda.
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
 * along with engine-cuda.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "ciphers_cuda.h"
#include "common.h"

#define DYNAMIC_ENGINE
#define CUDA_ENGINE_ID		"cudamrg"
#define CUDA_ENGINE_NAME	"OpenSSL engine for AES, DES, IDEA, Blowfish, Camellia and CAST5 with CUDA acceleration"
#define CMD_SO_PATH		ENGINE_CMD_BASE
#define CMD_VERBOSE		(ENGINE_CMD_BASE+1)
#define CMD_QUIET		(ENGINE_CMD_BASE+2)
#define CMD_BUFFER_SIZE		(ENGINE_CMD_BASE+3)

static int cuda_ciphers (ENGINE *e, const EVP_CIPHER **cipher, const int **nids, int nid);
static int cuda_crypt(EVP_CIPHER_CTX *ctx, unsigned char *out_arg, const unsigned char *in_arg, size_t nbytes);
void (*cuda_device_crypt) (const unsigned char *in_arg, unsigned char *out_arg, size_t nbytes, EVP_CIPHER_CTX *ctx, uint8_t **host_data, uint64_t **device_data, uint64_t **device_data_out);
void (*transferHostToDevice) (const unsigned char  **input, uint32_t **deviceMem, uint8_t **hostMem, size_t *size);
void (*transferDeviceToHost) (      unsigned char **output, uint32_t **deviceMem, uint8_t **hostMemS, uint8_t **hostMemOUT, size_t *size);

static int num_multiprocessors = 0;
static int buffer_size = 0;
#ifdef DEBUG
	static int verbose = 2;
#else
	static int verbose = 0;
#endif
static int quiet = 0;
static int initialized = 0;
static char *library_path=NULL;
static int maxbytes = 8388608;

static __device__ uint64_t *device_data;
static __device__ uint64_t *device_data_out;
static uint8_t *host_data;

int set_buffer_size(const char *buffer_size_string) {
	buffer_size=atoi(buffer_size_string)*1024;	// The size is in kilobytes
	return 1;
}

int inc_quiet(void) {
	quiet++;
	return 1;
}

int inc_verbose(void) {
	verbose++;
	return 1;
}

int cuda_finish(ENGINE * engine) {
	cuda_device_finish(host_data,device_data);

	return 1;
}

int cuda_init(ENGINE * engine) {
	if (!quiet && verbose) fprintf(stdout, "initializing engine\n");
	int verbosity=OUTPUT_NORMAL;
	if (quiet==1)
		verbosity=OUTPUT_QUIET;
	if(verbose) 
		verbosity=OUTPUT_VERBOSE;

	cuda_device_init(&num_multiprocessors,buffer_size,verbosity,&host_data,&device_data,&device_data_out);

	initialized=1;
	return 1;
}

static const ENGINE_CMD_DEFN cuda_cmd_defns[] = {
	{CMD_SO_PATH, "SO_PATH", "Specifies the path to the 'cuda-engine' shared library", ENGINE_CMD_FLAG_STRING},
	{CMD_VERBOSE, "VERBOSE", "Print additional details", ENGINE_CMD_FLAG_NO_INPUT},
	{CMD_QUIET, "QUIET", "Remove additional details", ENGINE_CMD_FLAG_NO_INPUT},
	{CMD_BUFFER_SIZE, "BUFFER_SIZE", "Specifies the size of the buffer between central memory and GPU memory in kilobytes (default: 8MB)", ENGINE_CMD_FLAG_STRING},
	{0, NULL, NULL, 0}
};

static int cuda_engine_ctrl(ENGINE * e, int cmd, long i, void *p, void (*f) ()) {
	switch (cmd) {
	case CMD_SO_PATH:
		if (p!=NULL && library_path==NULL) {
			library_path=strdup((const char*)p);
			return 1;
		} else return 0;
	case CMD_QUIET:
		if (initialized) { 
			if (!quiet) fprintf(stderr,"Error: you cannot set command %d when the engine is already initialized.",cmd);
			return 0;
		} else return inc_quiet();
	case CMD_VERBOSE:
		if (initialized) {
			if (!quiet) fprintf(stderr,"Error: you cannot set command %d when the engine is already initialized.",cmd);
			return 0;
		} else return inc_verbose();
	case CMD_BUFFER_SIZE:
		if (initialized) {
			if (!quiet) fprintf(stderr,"Error: you cannot set command %d when the engine is already initialized.",cmd);
			return 0;
		} else return set_buffer_size((const char *)p);
	default:
		break;
	}
	if (!quiet) fprintf(stderr,"Command not implemented: %d - %s",cmd,(char*)p);
	return 0;
}

static int cuda_bind_helper(ENGINE * e) {
	if (!ENGINE_set_id(e, CUDA_ENGINE_ID) ||
	    !ENGINE_set_init_function(e, cuda_init) ||
	    !ENGINE_set_finish_function(e, cuda_finish) ||
	    !ENGINE_set_ctrl_function(e, cuda_engine_ctrl) ||
	    !ENGINE_set_cmd_defns(e, cuda_cmd_defns) ||
	    !ENGINE_set_name(e, CUDA_ENGINE_NAME) ||
	    !ENGINE_set_ciphers (e, cuda_ciphers)) {
		return 0;
	} else {
		return 1;
	}
}

static int cuda_bind_fn(ENGINE *e, const char *id) {
	if (id && (strcmp(id, CUDA_ENGINE_ID) != 0)) {
		if (!quiet) fprintf(stderr, "bad engine id\n");
		return 0;
	}
	if (!cuda_bind_helper(e))  {
		if (!quiet) fprintf(stderr, "bind failed\n");
		return 0;
	}

	return 1;
}

IMPLEMENT_DYNAMIC_CHECK_FN()
IMPLEMENT_DYNAMIC_BIND_FN(cuda_bind_fn)

static int cuda_cipher_nids[] = {
	NID_bf_ecb,
	NID_bf_cbc,
	NID_camellia_128_ecb,
	NID_cast5_ecb,
	NID_des_ecb,
	NID_des_cbc,
	NID_idea_ecb,
	NID_aes_128_ecb,
	NID_aes_128_cbc,
	NID_aes_192_ecb,
	NID_aes_192_cbc,
	NID_aes_256_ecb,
	NID_aes_256_cbc
};

static int cuda_cipher_nids_num = (sizeof(cuda_cipher_nids)/sizeof(cuda_cipher_nids[0]));

static int cuda_init_key(EVP_CIPHER_CTX *ctx, const unsigned char *key, const unsigned char *iv, int enc) {
	switch ((ctx->cipher)->nid) {
	  case NID_des_ecb:
	  case NID_des_cbc:
	    if (!quiet && verbose) fprintf(stdout,"Start calculating DES key schedule...");
	    DES_key_schedule des_key_schedule;
	    DES_set_key((const_DES_cblock *)key,&des_key_schedule);
	    DES_cuda_transfer_key_schedule(&des_key_schedule);
	    if(iv)
	    	DES_cuda_transfer_iv(iv);
	    break;
	  case NID_bf_ecb:
	  case NID_bf_cbc:
	    if (!quiet && verbose) fprintf(stdout,"Start calculating Blowfish key schedule...\n");
	    BF_KEY key_schedule;
	    BF_set_key(&key_schedule,ctx->key_len,key);
	    BF_cuda_transfer_key_schedule(&key_schedule);
	    if(iv)
		BF_cuda_transfer_iv(iv);
	    break;
	  case NID_cast5_ecb:
	    if (!quiet && verbose) fprintf(stdout,"Start calculating CAST5 key schedule...\n");
	    CAST_KEY cast_key_schedule;
	    CAST_set_key(&cast_key_schedule,ctx->key_len*8,key);
	    CAST_cuda_transfer_key_schedule(&cast_key_schedule);
	    break;
	  case NID_camellia_128_ecb:
	  case NID_camellia_128_cbc:
	    if (!quiet && verbose) fprintf(stdout,"Start calculating Camellia key schedule...\n");
	    CAMELLIA_KEY cmll_key_schedule;
	    Camellia_set_key(key,ctx->key_len*8,&cmll_key_schedule);
	    CMLL_cuda_transfer_key_schedule(&cmll_key_schedule);
	    break;
	  case NID_idea_ecb:
	  case NID_idea_cbc:
	    if (!quiet && verbose) fprintf(stdout,"Start calculating IDEA key schedule...\n");
	    {
	    IDEA_KEY_SCHEDULE idea_key_schedule, idea_dec_key_schedule;
	    idea_set_encrypt_key(key,&idea_key_schedule);
	    if(!(ctx->encrypt)) {
	      idea_set_decrypt_key(&idea_key_schedule,&idea_dec_key_schedule);
	      IDEA_cuda_transfer_key_schedule(&idea_dec_key_schedule);
	    } else {
	      IDEA_cuda_transfer_key_schedule(&idea_key_schedule);
	    }
	    }
	    break;
	  case NID_aes_128_ecb:
	  case NID_aes_128_cbc:
	    {
	    if (!quiet && verbose) fprintf(stdout,"Start calculating AES-128 key schedule...\n");
	    AES_KEY aes_key_schedule;
	    if(ctx->encrypt)
	    	AES_cuda_set_encrypt_key(key,128,&aes_key_schedule);
	    else
	    	AES_cuda_set_decrypt_key(key,128,&aes_key_schedule);
	    AES_cuda_transfer_key_schedule(&aes_key_schedule);
	    }
	    if(iv)
		AES_cuda_transfer_iv(iv);
	    break;
	  case NID_aes_192_ecb:
	  case NID_aes_192_cbc:
	    if (!quiet && verbose) fprintf(stdout,"Start calculating AES-192 key schedule...\n");
	    {
	    AES_KEY aes_key_schedule;
	    if(ctx->encrypt)
	    	AES_cuda_set_encrypt_key(key,192,&aes_key_schedule);
	    else
	    	AES_cuda_set_decrypt_key(key,192,&aes_key_schedule);
	    AES_cuda_transfer_key_schedule(&aes_key_schedule);
	    if(iv)
		AES_cuda_transfer_iv(iv);
	    }
	    break;
	  case NID_aes_256_ecb:
	  case NID_aes_256_cbc:
	    if (!quiet && verbose) fprintf(stdout,"Start calculating AES-256 key schedule...\n");
	    {
	    AES_KEY aes_key_schedule;
	    if(ctx->encrypt)
	    	AES_cuda_set_encrypt_key(key,256,&aes_key_schedule);
	    else
	    	AES_cuda_set_decrypt_key(key,256,&aes_key_schedule);
	    AES_cuda_transfer_key_schedule(&aes_key_schedule);
	    if(iv)
		AES_cuda_transfer_iv(iv);
	    }
	    break;
	  default:
	    return 0;
	}
	if (!quiet && verbose) fprintf(stdout,"DONE!\n");
	return 1;
}

static int cuda_crypt(EVP_CIPHER_CTX *ctx, unsigned char *out_arg, const unsigned char *in_arg, size_t nbytes) {
	assert(in_arg && out_arg && ctx && nbytes);
	size_t current=0;
	int chunk;

	switch(EVP_CIPHER_CTX_nid(ctx)) {
	  case NID_des_ecb:
	  case NID_des_cbc:
	    cuda_device_crypt = DES_cuda_crypt;
	    break;
	  case NID_bf_ecb:
	  case NID_bf_cbc:
	    cuda_device_crypt = BF_cuda_crypt;
	    break;
	  case NID_cast5_ecb:
	    cuda_device_crypt = CAST_cuda_crypt;
	    break;
	  case NID_camellia_128_ecb:
	    cuda_device_crypt = CMLL_cuda_crypt;
	    break;
	  case NID_idea_ecb:
	    cuda_device_crypt = IDEA_cuda_crypt;
	    break;
	  case NID_aes_128_ecb:
	  case NID_aes_128_cbc:
	  case NID_aes_192_ecb:
	  case NID_aes_192_cbc:
	  case NID_aes_256_ecb:
	  case NID_aes_256_cbc:
	    cuda_device_crypt = AES_cuda_crypt;
	    break;
	  default:
	    return 0;
	}

	while (nbytes!=current) {
		chunk=(nbytes-current)/maxbytes;
		if(chunk>=1) {
			cuda_device_crypt((in_arg+current),(out_arg+current),maxbytes,ctx,&host_data,&device_data,&device_data_out);
			current+=maxbytes;  
		} else {
			cuda_device_crypt((in_arg+current),(out_arg+current),(nbytes-current),ctx,&host_data,&device_data,&device_data_out);
			current+=(nbytes-current);
		}
	}

	return 1;
}

#define DECLARE_EVP(lciph,uciph,ksize,lmode,umode)            \
static const EVP_CIPHER cuda_##lciph##_##ksize##_##lmode = {  \
        NID_##lciph##_##ksize##_##lmode,                      \
        uciph##_BLOCK_SIZE,                                   \
        uciph##_KEY_SIZE_##ksize,                             \
        uciph##_BLOCK_SIZE,                                   \
        0 | EVP_CIPH_##umode##_MODE,                          \
        cuda_init_key,                                        \
        cuda_crypt,                                           \
        NULL,                                                 \
        sizeof(AES_KEY) + 16,             \
        EVP_CIPHER_set_asn1_iv,                               \
        EVP_CIPHER_get_asn1_iv,                               \
        NULL,                                                 \
        NULL                                                  \
}

#define cuda_des_64_ecb cuda_des_ecb
#define cuda_des_64_cbc cuda_des_cbc
#define NID_des_64_ecb NID_des_ecb
#define NID_des_64_cbc NID_des_cbc

#define cuda_cast5_ecb cuda_cast_ecb
#define cuda_cast_128_ecb cuda_cast_ecb
#define NID_cast_128_ecb NID_cast5_ecb

#define cuda_bf_128_ecb cuda_bf_ecb
#define cuda_bf_128_cbc cuda_bf_cbc
#define NID_bf_128_ecb NID_bf_ecb
#define NID_bf_128_cbc NID_bf_cbc

#define cuda_idea_64_ecb cuda_idea_ecb
#define cuda_idea_64_cbc cuda_idea_cbc
#define NID_idea_64_ecb NID_idea_ecb
#define NID_idea_64_cbc NID_idea_cbc

DECLARE_EVP(bf,BF,128,ecb,ECB);
DECLARE_EVP(bf,BF,128,cbc,CBC);
DECLARE_EVP(camellia,CAMELLIA,128,ecb,ECB);
DECLARE_EVP(cast,CAST,128,ecb,ECB);
DECLARE_EVP(des,DES,64,ecb,ECB);
DECLARE_EVP(des,DES,64,cbc,CBC);
DECLARE_EVP(idea,IDEA,64,ecb,ECB);

DECLARE_EVP(aes,AES,128,ecb,ECB);
DECLARE_EVP(aes,AES,128,cbc,CBC);
DECLARE_EVP(aes,AES,192,ecb,ECB);
DECLARE_EVP(aes,AES,192,cbc,CBC);
DECLARE_EVP(aes,AES,256,ecb,ECB);
DECLARE_EVP(aes,AES,256,cbc,CBC);

static int cuda_ciphers(ENGINE *e, const EVP_CIPHER **cipher, const int **nids, int nid) {
	if (!cipher) {
		*nids = cuda_cipher_nids;
		return cuda_cipher_nids_num;
	}
	switch (nid) {
	  case NID_bf_ecb:
	    *cipher = &cuda_bf_ecb;
	    break;
	  case NID_bf_cbc:
	    *cipher = &cuda_bf_cbc;
	    break;
	  case NID_camellia_128_ecb:
	    *cipher = &cuda_camellia_128_ecb;
	    break;
	  case NID_cast5_ecb:
	    *cipher = &cuda_cast5_ecb;
	    break;
	  case NID_des_cbc:
	    *cipher = &cuda_des_cbc;
	    break;
	  case NID_des_ecb:
	    *cipher = &cuda_des_ecb;
	    break;
	  case NID_idea_ecb:
	    *cipher = &cuda_idea_ecb;
	    break;
	  case NID_aes_128_ecb:
	    *cipher = &cuda_aes_128_ecb;
	    break;
	  case NID_aes_128_cbc:
	    *cipher = &cuda_aes_128_cbc;
	    break;
	  case NID_aes_192_ecb:
	    *cipher = &cuda_aes_192_ecb;
	    break;
	  case NID_aes_192_cbc:
	    *cipher = &cuda_aes_192_cbc;
	    break;
	  case NID_aes_256_ecb:
	    *cipher = &cuda_aes_256_ecb;
	    break;
	  case NID_aes_256_cbc:
	    *cipher = &cuda_aes_256_cbc;
	    break;
	  default:
	    *cipher = NULL;
	    return 0;
	}
	return 1;
}
