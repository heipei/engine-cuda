/**
 * @version 0.1.0 (2010)
 * @author Paolo Margara <paolo.margara@gmail.com>
 * 
 * Copyright 2010 Paolo Margara
 *
 * This file is part of Engine_cudamrg.
 *
 * Engine_cudamrg is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License or
 * any later version.
 * 
 * Engine_cudamrg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Engine_cudamrg.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "aes_cuda.h"

static int cuda_ciphers (ENGINE *e, const EVP_CIPHER **cipher, const int **nids, int nid);
static int cuda_aes_ciphers(EVP_CIPHER_CTX *ctx, unsigned char *out_arg, const unsigned char *in_arg, size_t nbytes);

#define DYNAMIC_ENGINE
#define CUDA_ENGINE_ID		"cudamrg"
#define CUDA_ENGINE_NAME	"cuda engine for AES encrypting/decrypting developed by MRG"
#define CMD_SO_PATH		ENGINE_CMD_BASE
#define CMD_VERBOSE		(ENGINE_CMD_BASE+1)
#define CMD_QUIET		(ENGINE_CMD_BASE+2)
#define CMD_BUFFER_SIZE		(ENGINE_CMD_BASE+3)

static int num_multiprocessors = 0;
static int buffer_size = 0;
static int verbose = 0;
static int quiet = 0;
static int initialized = 0;
static char *library_path=NULL;

#ifdef CPU
static time_t startTime,endTime;
#endif

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
#ifndef CPU
	AES_cuda_finish();
#else
	AES_cuda_finish();
	endTime=time(NULL);
	//if (!quiet) fprintf(stdout,"\nTotal time: %g seconds\n",difftime(endTime,startTime));
#endif
	return 1;
}

int cuda_init(ENGINE * engine) {
	if (!quiet && verbose) fprintf(stdout, "initializing engine\n");
	int verbosity;
	if(verbose) {
		verbosity=OUTPUT_VERBOSE;
	} else { 
	if (quiet==1)
		verbosity=OUTPUT_QUIET;
	else 
		verbosity=OUTPUT_NORMAL;
	}
#ifndef CPU
	int deviceCount;
	cudaError_t cudaerrno;

	cudaGetDeviceCount(&deviceCount);
	cudaerrno=cudaGetLastError();
	if( cudaSuccess != cudaerrno) {
		if (!quiet) fprintf(stderr,"Cuda error: %s.\n",cudaGetErrorString(cudaerrno));
		return 0;
		}
	if (deviceCount == 0) {
		if (!quiet) fprintf(stderr,"\nError: no devices supporting CUDA.\n");
        	return 0;
		} else {
		if (!quiet) fprintf(stdout,"\nSuccessfully found a device supporting CUDA.\n");
		}
	AES_cuda_init(&num_multiprocessors,buffer_size,verbosity);
#else	
	AES_cuda_init(&num_multiprocessors,buffer_size,verbosity);
	startTime=time(NULL);
#endif
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

static ENGINE * ENGINE_cuda(void) {
	ENGINE *eng = ENGINE_new();
	if (!eng) {
		return NULL;
	}
	if (!cuda_bind_helper(eng)) {
		ENGINE_free(eng);
		return NULL;
	}
	return eng;
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
	NID_aes_128_ecb,
	NID_aes_128_cbc,
#if 0
	NID_aes_128_cfb128,
	NID_aes_128_ofb128,
#endif
	NID_aes_192_ecb,
	NID_aes_192_cbc,
#if 0
	NID_aes_192_cfb128,
	NID_aes_192_ofb128,
#endif
	NID_aes_256_ecb,
	NID_aes_256_cbc,
#if 0
	NID_aes_256_cfb128,
	NID_aes_256_ofb128,
#endif
};

static int cuda_cipher_nids_num = (sizeof(cuda_cipher_nids)/sizeof(cuda_cipher_nids[0]));

#define EVP_CIPHER_block_size_ECB	AES_BLOCK_SIZE
#define EVP_CIPHER_block_size_CBC	AES_BLOCK_SIZE
#if 0
#define EVP_CIPHER_block_size_OFB	1
#define EVP_CIPHER_block_size_CFB	1
#endif

typedef struct cuda_cipher_data { 
	AES_KEY ks; 
	} CUDA_CIPHER_DATA;

static int cuda_aes_init_key (EVP_CIPHER_CTX *ctx, const unsigned char *key, const unsigned char *iv, int enc){
	if (!quiet && verbose) fprintf(stdout,"Start calculating key schedule...");
	CUDA_CIPHER_DATA *ccd;
	int key_len = EVP_CIPHER_CTX_key_length(ctx) * 8;
	ccd=(struct cuda_cipher_data *)(ctx->cipher_data);
	if (ctx->encrypt) {
	switch(key_len) {
		case 128:
			if(AES_set_encrypt_key(key,128,&ccd->ks)!=0) return 0;
#ifndef CPU
			AES_cuda_transfer_key(&ccd->ks);
#endif
			break;
		case 192:
			if(AES_set_encrypt_key(key,192,&ccd->ks)!=0) return 0;
#ifndef CPU
			AES_cuda_transfer_key(&ccd->ks);
#endif
			break;
		case 256:
			if(AES_set_encrypt_key(key,256,&ccd->ks)!=0) return 0;
#ifndef CPU
			AES_cuda_transfer_key(&ccd->ks);
#endif
			break;
		default:
			return 0;
		}
	} else {
	switch(key_len) {
		case 128:
			if(AES_set_decrypt_key(key,128,&ccd->ks)!=0) return 0;
#ifndef CPU
			AES_cuda_transfer_key(&ccd->ks);
#endif
			break;
		case 192:
			if(AES_set_decrypt_key(key,192,&ccd->ks)!=0) return 0;
#ifndef CPU
			AES_cuda_transfer_key(&ccd->ks);
#endif
			break;
		case 256:
			if(AES_set_decrypt_key(key,256,&ccd->ks)!=0) return 0;
#ifndef CPU
			AES_cuda_transfer_key(&ccd->ks);
#endif
			break;
		default:
			return 0;
		}
	}
	
	AES_cuda_transfer_iv(iv);

	if (!quiet && verbose) fprintf(stdout,"DONE!\n");
	return 1;
}

#define	DECLARE_AES_EVP(ksize,lmode,umode)	\
static const EVP_CIPHER cuda_aes_##ksize##_##lmode = {	\
	NID_aes_##ksize##_##lmode,	\
	EVP_CIPHER_block_size_##umode,	\
	AES_KEY_SIZE_##ksize,		\
	AES_BLOCK_SIZE,			\
	0 | EVP_CIPH_##umode##_MODE,	\
	cuda_aes_init_key,		\
	cuda_aes_ciphers,		\
	NULL,				\
	sizeof(struct cuda_cipher_data) + 16,	\
	EVP_CIPHER_set_asn1_iv,		\
	EVP_CIPHER_get_asn1_iv,		\
	NULL,				\
	NULL				\
}

DECLARE_AES_EVP(128,ecb,ECB);
DECLARE_AES_EVP(128,cbc,CBC);
#if 0
DECLARE_AES_EVP(128,cfb128,CFB);
DECLARE_AES_EVP(128,ofb128,OFB);
#endif
DECLARE_AES_EVP(192,ecb,ECB);
DECLARE_AES_EVP(192,cbc,CBC);
#if 0
DECLARE_AES_EVP(192,cfb128,CFB);
DECLARE_AES_EVP(192,ofb128,OFB);
#endif 
DECLARE_AES_EVP(256,ecb,ECB);
DECLARE_AES_EVP(256,cbc,CBC);
#if 0
DECLARE_AES_EVP(256,cfb128,CFB);
DECLARE_AES_EVP(256,ofb128,OFB);
#endif

static int cuda_aes_ciphers(EVP_CIPHER_CTX *ctx, unsigned char *out_arg, const unsigned char *in_arg, size_t nbytes) {
	// EVP_CIPHER_CTX is defined in include/openssl/evp.h
	assert(in_arg && out_arg && ctx && nbytes);
	size_t current=0;
#if 0
	int n;
#endif
#ifdef CPU
	//if (!quiet && verbose) fprintf(stdout,"C");
	CUDA_CIPHER_DATA *ccd;
	AES_KEY *ak;
#else
	//if (!quiet && verbose) fprintf(stdout,"G");
	int chunk;
#endif
	switch (EVP_CIPHER_CTX_mode(ctx)) {
	case EVP_CIPH_ECB_MODE:
		if (ctx->encrypt) {
#ifdef CPU
			ccd=(struct cuda_cipher_data *)(ctx->cipher_data);
			ak=&ccd->ks;
			while (nbytes!=current) {
				AES_encrypt((in_arg+current),(out_arg+current),ak);
				current+=AES_BLOCK_SIZE;
				}
#else
			while (nbytes!=current) {
				chunk=(nbytes-current)/(MAX_THREAD*STATE_THREAD);
				if(chunk>=1) {
					AES_cuda_encrypt((in_arg+current),(out_arg+current),chunk*MAX_THREAD*STATE_THREAD);
					current+=chunk*MAX_THREAD*STATE_THREAD;	
					} else {
						AES_cuda_encrypt((in_arg+current),(out_arg+current),(nbytes-current));
						current+=(nbytes-current);
					}
				}
#endif
		} else {
#ifdef CPU
			ccd=(struct cuda_cipher_data *)(ctx->cipher_data);
			ak=&ccd->ks;
			while (nbytes!=current) {
				AES_decrypt((in_arg+current),(out_arg+current),ak);
				current+=AES_BLOCK_SIZE;
				}
#else
			while (nbytes!=current) {
				chunk=(nbytes-current)/(MAX_THREAD*STATE_THREAD);
				if(chunk>=1) {
					AES_cuda_decrypt((in_arg+current),(out_arg+current),chunk*MAX_THREAD*STATE_THREAD);
					current+=chunk*MAX_THREAD*STATE_THREAD;	
					} else {
						AES_cuda_decrypt((in_arg+current),(out_arg+current),(nbytes-current));
						current+=(nbytes-current);
					}
				}
#endif
			}
		break;

	case EVP_CIPH_CBC_MODE:
		if (!ctx->encrypt) {
#ifdef CPU
			ccd=(struct cuda_cipher_data *)(ctx->cipher_data);
			ak=&ccd->ks;
			AES_cbc_encrypt(in_arg,out_arg,nbytes,ak,ctx->iv,AES_DECRYPT);
#else
			while (nbytes!=current) {
				chunk=(nbytes-current)/(MAX_THREAD*STATE_THREAD);
				if(chunk>=1) {
					AES_cuda_decrypt_cbc((in_arg+current),(out_arg+current),chunk*MAX_THREAD*STATE_THREAD);
					current+=chunk*MAX_THREAD*STATE_THREAD;	
					} else {
						AES_cuda_decrypt_cbc((in_arg+current),(out_arg+current),(nbytes-current));
						current+=(nbytes-current);
					}
				}
#endif
		} else {
#ifdef CPU
			ccd=(struct cuda_cipher_data *)(ctx->cipher_data);
			ak=&ccd->ks;
			AES_cbc_decrypt(in_arg,out_arg,nbytes,ak,ctx->iv,AES_ENCRYPT);
#else
			while (nbytes!=current) {
				AES_cuda_encrypt_cbc((in_arg+current),(out_arg+current),(nbytes-current));
				current+=(nbytes-current);
				}
#endif
			}
		break;
#if 0
	case EVP_CIPH_CFB_MODE:
		if (ctx->encrypt) {
#ifdef CPU
			ccd=(struct cuda_cipher_data *)(ctx->cipher_data);
			ak=&ccd->ks;
			n=ctx->num;
			while (nbytes--) {
				if (n == 0) AES_encrypt(ctx->iv,ctx->iv,ak);
				ctx->iv[n] = *(out_arg++) = *(in_arg++) ^ ctx->iv[n];
				n = (n+1) % AES_BLOCK_SIZE;
				}
			ctx->num=n;
#else
			n=ctx->num;
			while (nbytes--) {
				if (n == 0) AES_cuda_encrypt(ctx->iv,ctx->iv,AES_BLOCK_SIZE);
				ctx->iv[n] = *(out_arg++) = *(in_arg++) ^ ctx->iv[n];
				n = (n+1) % AES_BLOCK_SIZE;
				}
			ctx->num=n;
#endif
			} else {
				if (!quiet) fprintf(stderr,"\nError: Decryption in CFB mode not yet supported.\n\n");
				exit(EXIT_FAILURE);
			}
		break;
	case EVP_CIPH_OFB_MODE:
		if (ctx->encrypt) {
#ifdef CPU
			ccd=(struct cuda_cipher_data *)(ctx->cipher_data);
			ak=&ccd->ks;
			n=ctx->num;
			while (nbytes--) {
				if (n == 0) AES_encrypt(ctx->iv, ctx->iv, ak);
				*(out_arg++) = *(in_arg++) ^ ctx->iv[n];
				n = (n+1) % AES_BLOCK_SIZE;
				}
			ctx->num=n;
#else
			n=ctx->num;
			while (nbytes--) {
				if (n == 0) AES_cuda_encrypt(ctx->iv, ctx->iv, AES_BLOCK_SIZE);
				*(out_arg++) = *(in_arg++) ^ ctx->iv[n];
				n = (n+1) % AES_BLOCK_SIZE;
				}
			ctx->num=n;
#endif
			} else {
				if (!quiet) fprintf(stderr,"\nError: Decryption in OFB mode not yet supported.\n\n");
				exit(EXIT_FAILURE);
			}
		break;
#endif
	default:
		return 0;
	}
	return 1;
}

static int cuda_ciphers(ENGINE *e, const EVP_CIPHER **cipher, const int **nids, int nid) {
	if (!cipher) {
		*nids = cuda_cipher_nids;
		return cuda_cipher_nids_num;
	}
	switch (nid) {
	  case NID_aes_128_ecb:
	    *cipher = &cuda_aes_128_ecb;
	    break;
	  case NID_aes_128_cbc:
	    *cipher = &cuda_aes_128_cbc;
	    break;
#if 0
	  case NID_aes_128_cfb128:
	    *cipher = &cuda_aes_128_cfb128;
	    break;
	  case NID_aes_128_ofb128:
	    *cipher = &cuda_aes_128_ofb128;
	    break;
#endif
	  case NID_aes_192_ecb:
	    *cipher = &cuda_aes_192_ecb;
	    break;
	  case NID_aes_192_cbc:
	    *cipher = &cuda_aes_192_cbc;
	    break;
#if 0
	  case NID_aes_192_cfb128:
	    *cipher = &cuda_aes_192_cfb128;
	    break;
	  case NID_aes_192_ofb128:
	    *cipher = &cuda_aes_192_ofb128;
	    break;
#endif
	  case NID_aes_256_ecb:
	    *cipher = &cuda_aes_256_ecb;
	    break;
	  case NID_aes_256_cbc:
	    *cipher = &cuda_aes_256_cbc;
	    break;
#if 0
	  case NID_aes_256_cfb128:
	    *cipher = &cuda_aes_256_cfb128;
	    break;
	  case NID_aes_256_ofb128:
	    *cipher = &cuda_aes_256_ofb128;
	    break;
#endif
	  default:
	    *cipher = NULL;
	    return 0;
	}
	return 1;
}
