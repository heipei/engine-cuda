/**
 * @version 0.1.2 (2010)
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
#include "des_cuda.h"
#include "idea_cuda.h"
#include "bf_cuda.h"
#include "cmll_cuda.h"
#include "cast_cuda.h"
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
static int cuda_aes_ciphers(EVP_CIPHER_CTX *ctx, unsigned char *out_arg, const unsigned char *in_arg, size_t nbytes);

static int num_multiprocessors = 0;
static int buffer_size = 0;
static int verbose = 0;
static int quiet = 0;
static int initialized = 0;
static char *library_path=NULL;
static int maxbytes = 8388608;

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
//	TODO: One finish for all ciphers

	//AES_cuda_finish();
	//DES_cuda_finish();
	//IDEA_cuda_finish();
	//BF_cuda_finish();
	//CMLL_cuda_finish();
	CAST_cuda_finish();
#ifdef CPU
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

//	TODO: One init for all ciphers
//	cuda_init(&num_multiprocessors,buffer_size,verbosity);
//	AES_cuda_init(&num_multiprocessors,buffer_size,verbosity);
//	DES_cuda_init(&num_multiprocessors,buffer_size,verbosity);
//	IDEA_cuda_init(&num_multiprocessors,buffer_size,verbosity);
//	BF_cuda_init(&num_multiprocessors,buffer_size,verbosity);
//	CMLL_cuda_init(&num_multiprocessors,buffer_size,verbosity);
	CAST_cuda_init(&num_multiprocessors,buffer_size,verbosity);
#ifdef CPU
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
	NID_camellia_128_ecb,
	NID_cast5_ecb,
	NID_des_ecb,
	NID_idea_ecb,
	NID_aes_128_ecb,
	NID_aes_128_cbc,
	NID_aes_192_ecb,
	NID_aes_192_cbc,
	NID_aes_256_ecb,
	NID_aes_256_cbc
};

static int cuda_cipher_nids_num = (sizeof(cuda_cipher_nids)/sizeof(cuda_cipher_nids[0]));

typedef struct cuda_aes_cipher_data { 
	AES_KEY ks; 
} cuda_aes_cipher_data;

static int cuda_aes_init_key (EVP_CIPHER_CTX *ctx, const unsigned char *key, const unsigned char *iv, int enc){
	if (!quiet && verbose) fprintf(stdout,"Start calculating key schedule...");
	cuda_aes_cipher_data *ccd;
	int key_len = EVP_CIPHER_CTX_key_length(ctx) * 8;
	ccd=(struct cuda_aes_cipher_data *)(ctx->cipher_data);
	if (ctx->encrypt) {
	switch(key_len) {
		case 128:
#ifdef CPU
			if(AES_set_encrypt_key(key,128,&ccd->ks)!=0) return 0;
#else
			if(AES_cuda_set_encrypt_key(key,128,&ccd->ks)!=0) return 0;
#endif
			break;
		case 192:
#ifdef CPU
			if(AES_set_encrypt_key(key,192,&ccd->ks)!=0) return 0;
#else
			if(AES_cuda_set_encrypt_key(key,192,&ccd->ks)!=0) return 0;
#endif
			break;
		case 256:
#ifdef CPU
			if(AES_set_encrypt_key(key,256,&ccd->ks)!=0) return 0;
#else
			if(AES_cuda_set_encrypt_key(key,256,&ccd->ks)!=0) return 0;
#endif
			break;
		default:
			return 0;
		}
	} else {
	switch(key_len) {
		case 128:
#ifdef CPU
			if(AES_set_decrypt_key(key,128,&ccd->ks)!=0) return 0;
#else
			if(AES_cuda_set_decrypt_key(key,128,&ccd->ks)!=0) return 0;
#endif
			break;
		case 192:
#ifdef CPU
			if(AES_set_decrypt_key(key,192,&ccd->ks)!=0) return 0;
#else
			if(AES_cuda_set_decrypt_key(key,192,&ccd->ks)!=0) return 0;
#endif
			break;
		case 256:
#ifdef CPU
			if(AES_set_decrypt_key(key,256,&ccd->ks)!=0) return 0;
#else
			if(AES_cuda_set_decrypt_key(key,256,&ccd->ks)!=0) return 0;
#endif
			break;
		default:
			return 0;
		}
	}
#ifndef CBC_ENC_CPU
	AES_cuda_transfer_iv(iv);
#endif
	return 1;
}

static int cuda_init_key(EVP_CIPHER_CTX *ctx, const unsigned char *key, const unsigned char *iv, int enc) {
	switch ((ctx->cipher)->nid) {
	  case NID_des_ecb:
	  case NID_des_cbc:
	    if (!quiet && verbose) fprintf(stdout,"Start calculating DES key schedule...");
	    DES_key_schedule des_key_schedule;
	    DES_set_key((const_DES_cblock *)key,&des_key_schedule);
	    DES_cuda_transfer_key_schedule(&des_key_schedule);
	    break;
	  case NID_bf_ecb:
	  case NID_bf_cbc:
	    if (!quiet && verbose) fprintf(stdout,"Start calculating Blowfish key schedule...\n");
	    BF_KEY key_schedule;
	    BF_set_key(&key_schedule,ctx->key_len,key);
	    BF_cuda_transfer_key_schedule(&key_schedule);
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
	    IDEA_KEY_SCHEDULE idea_key_schedule;
	    idea_set_encrypt_key(key,&idea_key_schedule);
	    IDEA_cuda_transfer_key_schedule(&idea_key_schedule);
	    break;
	  case NID_aes_128_ecb:
	  case NID_aes_128_cbc:
	  case NID_aes_192_ecb:
	  case NID_aes_192_cbc:
	  case NID_aes_256_ecb:
	  case NID_aes_256_cbc:
	    cuda_aes_init_key(ctx, key, iv, enc);
	    break;
	  default:
	    return 0;
	}
	if (!quiet && verbose) fprintf(stdout,"DONE!\n");
	return 1;
}

static int cuda_aes_ciphers(EVP_CIPHER_CTX *ctx, unsigned char *out_arg, const unsigned char *in_arg, size_t nbytes) {
	// EVP_CIPHER_CTX is defined in include/openssl/evp.h
	assert(in_arg && out_arg && ctx && nbytes);
	size_t current=0;
#if defined CPU
	//if (!quiet && verbose) fprintf(stdout,"C");
	cuda_aes_cipher_data *ccd;
	AES_KEY *ak;
#elif defined CBC_ENC_CPU
	//if (!quiet && verbose) fprintf(stdout,"G");
	cuda_aes_cipher_data *ccd;
	AES_KEY *ak;
	int chunk;
#else
	//if (!quiet && verbose) fprintf(stdout,"G");
	int chunk;
#endif
	switch (EVP_CIPHER_CTX_mode(ctx)) {
	case EVP_CIPH_ECB_MODE:
		if (ctx->encrypt) {
#ifdef CPU
			ccd=(struct cuda_aes_cipher_data *)(ctx->cipher_data);
			ak=&ccd->ks;
			while (nbytes!=current) {
				AES_encrypt((in_arg+current),(out_arg+current),ak);
				current+=AES_BLOCK_SIZE;
				}
#else
			while (nbytes!=current) {
				chunk=(nbytes-current)/(8*MAX_THREAD*STATE_THREAD_AES);
				if(chunk>=1) {
					AES_cuda_encrypt((in_arg+current),(out_arg+current),chunk*MAX_THREAD*STATE_THREAD_AES);
					current+=chunk*MAX_THREAD*STATE_THREAD_AES;	
					} else {
						AES_cuda_encrypt((in_arg+current),(out_arg+current),(nbytes-current));
						current+=(nbytes-current);
					}
				}
#endif
		} else {
#ifdef CPU
			ccd=(struct cuda_aes_cipher_data *)(ctx->cipher_data);
			ak=&ccd->ks;
			while (nbytes!=current) {
				AES_decrypt((in_arg+current),(out_arg+current),ak);
				current+=AES_BLOCK_SIZE;
				}
#else
			while (nbytes!=current) {
				chunk=(nbytes-current)/(MAX_THREAD*STATE_THREAD_AES);
				if(chunk>=1) {
					AES_cuda_decrypt((in_arg+current),(out_arg+current),chunk*MAX_THREAD*STATE_THREAD_AES);
					current+=chunk*MAX_THREAD*STATE_THREAD_AES;	
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
			ccd=(struct cuda_aes_cipher_data *)(ctx->cipher_data);
			ak=&ccd->ks;
			AES_cbc_encrypt(in_arg,out_arg,nbytes,ak,ctx->iv,AES_DECRYPT);
#else
			while (nbytes!=current) {
				chunk=(nbytes-current)/(MAX_THREAD*STATE_THREAD_AES);
				if(chunk>=1) {
					AES_cuda_transfer_iv(ctx->iv);
					AES_cuda_decrypt_cbc((in_arg+current),(out_arg+current),chunk*MAX_THREAD*STATE_THREAD_AES);
					current+=chunk*MAX_THREAD*STATE_THREAD_AES;	
					memcpy(ctx->iv,(in_arg+current-AES_BLOCK_SIZE),AES_BLOCK_SIZE);
					} else {
						AES_cuda_transfer_iv(ctx->iv);
						AES_cuda_decrypt_cbc((in_arg+current),(out_arg+current),(nbytes-current));
						current+=(nbytes-current);
						memcpy(ctx->iv,(in_arg+current-AES_BLOCK_SIZE),AES_BLOCK_SIZE);
					}
				}
#endif
		} else {
#if defined CPU || defined CBC_ENC_CPU
			ccd=(struct cuda_aes_cipher_data *)(ctx->cipher_data);
			ak=&ccd->ks;
			AES_cbc_encrypt(in_arg,out_arg,nbytes,ak,ctx->iv,AES_ENCRYPT);
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
			ccd=(struct cuda_aes_cipher_data *)(ctx->cipher_data);
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
			ccd=(struct cuda_aes_cipher_data *)(ctx->cipher_data);
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

static int cuda_crypt(EVP_CIPHER_CTX *ctx, unsigned char *out_arg, const unsigned char *in_arg, size_t nbytes) {
	assert(in_arg && out_arg && ctx && nbytes);
	size_t current=0;
	int chunk;

	switch(EVP_CIPHER_CTX_nid(ctx)) {
	  case NID_des_ecb:
	    while (nbytes!=current) {
	      chunk=(nbytes-current)/maxbytes;
	      if(chunk>=1) {
	        DES_cuda_crypt((in_arg+current),(out_arg+current),maxbytes,ctx->encrypt);
	        current+=maxbytes;  
	      } else {
	        DES_cuda_crypt((in_arg+current),(out_arg+current),(nbytes-current),ctx->encrypt);
	        current+=(nbytes-current);
	      }
	    }
	    break;

	  //case NID_bf_cbc:
	  case NID_bf_ecb:
	    while (nbytes!=current) {
	      chunk=(nbytes-current)/maxbytes;
	      if(chunk>=1) {
	        BF_cuda_crypt((in_arg+current),(out_arg+current),maxbytes,ctx->encrypt);
	        current+=maxbytes;  
	      } else {
	        BF_cuda_crypt((in_arg+current),(out_arg+current),(nbytes-current),ctx->encrypt);
	        current+=(nbytes-current);
	      }
	    }
	    break;
	  
	  case NID_cast5_ecb:
	    while (nbytes!=current) {
	      chunk=(nbytes-current)/maxbytes;
	      if(chunk>=1) {
	        CAST_cuda_crypt((in_arg+current),(out_arg+current),maxbytes,ctx->encrypt);
	        current+=maxbytes;  
	      } else {
	        CAST_cuda_crypt((in_arg+current),(out_arg+current),(nbytes-current),ctx->encrypt);
	        current+=(nbytes-current);
	      }
	    }
	    break;
	  
	  //case NID_camellia_128_cbc:
	  case NID_camellia_128_ecb:
	    while (nbytes!=current) {
	      chunk=(nbytes-current)/maxbytes;
	      if(chunk>=1) {
	        CMLL_cuda_crypt((in_arg+current),(out_arg+current),maxbytes,ctx->encrypt);
	        current+=maxbytes;  
	      } else {
	        CMLL_cuda_crypt((in_arg+current),(out_arg+current),(nbytes-current),ctx->encrypt);
	        current+=(nbytes-current);
	      }
	    }
	    break;
	  
	  //case NID_idea_cbc:
	  case NID_idea_ecb:
	    while (nbytes!=current) {
	      chunk=(nbytes-current)/maxbytes;
	      if(chunk>=1) {
	        IDEA_cuda_crypt((in_arg+current),(out_arg+current),maxbytes,ctx->encrypt);
	        current+=maxbytes;  
	      } else {
	        IDEA_cuda_crypt((in_arg+current),(out_arg+current),(nbytes-current),ctx->encrypt);
	        current+=(nbytes-current);
	      }
	    }
	    break;
	  
	  case NID_aes_128_ecb:
	  case NID_aes_128_cbc:
	  case NID_aes_192_ecb:
	  case NID_aes_192_cbc:
	  case NID_aes_256_ecb:
	  case NID_aes_256_cbc:
	    return cuda_aes_ciphers(ctx,out_arg,in_arg,nbytes);
	    break;
	  default:
	    return 0;
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
        sizeof(struct cuda_aes_cipher_data) + 16,             \
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
DECLARE_EVP(camellia,CAMELLIA,128,ecb,ECB);
DECLARE_EVP(cast,CAST,128,ecb,ECB);
DECLARE_EVP(des,DES,64,ecb,ECB);
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
	  case NID_camellia_128_ecb:
	    *cipher = &cuda_camellia_128_ecb;
	    break;
	  case NID_cast5_ecb:
	    *cipher = &cuda_cast5_ecb;
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
