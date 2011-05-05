// vim:foldenable:foldmethod=marker:foldmarker=[[,]]
/**
 * @version 0.1.3 (2011)
 * @author Johannes Gilger <heipei@hackvalue.de>
 * 
 * Copyright 2011 Johannes Gilger
 *
 * This file is part of engine_cuda
 *
 * engine_cuda is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License or
 * any later version.
 * 
 * engine_cuda is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with engine_cuda. If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <memory.h>
#include <openssl/engine.h>
#include <openssl/evp.h>
#include <cuda_runtime_api.h>

#include <openssl/aes.h>
void AES_cuda_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, EVP_CIPHER_CTX *ctx, uint8_t **host_data, uint64_t **device_data);
void AES_cuda_transfer_key_schedule(AES_KEY *ks);
int AES_cuda_set_encrypt_key(const unsigned char *userKey, const int bits, AES_KEY *key);
int AES_cuda_set_decrypt_key(const unsigned char *userKey, const int bits, AES_KEY *key);
void AES_cuda_transfer_iv(const unsigned char *iv);

/*
void AES_cuda_init(int* nm,int buffer_size_engine,int verbosity);
void AES_cuda_finish();
*/

#include <openssl/blowfish.h>
void BF_cuda_transfer_key_schedule(const BF_KEY *key);
void BF_cuda_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, EVP_CIPHER_CTX *ctx, uint8_t **host_data, uint64_t **device_data);

void BF_cuda_transfer_iv(const unsigned char *iv);
void BF_cuda_decrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);
void BF_cuda_encrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);

#include <openssl/cast.h>
void CAST_cuda_transfer_key_schedule(const CAST_KEY *key);
void CAST_cuda_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, EVP_CIPHER_CTX *ctx, uint8_t **host_data, uint64_t **device_data);

void CAST_cuda_transfer_iv(const unsigned char *iv);
void CAST_cuda_decrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);
void CAST_cuda_encrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);

#include <openssl/camellia.h>
void CMLL_cuda_transfer_key_schedule(const CAMELLIA_KEY *key);
void CMLL_cuda_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, EVP_CIPHER_CTX *ctx, uint8_t **host_data, uint64_t **device_data);

void CMLL_cuda_transfer_iv(const unsigned char *iv);
void CMLL_cuda_decrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);
void CMLL_cuda_encrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);

#include <openssl/des.h>
void DES_cuda_transfer_key_schedule(const DES_key_schedule *key);
void DES_cuda_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, EVP_CIPHER_CTX *ctx, uint8_t **host_data, uint64_t **device_data);

void DES_cuda_transfer_iv(const unsigned char *iv);
void DES_cuda_decrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);
void DES_cuda_encrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);

#include <openssl/idea.h>
void IDEA_cuda_transfer_key_schedule(const IDEA_KEY_SCHEDULE *key);
void IDEA_cuda_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, EVP_CIPHER_CTX *ctx, uint8_t **host_data, uint64_t **device_data);

void IDEA_cuda_transfer_iv(const unsigned char *iv);
void IDEA_cuda_decrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);
void IDEA_cuda_encrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);
