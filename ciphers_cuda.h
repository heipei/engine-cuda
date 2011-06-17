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
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <memory.h>
#include <openssl/engine.h>
#include <cuda_runtime_api.h>

#include "common.h"

#include <openssl/aes.h>
void AES_cuda_crypt(cuda_crypt_parameters *crypt);
void AES_cuda_transfer_key_schedule(AES_KEY *ks);
int AES_cuda_set_encrypt_key(const unsigned char *userKey, const int bits, AES_KEY *key);
int AES_cuda_set_decrypt_key(const unsigned char *userKey, const int bits, AES_KEY *key);
void AES_cuda_transfer_iv(const unsigned char *iv);

#include <openssl/blowfish.h>
void BF_cuda_transfer_key_schedule(const BF_KEY *key);
void BF_cuda_crypt(cuda_crypt_parameters *c);

void BF_cuda_transfer_iv(const unsigned char *iv);
void BF_cuda_decrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);
void BF_cuda_encrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);

#include <openssl/cast.h>
void CAST_cuda_transfer_key_schedule(const CAST_KEY *key);
void CAST_cuda_crypt(cuda_crypt_parameters *c);

void CAST_cuda_transfer_iv(const unsigned char *iv);
void CAST_cuda_decrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);
void CAST_cuda_encrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);

#include <openssl/camellia.h>
void CMLL_cuda_transfer_key_schedule(const CAMELLIA_KEY *key);
void CMLL_cuda_crypt(cuda_crypt_parameters *c);

void CMLL_cuda_transfer_iv(const unsigned char *iv);
void CMLL_cuda_decrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);
void CMLL_cuda_encrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);

#include <openssl/des.h>
void DES_cuda_transfer_key_schedule(const DES_key_schedule *key);
void DES_cuda_crypt(cuda_crypt_parameters *c);

void DES_cuda_transfer_iv(const unsigned char *iv);
void DES_cuda_decrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);
void DES_cuda_encrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);

#include <openssl/idea.h>
void IDEA_cuda_transfer_key_schedule(const IDEA_KEY_SCHEDULE *key);
void IDEA_cuda_crypt(cuda_crypt_parameters *c);

void IDEA_cuda_transfer_iv(const unsigned char *iv);
void IDEA_cuda_decrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);
void IDEA_cuda_encrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);
