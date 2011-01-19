/**
 * @version 0.1.1 - Copyright (c) 2010.
 *
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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <memory.h>
#include <openssl/aes.h>
#include <openssl/engine.h>
#include <cuda_runtime_api.h>


int AES_cuda_set_encrypt_key(const unsigned char *userKey, int bits, AES_KEY *key);
int AES_cuda_set_decrypt_key(const unsigned char *userKey, int bits, AES_KEY *key);

void AES_cuda_encrypt(const unsigned char *in, unsigned char *out,size_t nbytes);
void AES_cuda_decrypt(const unsigned char *in, unsigned char *out,size_t nbytes);

void AES_cuda_transfer_iv(const unsigned char *iv);
#ifndef CBC_ENC_CPU
void AES_cuda_encrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);
#endif
void AES_cuda_decrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);

void AES_cuda_init(int* nm,int buffer_size_engine,int verbosity);
void AES_cuda_finish();
