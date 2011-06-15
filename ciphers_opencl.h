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
#include <CL/opencl.h>

#include <openssl/aes.h>
void AES_opencl_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, int enc, cl_mem *device_buffer, cl_mem *device_schedule, cl_command_queue queue, cl_kernel device_kernel, cl_context context);
void AES_opencl_transfer_key_schedule(AES_KEY *ks, cl_mem *device_schedule,cl_command_queue queue);
void AES_opencl_transfer_iv(cl_context, const unsigned char*, cl_command_queue);
int AES_opencl_set_encrypt_key(const unsigned char *userKey, const int bits, AES_KEY *key);
int AES_opencl_set_decrypt_key(const unsigned char *userKey, const int bits, AES_KEY *key);

#include <openssl/blowfish.h>
void BF_opencl_transfer_key_schedule(BF_KEY *ks,cl_mem *device_schedule,cl_command_queue queue);
void BF_opencl_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, int enc, cl_mem *device_buffer, cl_mem *device_schedule, cl_command_queue queue, cl_kernel device_kernel, cl_context context);
void BF_opencl_transfer_iv(cl_context, const unsigned char*, cl_command_queue);

#include <openssl/des.h>
void DES_opencl_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, int enc, cl_mem *device_buffer, cl_mem *device_schedule, cl_command_queue queue, cl_kernel device_kernel, cl_context context);
void DES_opencl_transfer_key_schedule(DES_key_schedule *ks, cl_mem *device_schedule,cl_command_queue queue);
void DES_opencl_transfer_iv(cl_context, const unsigned char*, cl_command_queue);

#include <openssl/cast.h>
void CAST_opencl_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, int enc, cl_mem *device_buffer, cl_mem *device_schedule, cl_command_queue queue, cl_kernel device_kernel, cl_context context);
void CAST_opencl_transfer_key_schedule(CAST_KEY *ks, cl_mem *device_schedule,cl_command_queue queue);

#include <openssl/camellia.h>
void CMLL_opencl_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, int enc, cl_mem *device_buffer, cl_mem *device_schedule, cl_command_queue queue, cl_kernel device_kernel, cl_context context);
void CMLL_opencl_transfer_key_schedule(CAMELLIA_KEY *ks, cl_mem *device_schedule,cl_command_queue queue);

#include <openssl/idea.h>
void IDEA_opencl_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, int enc, cl_mem *device_buffer, cl_mem *device_schedule, cl_command_queue queue, cl_kernel device_kernel, cl_context context);
void IDEA_opencl_transfer_key_schedule(IDEA_KEY_SCHEDULE *ks, cl_mem *device_schedule,cl_command_queue queue);
