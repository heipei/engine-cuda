/**
 * @version 0.1.0 - Copyright (c) 2010.
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
#include <openssl/blowfish.h>
#include <openssl/engine.h>
#include <CL/opencl.h>

void BF_opencl_transfer_key_schedule(BF_KEY *ks,cl_mem *device_schedule,cl_command_queue queue);
void BF_opencl_crypt(const unsigned char *in, unsigned char *out, size_t nbytes, int enc, cl_mem *device_buffer, cl_mem *device_schedule, cl_command_queue queue, cl_kernel device_kernel);

void BF_opencl_transfer_iv(const unsigned char *iv);
void BF_opencl_decrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);
void BF_opencl_encrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);
