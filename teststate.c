/**
 * @version 0.1.1 (2010) - Copyright (c) 2010.
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

#include "aes_cuda.h"

int main(int argc, char **argv) {
	/* AES Test vector */
	uint8_t input[AES_BLOCK_SIZE] = { 0x4E, 0xC1, 0x37, 0xA4, 0x26, 0xDA, 0xBF,
			0x8A, 0xA0, 0xBE, 0xB8, 0xBC, 0x0C, 0x2B, 0x89, 0xD6 };
	uint8_t enc_output128[AES_BLOCK_SIZE] = { 0xD9, 0xB6, 0x5D, 0x12, 0x32,
			0xBA, 0x01, 0x99, 0xCD, 0xBD, 0x48, 0x7B, 0x2A, 0x1F, 0xD6, 0x46 };
	uint8_t enc_output192[AES_BLOCK_SIZE] = { 0xB1, 0x8B, 0xB3, 0xE7, 0xE1,
			0x07, 0x32, 0xBE, 0x13, 0x58, 0x44, 0x3A, 0x50, 0x4D, 0xBB, 0x49 };
	uint8_t enc_output256[AES_BLOCK_SIZE] = { 0x2F, 0x9C, 0xFD, 0xDB, 0xFF,
			0xCD, 0xE6, 0xB9, 0xF3, 0x7E, 0xF8, 0xE4, 0x0D, 0x51, 0x2C, 0xF4 };
	uint8_t dec_output128[AES_BLOCK_SIZE] = { 0x95, 0x70, 0xC3, 0x43, 0x63,
			0x56, 0x5B, 0x39, 0x35, 0x03, 0xA0, 0x01, 0xC0, 0xE2, 0x3B, 0x65 };
	uint8_t dec_output192[AES_BLOCK_SIZE] = { 0x29, 0xDF, 0xD7, 0x5B, 0x85,
			0xCE, 0xE4, 0xDE, 0x6E, 0x26, 0xA8, 0x08, 0xCD, 0xC2, 0xC9, 0xC3 };
	uint8_t dec_output256[AES_BLOCK_SIZE] = { 0x11, 0x0A, 0x35, 0x45, 0xCE,
			0x49, 0xB8, 0x4B, 0xBB, 0x7B, 0x35, 0x23, 0x61, 0x08, 0xFA, 0x6E };
	uint8_t key128[AES_KEY_SIZE_128] = { 0x95, 0xA8, 0xEE, 0x8E, 0x89, 0x97,
			0x9B, 0x9E, 0xFD, 0xCB, 0xC6, 0xEB, 0x97, 0x97, 0x52, 0x8D };
	uint8_t key192[AES_KEY_SIZE_192] = { 0x95, 0xA8, 0xEE, 0x8E, 0x89, 0x97,
			0x9B, 0x9E, 0xFD, 0xCB, 0xC6, 0xEB, 0x97, 0x97, 0x52, 0x8D, 0x43,
			0x2D, 0xC2, 0x60, 0x61, 0x55, 0x38, 0x18 };
	uint8_t key256[AES_KEY_SIZE_256] = { 0x95, 0xA8, 0xEE, 0x8E, 0x89, 0x97,
			0x9B, 0x9E, 0xFD, 0xCB, 0xC6, 0xEB, 0x97, 0x97, 0x52, 0x8D, 0x43,
			0x2D, 0xC2, 0x60, 0x61, 0x55, 0x38, 0x18, 0xEA, 0x63, 0x5E, 0xC5,
			0xD5, 0xA7, 0x72, 0x7E };

	/* Test Vector CBC */
	uint8_t key128_cbc[AES_KEY_SIZE_128] = {0x2b,0x7e,0x15,0x16,0x28,0xae,0xd2,0xa6,0xab,0xf7,0x15,0x88,0x09,0xcf,0x4f,0x3c};
	uint8_t key192_cbc[AES_KEY_SIZE_192] = {0x8e,0x73,0xb0,0xf7,0xda,0x0e,0x64,0x52,0xc8,0x10,0xf3,0x2b,0x80,0x90,0x79,0xe5,
						0x62,0xf8,0xea,0xd2,0x52,0x2c,0x6b,0x7b};
	uint8_t key256_cbc[AES_KEY_SIZE_256] = {0x60,0x3d,0xeb,0x10,0x15,0xca,0x71,0xbe,0x2b,0x73,0xae,0xf0,0x85,0x7d,0x77,0x81,
						0x1f,0x35,0x2c,0x07,0x3b,0x61,0x08,0xd7,0x2d,0x98,0x10,0xa3,0x09,0x14,0xdf,0xf4};
	uint8_t iv_cbc[AES_BLOCK_SIZE]       = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f};
	uint8_t plaintext_cbc[AES_BLOCK_SIZE]= {0x6b,0xc1,0xbe,0xe2,0x2e,0x40,0x9f,0x96,0xe9,0x3d,0x7e,0x11,0x73,0x93,0x17,0x2a};
	uint8_t ciphertextcbc128[AES_BLOCK_SIZE]= {0x76,0x49,0xab,0xac,0x81,0x19,0xb2,0x46,0xce,0xe9,0x8e,0x9b,0x12,0xe9,0x19,0x7d};
	uint8_t ciphertextcbc192[AES_BLOCK_SIZE]= {0x4f,0x02,0x1d,0xb2,0x43,0xbc,0x63,0x3d,0x71,0x78,0x18,0x3a,0x9f,0xa0,0x71,0xe8};
	uint8_t ciphertextcbc256[AES_BLOCK_SIZE]= {0xf5,0x8c,0x4c,0x04,0xd6,0xe5,0xf1,0xba,0x77,0x9e,0xab,0xfb,0x5f,0x7b,0xfb,0xd6};

	/* needed variables */
	int i, nm;
	uint8_t *out, *iv_cbc_tmp;
	AES_KEY *ak;
	cudaError_t cudaerrno;

	iv_cbc_tmp=(uint8_t *) malloc(AES_BLOCK_SIZE * sizeof(uint8_t));
	out= (uint8_t *) malloc(AES_BLOCK_SIZE * sizeof(uint8_t));
	ak = (AES_KEY *) malloc(sizeof(AES_KEY));

	/* Verify the presence of a CUDA supported device */
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	cudaerrno = cudaGetLastError();
	if (cudaSuccess != cudaerrno) {
		printf("Cuda error in file '%s' in line %i : %s.\n",__FILE__,__LINE__, cudaGetErrorString(cudaerrno));
		exit(EXIT_FAILURE);
	}
	if (deviceCount == 0) {
		printf("\nError: no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	} else {
		printf("\nSuccessfully found a device supporting CUDA.\n");
	}

	/* Show the test vector to the user*/
	printf("\nAES test vector:\n");
	printf("input: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", input[i]);
	printf("\nkey (128bit): ");
	for (i = 0; i < AES_KEY_SIZE_128; i++)
		printf("%x", key128[i]);
	printf("\nkey (192bit): ");
	for (i = 0; i < AES_KEY_SIZE_192; i++)
		printf("%x", key192[i]);
	printf("\nkey (256bit): ");
	for (i = 0; i < AES_KEY_SIZE_256; i++)
		printf("%x", key256[i]);
	printf("\nEncrypted output for a 128bit key: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", enc_output128[i]);
	printf("\nEncrypted output for a 192bit key: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", enc_output192[i]);
	printf("\nEncrypted output for a 256bit key: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", enc_output256[i]);
	printf("\nDecrypted output for a 128bit key: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", dec_output128[i]);
	printf("\nDecrypted output for a 192bit key: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", dec_output192[i]);
	printf("\nDecrypted output for a 256bit key: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", dec_output256[i]);

	printf("\n\nAES CBC test vector:");
	printf("\nCBC - IV:         ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", iv_cbc[i]);
	printf("\nCBC - key 128bit: ");
	for (i = 0; i < AES_KEY_SIZE_128; i++)
		printf("%x", key128_cbc[i]);
	printf("\nCBC - key 192bit: ");
	for (i = 0; i < AES_KEY_SIZE_192; i++)
		printf("%x", key192_cbc[i]);
	printf("\nCBC - key 256bit: ");
	for (i = 0; i < AES_KEY_SIZE_256; i++)
		printf("%x", key256_cbc[i]);
	printf("\nCBC - Plaintext:  ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", plaintext_cbc[i]);

	printf("\nCBC - Ciphertext for a 128bit key: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", ciphertextcbc128[i]);
	printf("\nCBC - Ciphertext for a 192bit key: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", ciphertextcbc192[i]);
	printf("\nCBC - Ciphertext for a 256bit key: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", ciphertextcbc256[i]);

	printf("\n\n");

	/* Initializing the CUDA encrypt library */
	AES_cuda_init(&nm,0,OUTPUT_NORMAL);

	/* Encrypt with a 128bit key */
	printf("Press any key to start encrypting with a 128bit key...\n");
	AES_set_encrypt_key((unsigned char*) key128, 128, ak);
	/* GPU work */
	AES_cuda_transfer_key(ak);
	AES_cuda_encrypt((unsigned char*) input, out, AES_BLOCK_SIZE);
	printf("output GPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");
	/* CPU work */
	AES_encrypt((unsigned char*) input, out, ak);
	printf("output CPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");

	/* Encrypt with a 192bit key */
	printf("\nPress any key to start encrypting with a 192bit key...\n");
	AES_set_encrypt_key((unsigned char*) key192, 192, ak);
	/* GPU work */
	AES_cuda_transfer_key(ak);
	AES_cuda_encrypt((unsigned char*) input, out, AES_BLOCK_SIZE);
	printf("output GPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");
	/* CPU work */
	AES_encrypt((unsigned char*) input, out, ak);
	printf("output CPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");

	/* Encrypt with a 256bit key */
	printf("\nPress any key to start encrypting with a 256bit key...\n");
	AES_set_encrypt_key((unsigned char*) key256, 256, ak);
	/* GPU work */
	AES_cuda_transfer_key(ak);
	AES_cuda_encrypt((unsigned char*) input, out, AES_BLOCK_SIZE);
	printf("output GPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");
	/* CPU work */
	AES_encrypt((unsigned char*) input, out, ak);
	printf("output CPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);

	/* Decrypt with a 128bit key */
	printf("\n\nPress any key to start decrypting with a 128bit key...\n");
	AES_set_decrypt_key((unsigned char*) key128, 128, ak);
	/* GPU work */
	AES_cuda_transfer_key(ak);
	AES_cuda_decrypt((unsigned char*) input, out, AES_BLOCK_SIZE);
	printf("output GPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");
	/* CPU work */
	AES_decrypt((unsigned char*) input, out, ak);
	printf("output CPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");

	/* Decrypt with a 192bit key */
	printf("\nPress any key to start decrypting with a 192bit key...\n");
	AES_set_decrypt_key((unsigned char*) key192, 192, ak);
	/* GPU work */
	AES_cuda_transfer_key(ak);
	AES_cuda_decrypt((unsigned char*) input, out, AES_BLOCK_SIZE);
	printf("output GPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");
	/* CPU work */
	AES_decrypt((unsigned char*) input, out, ak);
	printf("output CPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");

	/* Decrypt with a 256bit key */
	printf("\nPress any key to start decrypting with a 256bit key...\n");
	AES_set_decrypt_key((unsigned char*) key256, 256, ak);
	/* GPU work */
	AES_cuda_transfer_key(ak);
	AES_cuda_decrypt((unsigned char*) input, out, AES_BLOCK_SIZE);
	printf("output GPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");
	/* CPU work */
	AES_decrypt((unsigned char*) input, out, ak);
	printf("output CPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");
//
#ifndef CBC_ENC_CPU
	/* Encrypt with a 128bit key CBC */
	printf("\nPress any key to start encrypting with a 128bit key CBC...\n");
	AES_set_encrypt_key((unsigned char*) key128_cbc, 128, ak);
	/* GPU work*/
	AES_cuda_transfer_key(ak);
	AES_cuda_transfer_iv(iv_cbc);
	AES_cuda_encrypt_cbc(plaintext_cbc,out,AES_BLOCK_SIZE);
	printf("output GPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");
	/* CPU work */
	memcpy(iv_cbc_tmp,iv_cbc,AES_BLOCK_SIZE*sizeof(uint8_t)); 
	AES_cbc_encrypt(plaintext_cbc,out,AES_BLOCK_SIZE,ak,iv_cbc_tmp,AES_ENCRYPT);
	printf("output CPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");
#endif
//
	/* Decrypt with a 128bit key CBC */
	printf("\nPress any key to start decrypting with a 128bit key CBC...\n");
	AES_set_decrypt_key((unsigned char*) key128_cbc, 128, ak);
	/* GPU work*/
	AES_cuda_transfer_key(ak);
	AES_cuda_transfer_iv(iv_cbc);
	AES_cuda_decrypt_cbc(ciphertextcbc128,out,AES_BLOCK_SIZE);
	printf("output GPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");
	/* CPU work */
	memcpy(iv_cbc_tmp,iv_cbc,AES_BLOCK_SIZE*sizeof(uint8_t)); 
	AES_cbc_encrypt(ciphertextcbc128,out,AES_BLOCK_SIZE,ak,iv_cbc_tmp,AES_DECRYPT);
	printf("output CPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");
//
#ifndef CBC_ENC_CPU
	/* Encrypt with a 192bit key CBC */
	printf("\nPress any key to start encrypting with a 192bit key CBC...\n");
	AES_set_encrypt_key((unsigned char*) key192_cbc, 192, ak);
	/* GPU work*/
	AES_cuda_transfer_key(ak);
	AES_cuda_transfer_iv(iv_cbc);
	AES_cuda_encrypt_cbc(plaintext_cbc,out,AES_BLOCK_SIZE);
	printf("output GPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");
	/* CPU work */
	memcpy(iv_cbc_tmp,iv_cbc,AES_BLOCK_SIZE*sizeof(uint8_t)); 
	AES_cbc_encrypt(plaintext_cbc,out,AES_BLOCK_SIZE,ak,iv_cbc_tmp,AES_ENCRYPT);
	printf("output CPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");
#endif
//
	/* Decrypt with a 192bit key CBC */
	printf("\nPress any key to start decrypting with a 192bit key CBC...\n");
	AES_set_decrypt_key((unsigned char*) key192_cbc, 192, ak);
	/* GPU work*/
	AES_cuda_transfer_key(ak);
	AES_cuda_transfer_iv(iv_cbc);
	AES_cuda_decrypt_cbc(ciphertextcbc192,out,AES_BLOCK_SIZE);
	printf("output GPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");
	/* CPU work */
	memcpy(iv_cbc_tmp,iv_cbc,AES_BLOCK_SIZE*sizeof(uint8_t)); 
	AES_cbc_encrypt(ciphertextcbc192,out,AES_BLOCK_SIZE,ak,iv_cbc_tmp,AES_DECRYPT);
	printf("output CPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");
//
#ifndef CBC_ENC_CPU
	/* Encrypt with a 256bit key CBC */
	printf("\nPress any key to start encrypting with a 256bit key CBC...\n");
	AES_set_encrypt_key((unsigned char*) key256_cbc, 256, ak);
	/* GPU work*/
	AES_cuda_transfer_key(ak);
	AES_cuda_transfer_iv(iv_cbc);
	AES_cuda_encrypt_cbc(plaintext_cbc,out,AES_BLOCK_SIZE);
	printf("output GPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");
	/* CPU work */
	memcpy(iv_cbc_tmp,iv_cbc,AES_BLOCK_SIZE*sizeof(uint8_t)); 
	AES_cbc_encrypt(plaintext_cbc,out,AES_BLOCK_SIZE,ak,iv_cbc_tmp,AES_ENCRYPT);
	printf("output CPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");
#endif
//
	/* Decrypt with a 256bit key CBC */
	printf("\nPress any key to start decrypting with a 256bit key CBC...\n");
	AES_set_decrypt_key((unsigned char*) key256_cbc, 256, ak);
	/* GPU work*/
	AES_cuda_transfer_key(ak);
	AES_cuda_transfer_iv(iv_cbc);
	AES_cuda_decrypt_cbc(ciphertextcbc256,out,AES_BLOCK_SIZE);
	printf("output GPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");
	/* CPU work */
	memcpy(iv_cbc_tmp,iv_cbc,AES_BLOCK_SIZE*sizeof(uint8_t)); 
	AES_cbc_encrypt(ciphertextcbc256,out,AES_BLOCK_SIZE,ak,iv_cbc_tmp,AES_DECRYPT);
	printf("output CPU: ");
	for (i = 0; i < AES_BLOCK_SIZE; i++)
		printf("%x", out[i]);
	printf("\n");
//
	/* freeing memory */
	AES_cuda_finish();
	/* happy ending */
	exit(EXIT_SUCCESS);
}
