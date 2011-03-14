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
#ifndef CBC_ENC_CPU
void AES_cuda_encrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);
#endif
void AES_cuda_decrypt_cbc(const unsigned char *in, unsigned char *out,size_t nbytes);

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
