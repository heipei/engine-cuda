#!/usr/bin/env zsh
make -j5 -C ..

if [[ ! -e sample.in ]]; then;
	echo "Creating 100MB sample.in file..."
	dd bs=1048576 count=100 if=/dev/urandom of=sample.in
fi

echo "== DES tests =="
echo "CUDA encryption"
echo "---------------"
time openssl enc -engine cudamrg -e -des-ecb -nosalt -v -in sample.in -out des.cuda -bufsize 33554432 -K "010101F10101F1F1"
echo -e "\nCPU encryption"
echo "--------------"
time openssl enc -e -des-ecb -nosalt -v -in sample.in -out des.openssl -K "010101F10101F1F1"

echo -e "\nMD5:"
md5sum des.cuda des.openssl
rm -rf des.cuda des.openssl
