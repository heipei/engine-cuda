#!/bin/sh
make -j5

echo "CUDA encryption:"
openssl enc -engine cudamrg -e -des-ecb -nosalt -v -in o.in -out o.des -bufsize 8388608 -K "010101F10101F1F1"
echo -e "\nCPU encryption:"
openssl enc -e -des-ecb -nosalt -v -in o.in -out o.ref -K "010101F10101F1F1"
#echo -e "\nCUDA:"
#xxd o.des|head
#echo -e "\nCPU:"
#xxd o.ref|head
echo -e "\nMD5SUM:"
md5sum o.des o.ref
rm -rf o.ref o.des
