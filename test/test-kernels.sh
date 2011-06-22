#!/usr/bin/env zsh

KEY="FFFF"
DEC_CIPHERS=(aes-128-ecb aes-128-cbc aes-192-ecb aes-192-cbc aes-256-ecb aes-256-cbc bf-ecb bf-cbc camellia-128-ecb camellia-128-cbc cast5-ecb cast5-cbc des-ecb des-cbc idea-ecb idea-cbc)
ENC_CIPHERS=(aes-128-ecb aes-192-ecb aes-256-ecb bf-ecb camellia-128-ecb cast5-ecb des-ecb idea-ecb)

if [[ -n $1 ]]; then
	KEY=$1
fi

if [[ -n $2 ]]; then
	CIPHERS=$2
fi

echo "ENCRYPTION\n============="
echo "Cipher\tBlocksize\tKernel-CUDA\tKernel-OCL\tBW-CUDA\tBW-OCL"
for cipher in $ENC_CIPHERS; do 
	for file in sample.in.*; do 
		BUFSIZE=`ls -l $file|awk {'print $5'}`
		openssl enc -engine cudamrg -e -$cipher -nosalt -iv "FFFF" -nopad -v -in $file -out /dev/null -bufsize $BUFSIZE -K $KEY 2>/dev/null|grep secs|grep "CUDA"
		openssl enc -engine opencl -e -$cipher -nosalt -iv "FFFF" -nopad -v -in $file -out /dev/null -bufsize $BUFSIZE -K $KEY 2>/dev/null|grep secs|grep "OpenCL"
	done|sort -n -k 3|sed 'N;s/\n/ /'|awk {'OFS=" &\t "; OFMT="%.2f"; print $1, $3/1024, $5/1000, $13/1000, " & ", $7, $15'};
done

echo "DECRYPTION\n============="
echo "Cipher\tBlocksize\tKernel-CUDA\tKernel-OCL\tBW-CUDA\tBW-OCL"
for cipher in $DEC_CIPHERS; do 
	for file in sample.in.*; do 
		BUFSIZE=`ls -l $file|awk {'print $5'}`
		openssl enc -engine cudamrg -d -$cipher -nosalt -iv "FFFF" -nopad -v -in $file -out /dev/null -bufsize $BUFSIZE -K $KEY 2>/dev/null|grep secs|grep "CUDA"
		openssl enc -engine opencl -d -$cipher -nosalt -iv "FFFF" -nopad -v -in $file -out /dev/null -bufsize $BUFSIZE -K $KEY 2>/dev/null|grep secs|grep "OpenCL"
	done|sort -n -k 3|sed 'N;s/\n/ /'|awk {'OFS=" &\t "; OFMT="%.2f"; print $1, $3/1024, $5/1000, $13/1000, " & ", $7, $15'};
done

echo ""
date
