#!/usr/bin/env zsh

KEY="FFFF"
if [[ -n $2 ]]; then
	KEY=$2
fi

echo "Cipher\tBlocksize\tKernel-CUDA\tKernel-OCL\tBW-CUDA\tBW-OCL"
for cipher in {aes-128-ecb,aes-192-ecb,aes-256-ecb,bf-ecb,camellia-128-ecb,cast5-ecb,des-ecb,idea-ecb}; do 
	for file in sample.in.*; do 
		BUFSIZE=`ls -l $file|awk {'print $5'}`
		openssl enc -engine cudamrg -e -$cipher -nosalt -nopad -v -in $file -out /dev/null -bufsize $BUFSIZE -K $KEY 2>/dev/null|grep secs|grep "CUDA"
		openssl enc -engine opencl -e -$cipher -nosalt -nopad -v -in $file -out /dev/null -bufsize $BUFSIZE -K $KEY 2>/dev/null|grep secs|grep "OpenCL"
	done|sort -n -k 3|sed 'N;s/\n/ /'|awk {'OFS=" &\t "; OFMT="%.2f"; print $1, $3/1024, $5/1000, $13/1000, " & ", $7, $15'};
done

echo ""
date
