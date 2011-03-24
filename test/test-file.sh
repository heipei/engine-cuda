#!/usr/bin/env zsh

for alg in {aes-128-ecb,aes-192-ecb,aes-256-ecb,bf-ecb,camellia-128-ecb,cast5-ecb,des-ecb,idea-ecb}; do 
	for file in sample.in.*; do 
		./test.sh $alg $file FFFF auto 2>/dev/null|grep secs; 
	done|sort -n -k 3|grep i;
done
