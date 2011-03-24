#!/usr/bin/env zsh


for size in {1,2,8,256,1024,2048,8192}; do
	dd bs=1024 count=$size if=$1 of=sample.in.${size}kb
done

