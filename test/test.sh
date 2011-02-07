#!/usr/bin/env zsh
# vim:ft=sh

SIZE=100
CIPHERS=(bf-ecb camellia-128-ecb cast5-ecb des-ecb idea-ecb)
# TODO: Use getopt or zparseopts

if [[ $ARGC -le 1 ]]; then
	echo " Usage: ./test.sh <algo> <file> <key>"
	echo "        ./test.sh <algo> <file>"
	echo ""
	echo "        <algo> in {bf-ecb,camellia-128-ecb,cast-ecb,des-ecb,idea-ecb,aes-ecb} or"
	echo "        <algo> = all"
	exit 0
fi

KEY="A10101F10101F1F1"
if [[ -n $3 ]]; then
	KEY=$3;
fi

make -s -j5 -C ..

if [[ $1 != "all" ]]; then
	CIPHERS=$1
fi

if [[ $2 == "sample.in" && ! -e sample.in ]]; then;
	echo "Creating a 100 MB sample.in file..."
	dd bs=1048576 count=100 if=/dev/urandom of=sample.in
fi

for cipher in $CIPHERS; do
	echo "==== $cipher tests ===="
	echo ">> CUDA encryption"
	echo "---------------"
	time openssl enc -engine cudamrg -e -$cipher -nosalt -nopad -v -in $2 -out $cipher.out.gpu -bufsize 8388608 -K "$KEY"
	echo -e "\n>> CPU encryption"
	echo "--------------"
	time openssl enc -e -$cipher -nosalt -nopad -v -in $2 -out $cipher.out.cpu -K "$KEY"

	CHKCPU=`cksum $cipher.out.cpu|awk {'print $1'}`
	CHKCUDA=`cksum $cipher.out.gpu|awk {'print $1'}`

	echo ""
	if [[ $CHKCPU != $CHKCUDA ]]; then
		echo ">> CAUTION: cksum mismatch!"
		echo ">> CPU: $CHKCPU; CUDA: $CHKCUDA"
		echo ">> XXD CPU:"
		xxd $cipher.out.cpu|head
		echo ">> XXD GPU:"
		xxd $cipher.out.gpu|head
	else
		echo ">> CKSUM matches"
		rm -rf $cipher.out.gpu $cipher.out.cpu
	fi
done
