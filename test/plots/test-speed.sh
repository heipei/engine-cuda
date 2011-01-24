#!/usr/bin/env zsh
OPENSSL=openssl
RUN=17


for cipher in {bf-ecb,camellia-128-ecb,cast5-ecb,des-ecb,idea-ecb}; do
#for cipher in {bf-ecb,des-ecb}; do
	echo "Execution of $cipher"
	echo -n "" > ${cipher}_gpu.dat > ${cipher}_cpu.dat

	$OPENSSL speed -engine cudamrg -evp $cipher -mr | egrep -e '+H:' -e '+F:' | sed s/+H:// | sed s/+F:22:$cipher:// > out-filtered-$cipher.txt
	for i in $(seq 1 $RUN); do
		row=$(cat out-filtered-$cipher.txt |cut -f $i -d :|tr '\n' ' ')
		echo $row >> ${cipher}_gpu.dat
	done
	rm out-filtered-$cipher.txt

	
	$OPENSSL speed -evp $cipher -mr | egrep -e '+H:' -e '+F:' | sed s/+H:// | sed s/+F:22:$cipher:// > out-filtered-$cipher.txt
	for i in $(seq 1 $RUN); do
		row=$(cat out-filtered-${cipher}.txt |cut -f $i -d :|tr '\n' ' ')
		echo $row >> ${cipher}_cpu.dat
	done
	rm out-filtered-$cipher.txt
done

./ecb-encrypt.plt > ecb-encrypt.png
