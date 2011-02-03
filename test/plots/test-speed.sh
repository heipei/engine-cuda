#!/usr/bin/env zsh
OPENSSL=openssl
RUN=17
AVG_RUNS=1
CIPHERS=(bf-ecb camellia-128-ecb cast5-ecb des-ecb idea-ecb)

if [[ -n $3 ]]; then
	CIPHERS=$3
fi
echo "Ciphers: $3\n==========="

for cipher in $CIPHERS; do
	if [[ -n $2 ]]; then
		echo "$2\n==========";
	fi

	echo "Execution of $cipher"
	if [[ -n $1 ]]; then
		echo "Averaging over $1 runs"
		AVG_RUNS=$1
	fi
	echo "===================="

	if [[ $2 != "CPUONLY" ]]; then
		echo -n "" > ${cipher}_gpu.dat > ${cipher}_gpu_average.dat
		for avg_run in $(seq 1 $AVG_RUNS); do
			$OPENSSL speed -engine cudamrg -evp $cipher -mr | egrep -e '+H:' -e '+F:' | sed s/+H:// | sed s/+F:22:$cipher:// > out-filtered-$cipher.txt
			for i in $(seq 1 $RUN); do
				row=$(cat out-filtered-$cipher.txt |cut -f $i -d :|tr '\n' ' ')
				echo $row >> ${cipher}_gpu.dat
			done
		done
		if [[ $AVG_RUNS > 1 ]]; then
			for run in $(seq 1 $RUN); do
				sed -n "${run}~17p" ${cipher}_gpu.dat|awk '{sum+=$2} END { OFMT = "%.0f"; print $1, "\t", sum/NR}' >> ${cipher}_gpu_average.dat
			done
			sort -n ${cipher}_gpu_average.dat|sed '/^$/d' > ${cipher}_gpu.dat
		fi
		rm out-filtered-$cipher.txt ${cipher}_gpu_average.dat
	fi

	
	if [[ $2 != "GPUONLY" ]]; then
		echo -n "" > ${cipher}_cpu.dat
		$OPENSSL speed -evp $cipher -mr | egrep -e '+H:' -e '+F:' | sed s/+H:// | sed s/+F:22:$cipher:// > out-filtered-$cipher.txt
		for i in $(seq 1 $RUN); do
			row=$(cat out-filtered-${cipher}.txt |cut -f $i -d :|tr '\n' ' ')
			echo $row >> ${cipher}_cpu.dat
		done
		rm out-filtered-$cipher.txt
	fi
done

./ecb-encrypt.plt > ecb-encrypt.png
