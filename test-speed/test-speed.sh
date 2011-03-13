#!/bin/bash
#
# @version 0.1.2 (2011)
# @author Paolo Margara <paolo.margara@gmail.com>
# 
# Copyright 2010 Paolo Margara
#
# This file is part of Engine_cudamrg.
#
# Engine_cudamrg is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License or
# any later version.
# 
# Engine_cudamrg is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Engine_cudamrg.  If not, see <http://www.gnu.org/licenses/>.
#
OPENSSL=openssl
RUN=17
OPT=''
# OPT='-elapsed' ## measure time in real time instead of CPU user time.
OUTPUT_DIR='./test-speed-'$(date +%d-%m-%Y_%R)

mkdir -p $OUTPUT_DIR
mkdir -p data

for cipher in aes-128-ecb aes-192-ecb aes-256-ecb aes-128-cbc aes-192-cbc aes-256-cbc
	do
	$OPENSSL speed $OPT -engine cudamrg -evp $cipher -mr | egrep -e '+H:' -e '+F:' | sed s/+H:// | sed s/+F:22:$cipher:// > ./data/out-filtered-$cipher.txt
	for i in $(seq 1 $RUN) 
		do
		row=$(cat ./data/out-filtered-$cipher.txt |cut -f $i -d :)
		echo $row >> ./data/$cipher.dat
		done
	rm ./data/out-filtered-$cipher.txt
	done

for cipher in aes-128-ecb aes-192-ecb aes-256-ecb aes-128-cbc aes-192-cbc aes-256-cbc
	do
	$OPENSSL speed $OPT -engine cudamrg -decrypt -evp $cipher -mr | egrep -e '+H:' -e '+F:' | sed s/+H:// | sed s/+F:22:$cipher:// > ./data/out-filtered-$cipher-decrypt.txt
	for i in $(seq 1 $RUN) 
		do
		row=$(cat ./data/out-filtered-$cipher-decrypt.txt |cut -f $i -d :)
		echo $row >> ./data/$cipher-decrypt.dat
		done
	rm ./data/out-filtered-$cipher-decrypt.txt
	done

for cipher in aes-128-ecb aes-192-ecb aes-256-ecb aes-128-cbc aes-192-cbc aes-256-cbc
	do
	$OPENSSL speed $OPT -evp $cipher -mr | egrep -e '+H:' -e '+F:' | sed s/+H:// | sed s/+F:22:$cipher:// > ./data/out-filtered-$cipher.txt
	for i in $(seq 1 $RUN) 
		do
		row=$(cat ./data/out-filtered-$cipher.txt |cut -f $i -d :)
		echo $row >> ./data/$cipher-cpu.dat
		done
	rm ./data/out-filtered-$cipher.txt
	done

for cipher in aes-128-ecb aes-192-ecb aes-256-ecb aes-128-cbc aes-192-cbc aes-256-cbc
	do
	$OPENSSL speed $OPT -decrypt -evp $cipher -mr | egrep -e '+H:' -e '+F:' | sed s/+H:// | sed s/+F:22:$cipher:// > ./data/out-filtered-$cipher-decrypt.txt
	for i in $(seq 1 $RUN) 
		do
		row=$(cat ./data/out-filtered-$cipher-decrypt.txt |cut -f $i -d :)
		echo $row >> ./data/$cipher-decrypt-cpu.dat
		done
	rm ./data/out-filtered-$cipher-decrypt.txt
	done

./aes-ecb-encrypt.plt > aes-ecb-encrypt.png
./aes-cbc-encrypt.plt > aes-cbc-encrypt.png
./aes-cbc-decrypt.plt > aes-cbc-decrypt.png
./aes-ecb-decrypt.plt > aes-ecb-decrypt.png
#
# MAKE TABLE SPEED - START
#
WIKI_PATH='http://engine-cuda.googlecode.com/svn/wiki'
#HTML_PATH='http://engine-cuda.googlecode.com/svn/wiki'
HTML_PATH='.'
#
rm table.html  2> /dev/null
rm table.wiki  2> /dev/null
rm table-aes-ecb-enc.csv 2> /dev/null
rm table-aes-ecb-dec.csv 2> /dev/null
rm table-aes-cbc-enc.csv 2> /dev/null
rm table-aes-cbc-dec.csv 2> /dev/null
#
echo "<html><head><title>Engine_cudamrg Benchmark suite - Test Speed</title></head><body>" > table.html
echo "<h1>AES ECB encryption performance</h1>" >> table.html
echo "=AES ECB encryption performance=" > table.wiki
echo ' ' >> table.wiki
echo '<img src="'$HTML_PATH'/aes-ecb-encrypt.png" alt="aes-ecb-encrypt"/>' >> table.html
echo '['$WIKI_PATH'/aes-ecb-encrypt.png]' >> table.wiki
echo ' ' >> table.wiki
echo '<table><tr><th>blocksize</th><th>aes-128-ecb-gpu</th><th>aes-128-ecb-cpu</th><th>aes-192-ecb-gpu</th><th>aes-192-ecb-cpu</th><th>aes-256-ecb-gpu</th><th>aes-256-ecb-cpu</th></tr>' >> table.html
echo '|| *blocksize* || *aes-128-ecb-gpu* || *aes-128-ecb-cpu* || *aes-192-ecb-gpu* || *aes-192-ecb-cpu* || *aes-256-ecb-gpu* || *aes-256-ecb-cpu* ||' >> table.wiki
echo 'blocksize;aes-128-ecb-gpu;aes-128-ecb-cpu;aes-192-ecb-gpu;aes-192-ecb-cpu;aes-256-ecb-gpu;aes-256-ecb-cpu' >> table-aes-ecb-enc.csv
for blocksize in 16 64 256 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608
	do
	speedGPU_128_ecb=$(cat ./data/aes-128-ecb.dat    |grep ^$blocksize |cut -f 2 -d ' '|head -1)
	speedCPU_128_ecb=$(cat ./data/aes-128-ecb-cpu.dat|grep ^$blocksize |cut -f 2 -d ' '|head -1)
	speedGPU_192_ecb=$(cat ./data/aes-192-ecb.dat    |grep ^$blocksize |cut -f 2 -d ' '|head -1)
	speedCPU_192_ecb=$(cat ./data/aes-192-ecb-cpu.dat|grep ^$blocksize |cut -f 2 -d ' '|head -1)
	speedGPU_256_ecb=$(cat ./data/aes-256-ecb.dat    |grep ^$blocksize |cut -f 2 -d ' '|head -1)
	speedCPU_256_ecb=$(cat ./data/aes-256-ecb-cpu.dat|grep ^$blocksize |cut -f 2 -d ' '|head -1)
	echo '<tr><th>'$blocksize'</th><td>'$speedGPU_128_ecb'</td><td>'$speedCPU_128_ecb'</td><td>'$speedGPU_192_ecb'</td><td>'$speedCPU_192_ecb'</td><td>'$speedGPU_256_ecb'</td><td>'$speedCPU_256_ecb'</td></tr>' >> table.html
	echo '|| *'$blocksize'* || '$speedGPU_128_ecb' || '$speedCPU_128_ecb' || '$speedGPU_192_ecb' || '$speedCPU_192_ecb' || '$speedGPU_256_ecb' || '$speedCPU_256_ecb' ||' >> table.wiki
	echo $blocksize';'$speedGPU_128_ecb';'$speedCPU_128_ecb';'$speedGPU_192_ecb';'$speedCPU_192_ecb';'$speedGPU_256_ecb';'$speedCPU_256_ecb >> table-aes-ecb-enc.csv
	done
echo '</table>' >> table.html
echo '<p>The "numbers" in the column "blocksize" are in bytes.</p>' >> table.html
echo '<p>All other "numbers" are in bytes per second processed.</p>' >> table.html
echo ' ' >> table.wiki
echo 'The "numbers" in the column "blocksize" are in bytes.' >> table.wiki
echo ' ' >> table.wiki
echo 'All other "numbers" are in bytes per second processed.' >> table.wiki
echo ' ' >> table.wiki

echo "<h1>AES ECB decryption performance</h1>" >> table.html
echo "=AES ECB decryption performance=" >> table.wiki
echo ' ' >> table.wiki
echo '<img src="'$HTML_PATH'/aes-ecb-decrypt.png" alt="aes-ecb-decrypt"/>' >> table.html
echo '['$WIKI_PATH'/aes-ecb-decrypt.png]' >> table.wiki
echo ' ' >> table.wiki
echo '<table><tr><th>blocksize</th><th>aes-128-ecb-gpu</th><th>aes-128-ecb-cpu</th><th>aes-192-ecb-gpu</th><th>aes-192-ecb-cpu</th><th>aes-256-ecb-gpu</th><th>aes-256-ecb-cpu</th></tr>'  >> table.html
echo '|| *blocksize* || *aes-128-ecb-gpu* || *aes-128-ecb-cpu* || *aes-192-ecb-gpu* || *aes-192-ecb-cpu* || *aes-256-ecb-gpu* || *aes-256-ecb-cpu* ||' >> table.wiki
echo 'blocksize;aes-128-ecb-gpu;aes-128-ecb-cpu;aes-192-ecb-gpu;aes-192-ecb-cpu;aes-256-ecb-gpu;aes-256-ecb-cpu' >> table-aes-ecb-dec.csv
for blocksize in 16 64 256 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608
	do
	speedGPU_128_ecb=$(cat ./data/aes-128-ecb-decrypt.dat    |grep ^$blocksize |cut -f 2 -d ' '|head -1)
	speedCPU_128_ecb=$(cat ./data/aes-128-ecb-decrypt-cpu.dat|grep ^$blocksize |cut -f 2 -d ' '|head -1)
	speedGPU_192_ecb=$(cat ./data/aes-192-ecb-decrypt.dat    |grep ^$blocksize |cut -f 2 -d ' '|head -1)
	speedCPU_192_ecb=$(cat ./data/aes-192-ecb-decrypt-cpu.dat|grep ^$blocksize |cut -f 2 -d ' '|head -1)
	speedGPU_256_ecb=$(cat ./data/aes-256-ecb-decrypt.dat    |grep ^$blocksize |cut -f 2 -d ' '|head -1)
	speedCPU_256_ecb=$(cat ./data/aes-256-ecb-decrypt-cpu.dat|grep ^$blocksize |cut -f 2 -d ' '|head -1)
	echo '<tr><th>'$blocksize'</th><td>'$speedGPU_128_ecb'</td><td>'$speedCPU_128_ecb'</td><td>'$speedGPU_192_ecb'</td><td>'$speedCPU_192_ecb'</td><td>'$speedGPU_256_ecb'</td><td>'$speedCPU_256_ecb'</td></tr>' >> table.html
	echo '|| *'$blocksize'* || '$speedGPU_128_ecb' || '$speedCPU_128_ecb' || '$speedGPU_192_ecb' || '$speedCPU_192_ecb' || '$speedGPU_256_ecb' || '$speedCPU_256_ecb' ||' >> table.wiki
	echo $blocksize';'$speedGPU_128_ecb';'$speedCPU_128_ecb';'$speedGPU_192_ecb';'$speedCPU_192_ecb';'$speedGPU_256_ecb';'$speedCPU_256_ecb >> table-aes-ecb-dec.csv
	done
echo '</table>' >> table.html
echo '<p>The "numbers" in the column "blocksize" are in bytes.</p>' >> table.html
echo '<p>All other "numbers" are in bytes per second processed.</p>' >> table.html
echo ' ' >> table.wiki
echo 'The "numbers" in the column "blocksize" are in bytes.' >> table.wiki
echo ' ' >> table.wiki
echo 'All other "numbers" are in bytes per second processed.' >> table.wiki
echo ' ' >> table.wiki

echo "<h1>AES CBC encryption performance</h1>" >> table.html
echo "=AES CBC encryption performance=" >> table.wiki
echo ' ' >> table.wiki
echo '<img src="'$HTML_PATH'/aes-cbc-encrypt.png" alt="aes-cbc-encrypt"/>' >> table.html
echo '['$WIKI_PATH'/aes-cbc-encrypt.png]' >> table.wiki
echo ' ' >> table.wiki
echo '<table><tr><th>blocksize</th><th>aes-128-cbc-gpu</th><th>aes-128-cbc-cpu</th><th>aes-192-cbc-gpu</th><th>aes-192-cbc-cpu</th><th>aes-256-cbc-gpu</th><th>aes-256-cbc-cpu</th></tr>' >> table.html
echo '|| *blocksize* || *aes-128-cbc-gpu* || *aes-128-cbc-cpu* || *aes-192-cbc-gpu* || *aes-192-cbc-cpu* || *aes-256-cbc-gpu* || *aes-256-cbc-cpu* ||' >> table.wiki
echo 'blocksize;aes-128-cbc-gpu;aes-128-cbc-cpu;aes-192-cbc-gpu;aes-192-cbc-cpu;aes-256-cbc-gpu;aes-256-cbc-cpu' >> table-aes-cbc-enc.csv
for blocksize in 16 64 256 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608
	do
	speedGPU_128_cbc=$(cat ./data/aes-128-cbc.dat    |grep ^$blocksize |cut -f 2 -d ' '|head -1)
	speedCPU_128_cbc=$(cat ./data/aes-128-cbc-cpu.dat|grep ^$blocksize |cut -f 2 -d ' '|head -1)
	speedGPU_192_cbc=$(cat ./data/aes-192-cbc.dat    |grep ^$blocksize |cut -f 2 -d ' '|head -1)
	speedCPU_192_cbc=$(cat ./data/aes-192-cbc-cpu.dat|grep ^$blocksize |cut -f 2 -d ' '|head -1)
	speedGPU_256_cbc=$(cat ./data/aes-256-cbc.dat    |grep ^$blocksize |cut -f 2 -d ' '|head -1)
	speedCPU_256_cbc=$(cat ./data/aes-256-cbc-cpu.dat|grep ^$blocksize |cut -f 2 -d ' '|head -1)
	echo '<tr><th>'$blocksize'</th><td>'$speedGPU_128_cbc'</td><td>'$speedCPU_128_cbc'</td><td>'$speedGPU_192_cbc'</td><td>'$speedCPU_192_cbc'</td><td>'$speedGPU_256_cbc'</td><td>'$speedCPU_256_cbc'</td></tr>' >> table.html
	echo '|| *'$blocksize'* || '$speedGPU_128_cbc' || '$speedCPU_128_cbc' || '$speedGPU_192_cbc' || '$speedCPU_192_cbc' || '$speedGPU_256_cbc' || '$speedCPU_256_cbc' ||' >> table.wiki
	echo $blocksize';'$speedGPU_128_cbc';'$speedCPU_128_cbc';'$speedGPU_192_cbc';'$speedCPU_192_cbc';'$speedGPU_256_cbc';'$speedCPU_256_cbc >> table-aes-cbc-enc.csv
	done
echo '</table>' >> table.html
echo '<p>The "numbers" in the column "blocksize" are in bytes.</p>' >> table.html
echo '<p>All other "numbers" are in bytes per second processed.</p>' >> table.html
echo ' ' >> table.wiki
echo 'The "numbers" in the column "blocksize" are in bytes.' >> table.wiki
echo ' ' >> table.wiki
echo 'All other "numbers" are in bytes per second processed.' >> table.wiki
echo ' ' >> table.wiki

echo "<h1>AES CBC decryption performance</h1>" >> table.html
echo "=AES CBC decryption performance=" >> table.wiki
echo ' ' >> table.wiki
echo '<img src="'$HTML_PATH'/aes-cbc-decrypt.png" alt="aes-cbc-decrypt"/>' >> table.html
echo '['$WIKI_PATH'/aes-cbc-decrypt.png]' >> table.wiki
echo ' ' >> table.wiki
echo '<table><tr><th>blocksize</th><th>aes-128-cbc-gpu</th><th>aes-128-cbc-cpu</th><th>aes-192-cbc-gpu</th><th>aes-192-cbc-cpu</th><th>aes-256-cbc-gpu</th><th>aes-256-cbc-cpu</th></tr>' >> table.html
echo '|| *blocksize* || *aes-128-cbc-gpu* || *aes-128-cbc-cpu* || *aes-192-cbc-gpu* || *aes-192-cbc-cpu* || *aes-256-cbc-gpu* || *aes-256-cbc-cpu* ||' >> table.wiki
echo 'blocksize;aes-128-cbc-gpu;aes-128-cbc-cpu;aes-192-cbc-gpu;aes-192-cbc-cpu;aes-256-cbc-gpu;aes-256-cbc-cpu' >> table-aes-cbc-dec.csv
for blocksize in 16 64 256 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608
        do
        speedGPU_128_cbc=$(cat ./data/aes-128-cbc-decrypt.dat    |grep ^$blocksize |cut -f 2 -d ' '|head -1)
        speedCPU_128_cbc=$(cat ./data/aes-128-cbc-decrypt-cpu.dat|grep ^$blocksize |cut -f 2 -d ' '|head -1)
        speedGPU_192_cbc=$(cat ./data/aes-192-cbc-decrypt.dat    |grep ^$blocksize |cut -f 2 -d ' '|head -1)
        speedCPU_192_cbc=$(cat ./data/aes-192-cbc-decrypt-cpu.dat|grep ^$blocksize |cut -f 2 -d ' '|head -1)
        speedGPU_256_cbc=$(cat ./data/aes-256-cbc-decrypt.dat    |grep ^$blocksize |cut -f 2 -d ' '|head -1)
        speedCPU_256_cbc=$(cat ./data/aes-256-cbc-decrypt-cpu.dat|grep ^$blocksize |cut -f 2 -d ' '|head -1)
        echo '<tr><th>'$blocksize'</th><td>'$speedGPU_128_cbc'</td><td>'$speedCPU_128_cbc'</td><td>'$speedGPU_192_cbc'</td><td>'$speedCPU_192_cbc'</td><td>'$speedGPU_256_cbc'</td><td>'$speedCPU_256_cbc'</td></tr>' >> table.html
        echo '|| *'$blocksize'* || '$speedGPU_128_cbc' || '$speedCPU_128_cbc' || '$speedGPU_192_cbc' || '$speedCPU_192_cbc' || '$speedGPU_256_cbc' || '$speedCPU_256_cbc' ||' >> table.wiki
        echo $blocksize';'$speedGPU_128_cbc';'$speedCPU_128_cbc';'$speedGPU_192_cbc';'$speedCPU_192_cbc';'$speedGPU_256_cbc';'$speedCPU_256_cbc >> table-aes-cbc-dec.csv
        done
echo '</table>' >> table.html
echo '<p>The "numbers" in the column "blocksize" are in bytes.</p>' >> table.html
echo '<p>All other "numbers" are in bytes per second processed.</p>' >> table.html
echo ' ' >> table.wiki
echo 'The "numbers" in the column "blocksize" are in bytes.' >> table.wiki
echo ' ' >> table.wiki
echo 'All other "numbers" are in bytes per second processed.' >> table.wiki
echo '</body></html>' >> table.html
#
# MAKE TABLE SPEED - END
#
mv *.png $OUTPUT_DIR
mv table* $OUTPUT_DIR
mv data $OUTPUT_DIR
