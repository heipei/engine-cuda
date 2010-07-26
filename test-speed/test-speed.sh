#!/bin/bash
#
# @version 0.1.0 (2010)
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
OPENSSL=/opt/bin/openssl
RUN=17

for cipher in aes-128-ecb aes-192-ecb aes-256-ecb aes-128-cbc aes-192-cbc aes-256-cbc
	do
	$OPENSSL speed -engine cudamrg -evp $cipher -mr | egrep -e '+H:' -e '+F:' | sed s/+H:// | sed s/+F:22:$cipher:// > out-filtered-$cipher.txt
	for i in $(seq 1 $RUN) 
		do
		row=$(cat out-filtered-$cipher.txt |cut -f $i -d :)
		echo $row >> $cipher.dat
		done
	rm out-filtered-$cipher.txt
	done

for cipher in aes-128-ecb aes-192-ecb aes-256-ecb aes-128-cbc aes-192-cbc aes-256-cbc
	do
	$OPENSSL speed -engine cudamrg -decrypt -evp $cipher -mr | egrep -e '+H:' -e '+F:' | sed s/+H:// | sed s/+F:22:$cipher:// > out-filtered-$cipher-decrypt.txt
	for i in $(seq 1 $RUN) 
		do
		row=$(cat out-filtered-$cipher-decrypt.txt |cut -f $i -d :)
		echo $row >> $cipher-decrypt.dat
		done
	rm out-filtered-$cipher-decrypt.txt
	done

for cipher in aes-128-ecb aes-192-ecb aes-256-ecb aes-128-cbc aes-192-cbc aes-256-cbc
	do
	$OPENSSL speed -evp $cipher -mr | egrep -e '+H:' -e '+F:' | sed s/+H:// | sed s/+F:22:$cipher:// > out-filtered-$cipher.txt
	for i in $(seq 1 $RUN) 
		do
		row=$(cat out-filtered-$cipher.txt |cut -f $i -d :)
		echo $row >> $cipher-cpu.dat
		done
	rm out-filtered-$cipher.txt
	done

for cipher in aes-128-ecb aes-192-ecb aes-256-ecb aes-128-cbc aes-192-cbc aes-256-cbc
	do
	$OPENSSL speed -decrypt -evp $cipher -mr | egrep -e '+H:' -e '+F:' | sed s/+H:// | sed s/+F:22:$cipher:// > out-filtered-$cipher-decrypt.txt
	for i in $(seq 1 $RUN) 
		do
		row=$(cat out-filtered-$cipher-decrypt.txt |cut -f $i -d :)
		echo $row >> $cipher-decrypt-cpu.dat
		done
	rm out-filtered-$cipher-decrypt.txt
	done

./aes-ecb-encrypt.plt > aes-ecb-encrypt.png
./aes-cbc-encrypt.plt > aes-cbc-encrypt.png
./aes-cbc-decrypt.plt > aes-cbc-decrypt.png
./aes-ecb-decrypt.plt > aes-ecb-decrypt.png
