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
infile_e='./trailer.mp4'
infile_d='./trailer.aes'
outfile='/dev/null'
key='c26d8562740cdcea548efc08babd19a3d1aaedf6'
cmd='/opt/bin/openssl'

for cipher in aes-128-ecb aes-192-ecb aes-256-ecb aes-128-cbc aes-192-cbc aes-256-cbc
do
$cmd enc -engine cudamrg -e -$cipher -v -in $infile_e -out $outfile -bufsize    4096 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher.dat #   4K
$cmd enc -engine cudamrg -e -$cipher -v -in $infile_e -out $outfile -bufsize    8192 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher.dat #   8K
$cmd enc -engine cudamrg -e -$cipher -v -in $infile_e -out $outfile -bufsize   16384 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher.dat #  16K
$cmd enc -engine cudamrg -e -$cipher -v -in $infile_e -out $outfile -bufsize   32768 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher.dat #  32K
$cmd enc -engine cudamrg -e -$cipher -v -in $infile_e -out $outfile -bufsize   65536 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher.dat #  64K
$cmd enc -engine cudamrg -e -$cipher -v -in $infile_e -out $outfile -bufsize  131072 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher.dat # 128K
$cmd enc -engine cudamrg -e -$cipher -v -in $infile_e -out $outfile -bufsize  262144 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher.dat # 256K
$cmd enc -engine cudamrg -e -$cipher -v -in $infile_e -out $outfile -bufsize  524288 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher.dat # 512K
$cmd enc -engine cudamrg -e -$cipher -v -in $infile_e -out $outfile -bufsize 1048576 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher.dat #   1M
$cmd enc -engine cudamrg -e -$cipher -v -in $infile_e -out $outfile -bufsize 2097151 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher.dat #   2M
$cmd enc -engine cudamrg -e -$cipher -v -in $infile_e -out $outfile -bufsize 4194304 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher.dat #   4M
$cmd enc -engine cudamrg -e -$cipher -v -in $infile_e -out $outfile -bufsize 8388608 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher.dat #   8M
done

for cipher in aes-128-ecb aes-192-ecb aes-256-ecb aes-128-cbc aes-192-cbc aes-256-cbc
do
$cmd enc -e -$cipher -v -in $infile_e -out $infile_e.aes -k $key
$cmd enc -engine cudamrg -d -$cipher -v -in $infile_e.aes -out $outfile -bufsize    4096 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher-decrypt.dat #   4K
$cmd enc -engine cudamrg -d -$cipher -v -in $infile_e.aes -out $outfile -bufsize    8192 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher-decrypt.dat #   8K
$cmd enc -engine cudamrg -d -$cipher -v -in $infile_e.aes -out $outfile -bufsize   16384 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher-decrypt.dat #  16K
$cmd enc -engine cudamrg -d -$cipher -v -in $infile_e.aes -out $outfile -bufsize   32768 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher-decrypt.dat #  32K
$cmd enc -engine cudamrg -d -$cipher -v -in $infile_e.aes -out $outfile -bufsize   65536 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher-decrypt.dat #  64K
$cmd enc -engine cudamrg -d -$cipher -v -in $infile_e.aes -out $outfile -bufsize  131072 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher-decrypt.dat # 128K
$cmd enc -engine cudamrg -d -$cipher -v -in $infile_e.aes -out $outfile -bufsize  262144 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher-decrypt.dat # 256K
$cmd enc -engine cudamrg -d -$cipher -v -in $infile_e.aes -out $outfile -bufsize  524288 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher-decrypt.dat # 512K
$cmd enc -engine cudamrg -d -$cipher -v -in $infile_e.aes -out $outfile -bufsize 1048576 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher-decrypt.dat #   1M
$cmd enc -engine cudamrg -d -$cipher -v -in $infile_e.aes -out $outfile -bufsize 2097151 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher-decrypt.dat #   2M
$cmd enc -engine cudamrg -d -$cipher -v -in $infile_e.aes -out $outfile -bufsize 4194304 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher-decrypt.dat #   4M
$cmd enc -engine cudamrg -d -$cipher -v -in $infile_e.aes -out $outfile -bufsize 8388608 -k $key | grep "Total time: " | cut -f 2 -d ':' | cut -f 2 -d ' ' >>  $cipher-decrypt.dat #   8M
rm $infile_e.aes
done
