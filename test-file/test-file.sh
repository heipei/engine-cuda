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
INPUT_FILE='AirForceOne.avi'
OUTPUT_FILE='AirForceOne.avi'
INPUT_DIR=$HOME/Scrivania
OUTPUT_DIR='./test-file-'$(date +%d-%m-%Y_%R)
IV='42ec8f3a1806b0b7a35821d3a2481291c5bb41bcc2127f00c2f3bd73b15ac94c'
KEY='8ec598172dcad6bcc07e61d54751ee2f10773bf301907603bec44e8374f17836'
BUFSIZE=8192
#
# INIT
#
mkdir -p $OUTPUT_DIR
#
# Encrypt with the GPU
#
$OPENSSL enc -in $INPUT_DIR/$INPUT_FILE -out $OUTPUT_DIR/crypt-gpu-aes-128-cbc-$OUTPUT_FILE -k $KEY -aes-128-cbc -e -engine cudamrg -bufsize $BUFSIZE
$OPENSSL enc -in $INPUT_DIR/$INPUT_FILE -out $OUTPUT_DIR/crypt-gpu-aes-128-ecb-$OUTPUT_FILE -k $KEY -aes-128-ecb -e -engine cudamrg -bufsize $BUFSIZE
$OPENSSL enc -in $INPUT_DIR/$INPUT_FILE -out $OUTPUT_DIR/crypt-gpu-aes-192-cbc-$OUTPUT_FILE -k $KEY -aes-192-cbc -e -engine cudamrg -bufsize $BUFSIZE
$OPENSSL enc -in $INPUT_DIR/$INPUT_FILE -out $OUTPUT_DIR/crypt-gpu-aes-192-ecb-$OUTPUT_FILE -k $KEY -aes-192-ecb -e -engine cudamrg -bufsize $BUFSIZE
$OPENSSL enc -in $INPUT_DIR/$INPUT_FILE -out $OUTPUT_DIR/crypt-gpu-aes-256-cbc-$OUTPUT_FILE -k $KEY -aes-256-cbc -e -engine cudamrg -bufsize $BUFSIZE
$OPENSSL enc -in $INPUT_DIR/$INPUT_FILE -out $OUTPUT_DIR/crypt-gpu-aes-256-ecb-$OUTPUT_FILE -k $KEY -aes-256-ecb -e -engine cudamrg -bufsize $BUFSIZE
#
# Encrypt with the CPU
#
$OPENSSL enc -in $INPUT_DIR/$INPUT_FILE -out $OUTPUT_DIR/crypt-cpu-aes-128-cbc-$OUTPUT_FILE -k $KEY -aes-128-cbc -e
$OPENSSL enc -in $INPUT_DIR/$INPUT_FILE -out $OUTPUT_DIR/crypt-cpu-aes-128-ecb-$OUTPUT_FILE -k $KEY -aes-128-ecb -e
$OPENSSL enc -in $INPUT_DIR/$INPUT_FILE -out $OUTPUT_DIR/crypt-cpu-aes-192-cbc-$OUTPUT_FILE -k $KEY -aes-192-cbc -e
$OPENSSL enc -in $INPUT_DIR/$INPUT_FILE -out $OUTPUT_DIR/crypt-cpu-aes-192-ecb-$OUTPUT_FILE -k $KEY -aes-192-ecb -e
$OPENSSL enc -in $INPUT_DIR/$INPUT_FILE -out $OUTPUT_DIR/crypt-cpu-aes-256-cbc-$OUTPUT_FILE -k $KEY -aes-256-cbc -e
$OPENSSL enc -in $INPUT_DIR/$INPUT_FILE -out $OUTPUT_DIR/crypt-cpu-aes-256-ecb-$OUTPUT_FILE -k $KEY -aes-256-ecb -e
#
# Dencrypt file encrypted with the GPU with the GPU
#
$OPENSSL enc -in $OUTPUT_DIR/crypt-gpu-aes-128-cbc-$OUTPUT_FILE -out $OUTPUT_DIR/plain-gpu2gpu-aes-128-cbc-$OUTPUT_FILE -k $KEY -aes-128-cbc -d -engine cudamrg -bufsize $BUFSIZE
$OPENSSL enc -in $OUTPUT_DIR/crypt-gpu-aes-128-ecb-$OUTPUT_FILE -out $OUTPUT_DIR/plain-gpu2gpu-aes-128-ecb-$OUTPUT_FILE -k $KEY -aes-128-ecb -d -engine cudamrg -bufsize $BUFSIZE
$OPENSSL enc -in $OUTPUT_DIR/crypt-gpu-aes-192-cbc-$OUTPUT_FILE -out $OUTPUT_DIR/plain-gpu2gpu-aes-192-cbc-$OUTPUT_FILE -k $KEY -aes-192-cbc -d -engine cudamrg -bufsize $BUFSIZE
$OPENSSL enc -in $OUTPUT_DIR/crypt-gpu-aes-192-ecb-$OUTPUT_FILE -out $OUTPUT_DIR/plain-gpu2gpu-aes-192-ecb-$OUTPUT_FILE -k $KEY -aes-192-ecb -d -engine cudamrg -bufsize $BUFSIZE
$OPENSSL enc -in $OUTPUT_DIR/crypt-gpu-aes-256-cbc-$OUTPUT_FILE -out $OUTPUT_DIR/plain-gpu2gpu-aes-256-cbc-$OUTPUT_FILE -k $KEY -aes-256-cbc -d -engine cudamrg -bufsize $BUFSIZE
$OPENSSL enc -in $OUTPUT_DIR/crypt-gpu-aes-256-ecb-$OUTPUT_FILE -out $OUTPUT_DIR/plain-gpu2gpu-aes-256-ecb-$OUTPUT_FILE -k $KEY -aes-256-ecb -d -engine cudamrg -bufsize $BUFSIZE
#
# Dencrypt file encrypted with the GPU with the CPU
#
$OPENSSL enc -in $OUTPUT_DIR/crypt-gpu-aes-128-cbc-$OUTPUT_FILE -out $OUTPUT_DIR/plain-gpu2cpu-aes-128-cbc-$OUTPUT_FILE -k $KEY -aes-128-cbc -d
$OPENSSL enc -in $OUTPUT_DIR/crypt-gpu-aes-128-ecb-$OUTPUT_FILE -out $OUTPUT_DIR/plain-gpu2cpu-aes-128-ecb-$OUTPUT_FILE -k $KEY -aes-128-ecb -d
$OPENSSL enc -in $OUTPUT_DIR/crypt-gpu-aes-192-cbc-$OUTPUT_FILE -out $OUTPUT_DIR/plain-gpu2cpu-aes-192-cbc-$OUTPUT_FILE -k $KEY -aes-192-cbc -d
$OPENSSL enc -in $OUTPUT_DIR/crypt-gpu-aes-192-ecb-$OUTPUT_FILE -out $OUTPUT_DIR/plain-gpu2cpu-aes-192-ecb-$OUTPUT_FILE -k $KEY -aes-192-ecb -d
$OPENSSL enc -in $OUTPUT_DIR/crypt-gpu-aes-256-cbc-$OUTPUT_FILE -out $OUTPUT_DIR/plain-gpu2cpu-aes-256-cbc-$OUTPUT_FILE -k $KEY -aes-256-cbc -d
$OPENSSL enc -in $OUTPUT_DIR/crypt-gpu-aes-256-ecb-$OUTPUT_FILE -out $OUTPUT_DIR/plain-gpu2cpu-aes-256-ecb-$OUTPUT_FILE -k $KEY -aes-256-ecb -d
#
# Dencrypt file encrypted with the CPU with the GPU
#
$OPENSSL enc -in $OUTPUT_DIR/crypt-cpu-aes-128-cbc-$OUTPUT_FILE -out $OUTPUT_DIR/plain-cpu2gpu-aes-128-cbc-$OUTPUT_FILE -k $KEY -aes-128-cbc -d -engine cudamrg -bufsize $BUFSIZE
$OPENSSL enc -in $OUTPUT_DIR/crypt-cpu-aes-128-ecb-$OUTPUT_FILE -out $OUTPUT_DIR/plain-cpu2gpu-aes-128-ecb-$OUTPUT_FILE -k $KEY -aes-128-ecb -d -engine cudamrg -bufsize $BUFSIZE
$OPENSSL enc -in $OUTPUT_DIR/crypt-cpu-aes-192-cbc-$OUTPUT_FILE -out $OUTPUT_DIR/plain-cpu2gpu-aes-192-cbc-$OUTPUT_FILE -k $KEY -aes-192-cbc -d -engine cudamrg -bufsize $BUFSIZE
$OPENSSL enc -in $OUTPUT_DIR/crypt-cpu-aes-192-ecb-$OUTPUT_FILE -out $OUTPUT_DIR/plain-cpu2gpu-aes-192-ecb-$OUTPUT_FILE -k $KEY -aes-192-ecb -d -engine cudamrg -bufsize $BUFSIZE
$OPENSSL enc -in $OUTPUT_DIR/crypt-cpu-aes-256-cbc-$OUTPUT_FILE -out $OUTPUT_DIR/plain-cpu2gpu-aes-256-cbc-$OUTPUT_FILE -k $KEY -aes-256-cbc -d -engine cudamrg -bufsize $BUFSIZE
$OPENSSL enc -in $OUTPUT_DIR/crypt-cpu-aes-256-ecb-$OUTPUT_FILE -out $OUTPUT_DIR/plain-cpu2gpu-aes-256-ecb-$OUTPUT_FILE -k $KEY -aes-256-ecb -d -engine cudamrg -bufsize $BUFSIZE
#
# Dencrypt file encrypted with the CPU with the CPU
#
$OPENSSL enc -in $OUTPUT_DIR/crypt-cpu-aes-128-cbc-$OUTPUT_FILE -out $OUTPUT_DIR/plain-cpu2cpu-aes-128-cbc-$OUTPUT_FILE -k $KEY -aes-128-cbc -d
$OPENSSL enc -in $OUTPUT_DIR/crypt-cpu-aes-128-ecb-$OUTPUT_FILE -out $OUTPUT_DIR/plain-cpu2cpu-aes-128-ecb-$OUTPUT_FILE -k $KEY -aes-128-ecb -d
$OPENSSL enc -in $OUTPUT_DIR/crypt-cpu-aes-192-cbc-$OUTPUT_FILE -out $OUTPUT_DIR/plain-cpu2cpu-aes-192-cbc-$OUTPUT_FILE -k $KEY -aes-192-cbc -d
$OPENSSL enc -in $OUTPUT_DIR/crypt-cpu-aes-192-ecb-$OUTPUT_FILE -out $OUTPUT_DIR/plain-cpu2cpu-aes-192-ecb-$OUTPUT_FILE -k $KEY -aes-192-ecb -d
$OPENSSL enc -in $OUTPUT_DIR/crypt-cpu-aes-256-cbc-$OUTPUT_FILE -out $OUTPUT_DIR/plain-cpu2cpu-aes-256-cbc-$OUTPUT_FILE -k $KEY -aes-256-cbc -d
$OPENSSL enc -in $OUTPUT_DIR/crypt-cpu-aes-256-ecb-$OUTPUT_FILE -out $OUTPUT_DIR/plain-cpu2cpu-aes-256-ecb-$OUTPUT_FILE -k $KEY -aes-256-ecb -d
#
# Verify decrypted file
#
md5sum $OUTPUT_DIR/plain-*-$OUTPUT_FILE
md5sum $INPUT_DIR/$INPUT_FILE
ls -l $OUTPUT_DIR/plain-*-$OUTPUT_FILE
ls -l $INPUT_DIR/$INPUT_FILE
#
# Verify encrypted file
#
md5sum $OUTPUT_DIR/crypt-*-$OUTPUT_FILE
ls -l $OUTPUT_DIR/crypt-*-$OUTPUT_FILE
#
# END OF FILE
