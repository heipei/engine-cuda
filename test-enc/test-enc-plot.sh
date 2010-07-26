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
./aes-ecb-encrypt.plt > aes-ecb-encrypt.png
./aes-cbc-encrypt.plt > aes-cbc-encrypt.png
./aes-ecb-decrypt.plt > aes-ecb-decrypt.png
./aes-cbc-decrypt.plt > aes-cbc-decrypt.png
