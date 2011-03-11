#!/usr/bin/gnuplot -persist

load "engine_cuda.plt"

#set terminal png size 1000,500

set title "IDEA ECB (CUDA)" 
set output "idea_mul_umul.pdf"

plot 'idea-ecb_32bit_mul.dat' using ($2/1048576) title 'IDEA: a*b' with linespoints lw 2, \
     'idea-ecb_32bit_umul.dat' using ($2/1048576) title 'IDEA: __umul24(a,b)' with linespoints lw 2

