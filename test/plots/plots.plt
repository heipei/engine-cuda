#!/usr/bin/gnuplot -persist

load "engine_cuda.plt"

unset title

set output "ecb-encrypt_aes.pdf"
plot 'plot-data/aes-128-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'AES-128 CUDA' with linespoints lw 2 pt 2, \
     'plot-data/aes-192-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'AES-192 CUDA' with linespoints lw 2 pt 2, \
     'plot-data/aes-256-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'AES-256 CUDA' with linespoints lw 2 pt 2, \
     'plot-data/aes-128-ecb_gpu_opencl.dat' using ($2/1048576) title 'AES-128 OpenCL' with linespoints lw 2 pt 6, \
     'plot-data/aes-192-ecb_gpu_opencl.dat' using ($2/1048576) title 'AES-192 OpenCL' with linespoints lw 2 pt 6, \
     'plot-data/aes-256-ecb_gpu_opencl.dat' using ($2/1048576) title 'AES-256 OpenCL' with linespoints lw 2 pt 6, \
     'plot-data/aes-128-ecb_cpu.dat' using ($2/1048576) title 'AES-128 CPU' with linespoints lw 2 pt 9, \
     'plot-data/aes-192-ecb_cpu.dat' using ($2/1048576) title 'AES-192 CPU' with linespoints lw 2 pt 9, \
     'plot-data/aes-256-ecb_cpu.dat' using ($2/1048576) title 'AES-256 CPU' with linespoints lw 2 pt 9

set output "ecb-encrypt_bf.pdf"
plot 'plot-data/bf-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'Blowfish CUDA' with linespoints lw 2 pt 9, \
     'plot-data/bf-ecb_gpu_opencl.dat' using ($2/1048576) title 'Blowfish OpenCL' with linespoints lw 2 pt 11, \
     'plot-data/bf-ecb_cpu.dat' using ($2/1048576) title 'Blowfish CPU' with linespoints lw 2 pt 7

set key inside left top vertical Right autotitles width 3 box linetype 1 linewidth 1.000
set output "ecb-encrypt_cast5.pdf"
plot 'plot-data/cast5-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'CAST5 CUDA' with linespoints lw 2 pt 9, \
     'plot-data/cast5-ecb_gpu_opencl.dat' using ($2/1048576) title 'CAST5 OpenCL' with linespoints lw 2 pt 11, \
     'plot-data/cast5-ecb_cpu.dat' using ($2/1048576) title 'CAST5 CPU' with linespoints lw 2 pt 7

set output "ecb-encrypt_des.pdf"
plot 'plot-data/des-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'DES CUDA' with linespoints lw 2 pt 9, \
     'plot-data/des-ecb_gpu_opencl.dat' using ($2/1048576) title 'DES OpenCL' with linespoints lw 2 pt 11, \
     'plot-data/des-ecb_cpu.dat' using ($2/1048576) title 'DES CPU' with linespoints lw 2 pt 7

set output "ecb-encrypt_idea.pdf"
plot 'plot-data/idea-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'IDEA CUDA' with linespoints lw 2 pt 9, \
     'plot-data/idea-ecb_gpu_opencl.dat' using ($2/1048576) title 'IDEA OpenCL' with linespoints lw 2 pt 11, \
     'plot-data/idea-ecb_cpu.dat' using ($2/1048576) title 'IDEA CPU' with linespoints lw 2 pt 7

set output "ecb-encrypt_camellia-128.pdf"
plot 'plot-data/camellia-128-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'Camellia CUDA' with linespoints lw 2 pt 9, \
     'plot-data/camellia-128-ecb_gpu_opencl.dat' using ($2/1048576) title 'Camellia OpenCL' with linespoints lw 2 pt 11, \
     'plot-data/camellia-128-ecb_cpu.dat' using ($2/1048576) title 'Camellia CPU' with linespoints lw 2 pt 7

set terminal pdf size 15cm,11cm font "Palatino"
set output "ecb-encrypt_cuda.pdf"
plot 'plot-data/aes-128-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'AES-128 GPU' with linespoints lw 2 pt 11 lc rgb "red", \
     'plot-data/aes-128-ecb_cpu.dat' using ($2/1048576) title 'AES-128 CPU' with linespoints lw 2 pt 11 lc rgb "#800000", \
     'plot-data/bf-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'Blowfish GPU' with linespoints lw 2 pt 2 lc rgb "green", \
     'plot-data/bf-ecb_cpu.dat' using ($2/1048576) title 'Blowfish CPU' with linespoints lw 2 pt 2 lc rgb "#006400", \
     'plot-data/idea-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'IDEA GPU' with linespoints lw 2 pt 3 lc rgb "orange", \
     'plot-data/idea-ecb_cpu.dat' using ($2/1048576) title 'IDEA CPU' with linespoints lw 2 pt 3 lc rgb "#995400", \
     'plot-data/des-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'DES GPU' with linespoints lw 2 pt 13 lc rgb "blue", \
     'plot-data/des-ecb_cpu.dat' using ($2/1048576) title 'DES CPU' with linespoints lw 2 pt 13 lc rgb "#00008B", \
     'plot-data/cast5-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'CAST5 GPU' with linespoints lw 2 pt 9 lc rgb "#778899", \
     'plot-data/cast5-ecb_cpu.dat' using ($2/1048576) title 'CAST5 CPU' with linespoints lw 2 pt 9 lc rgb "#696969", \
     'plot-data/camellia-128-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'Camellia 128 GPU' with linespoints lw 2 pt 6 lc rgb "#C71585", \
     'plot-data/camellia-128-ecb_cpu.dat' using ($2/1048576) title 'Camellia 128 CPU' with linespoints lw 2 pt 6 lc rgb "purple"

set terminal pdf size 15cm,11cm font "Palatino"
set output "ecb-encrypt_opencl.pdf"
plot 'plot-data/aes-128-ecb_gpu_opencl.dat' using ($2/1048576) title 'AES-128 GPU' with linespoints lw 2 pt 11 lc rgb "red", \
     'plot-data/aes-128-ecb_cpu.dat' using ($2/1048576) title 'AES-128 CPU' with linespoints lw 2 pt 11 lc rgb "#800000", \
     'plot-data/bf-ecb_gpu_opencl.dat' using ($2/1048576) title 'Blowfish GPU' with linespoints lw 2 pt 2 lc rgb "green", \
     'plot-data/bf-ecb_cpu.dat' using ($2/1048576) title 'Blowfish CPU' with linespoints lw 2 pt 2 lc rgb "#006400", \
     'plot-data/idea-ecb_gpu_opencl.dat' using ($2/1048576) title 'IDEA GPU' with linespoints lw 2 pt 3 lc rgb "orange", \
     'plot-data/idea-ecb_cpu.dat' using ($2/1048576) title 'IDEA CPU' with linespoints lw 2 pt 3 lc rgb "#995400", \
     'plot-data/des-ecb_gpu_opencl.dat' using ($2/1048576) title 'DES GPU' with linespoints lw 2 pt 13 lc rgb "blue", \
     'plot-data/des-ecb_cpu.dat' using ($2/1048576) title 'DES CPU' with linespoints lw 2 pt 13 lc rgb "#00008B", \
     'plot-data/cast5-ecb_gpu_opencl.dat' using ($2/1048576) title 'CAST5 GPU' with linespoints lw 2 pt 9 lc rgb "#778899", \
     'plot-data/cast5-ecb_cpu.dat' using ($2/1048576) title 'CAST5 CPU' with linespoints lw 2 pt 9 lc rgb "#696969", \
     'plot-data/camellia-128-ecb_gpu_opencl.dat' using ($2/1048576) title 'Camellia 128 GPU' with linespoints lw 2 pt 6 lc rgb "#C71585", \
     'plot-data/camellia-128-ecb_cpu.dat' using ($2/1048576) title 'Camellia 128 CPU' with linespoints lw 2 pt 6 lc rgb "purple"

set terminal pdf size 15cm,8cm font "Palatino"
set output "aes_fine_coarse.pdf"
plot 'plot-data/aes-128-coarse-rambo-cudamrg.dat' using ($2/1048576) title 'AES-128 (CUDA) coarse' with linespoints lw 2 pt 9 lc rgb "red", \
     'plot-data/aes-128-fine-rambo-cudamrg.dat' using ($2/1048576) title 'AES-128 (CUDA) fine' with linespoints lw 2 pt 11 lc rgb "#800000", \
     'plot-data/aes-128-coarse-rambo-opencl.dat' using ($2/1048576) title 'AES-128 (OpenCL) coarse' with linespoints lw 2 pt 5 lc rgb "blue", \
     'plot-data/aes-128-fine-rambo-opencl.dat' using ($2/1048576) title 'AES-128 (OpenCL) fine' with linespoints lw 2 pt 4 lc rgb "#00008B"

set terminal pdf size 15cm,8cm font "Palatino"
set output "idea_mul_umul.pdf"
plot 'plot-data/idea-ecb_32bit_mul.dat' using ($2/1048576) title 'IDEA: a*b' with linespoints lw 2 pt 9, \
     'plot-data/idea-ecb_32bit_umul.dat' using ($2/1048576) title 'IDEA: __umul24(a,b)' with linespoints lw 2 pt 11, \
     'plot-data/idea-ecb_cpu.dat' using ($2/1048576) title 'IDEA CPU' with linespoints lw 2 pt 7

set output "idea_schedule.pdf"
set terminal pdf size 15cm,8cm font "Palatino"
plot 'plot-data/idea-ecb_yield.dat' using ($2/1048576) title 'IDEA: yield' with linespoints lw 2 pt 9, \
     'plot-data/idea-ecb_spin.dat' using ($2/1048576) title 'IDEA: spin' with linespoints lw 2 pt 11, \
     'plot-data/idea-ecb_cpu.dat' using ($2/1048576) title 'IDEA CPU' with linespoints lw 2 pt 7 lc rgb "orange"

set output "idea_multi_block.pdf"
set key inside right center vertical Right noreverse enhanced autotitles box linetype 1 linewidth 1.000
plot 'plot-data/idea-ecb_single_block.dat' using ($2/1048576) title 'IDEA (regular)' with linespoints lw 2, \
     'plot-data/idea-ecb_two_blocks.dat' using ($2/1048576) title 'IDEA (two blocks)' with linespoints lw 2 pt 10, \
     'plot-data/idea-ecb_two_blocks_maxreg.dat' using ($2/1048576) title 'IDEA (two blocks, maxreg=10)' with linespoints lw 2 pt 8, \
     'plot-data/idea-ecb_four_blocks.dat' using ($2/1048576) title 'IDEA (four blocks)' with linespoints lw 2,\
     'plot-data/idea-ecb_four_blocks_maxreg.dat' using ($2/1048576) title 'IDEA (four blocks, maxreg=10)' with linespoints lw 2 pt 9 lc rgb "gray", \
     'plot-data/idea-ecb_cpu.dat' using ($2/1048576) title 'IDEA CPU' with linespoints lw 2 pt 11 lc rgb "orange" 

set output "cuda_blocks.pdf"
set terminal pdf size 15cm,8cm font "Palatino"
set key inside left top vertical Right noreverse enhanced autotitles box linetype 1 linewidth 1.000
unset xtics
set yrange[0:*]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
unset xlabel
plot 'plot-data/speed_blocks.dat' using ($2/1048576):xtic(1) title '128 threads', '' using ($3/1048576):xtic(1) title '256 threads'

set output "fast_mem_limits.pdf"
set terminal pdf size 15cm,8cm font "Palatino"
set key inside right bottom horizontal Right noreverse enhanced autotitles box linetype 1 linewidth 1.000
set yrange[0:60000]
set xrange[0:192]
set xlabel "Independent keys per symmetric multiprocessor (SM)"
set ylabel "Fast memory consumption per SM (KB)" 
set xtics 32
#set ytics 4096
set ytics   ("0KB" 0, "8KB" 8192, "16KB" 16384, "24KB" 24576, "32KB" 32768, "40KB" 40960, "48KB" 49152)
set style data points
set style function lines
set samples 16
set label 1 "Shared Memory\n(CC 2.0)" at 24, 55000 font "Palatino,7" front nopoint tc def
set label 2 "Shared Memory\n(CC 1.x)" at 18, 22500 font "Palatino,7" front nopoint tc def
plot 16384 notitle lw 5 lc rgb "black", \
	49152 notitle lw 5 lc rgb "black", \
	4096*x title "Blowfish" with linespoints lw 2 lc rgb "#800000", \
	4096 + 280*x title "Camellia" with linespoints lw 2 lc rgb "#696969" pt 9, \
	240*x+1024 title "AES" with linespoints lw 2 lc rgb "#00008B", \
	2048+128*x title "DES" with linespoints lw 2 lc rgb "#C71585", \
	216*x title "IDEA" with linespoints lw 2 lc rgb "#FF8C00" pointsize 0.7 pt 5, \
	4096 + 132*x title "CAST 5" with linespoints lw 2 lc rgb "#006400" pt 11


