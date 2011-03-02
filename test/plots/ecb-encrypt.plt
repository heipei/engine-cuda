#!/usr/bin/gnuplot -persist
#
#    
#    	G N U P L O T
#    	Version 4.2 patchlevel 6 
#    	last modified Sep 2009
#    	System: Linux 2.6.32-23-generic
#    
#    	Copyright (C) 1986 - 1993, 1998, 2004, 2007 - 2009
#    	Thomas Williams, Colin Kelley and many others
#    
#    	Type `help` to access the on-line reference manual.
#    	The gnuplot FAQ is available from http://www.gnuplot.info/faq/
#    
#    	Send bug reports and suggestions to <http://sourceforge.net/projects/gnuplot>
#    
# set terminal png nocrop font /usr/share/fonts/truetype/ttf-liberation/LiberationSans-Regular.ttf 12 size 200,100 
# set output
unset clip points
set clip one
unset clip two
set bar 1.000000
set border 31 front linetype -1 linewidth 1.000
set xdata
set ydata
set zdata
set x2data
set y2data
set boxwidth
set style fill  empty border
set style rectangle back fc lt -3 fillstyle  solid 1.00 border -1
set dummy x,y
set format x "% g"
set format y "%4.0f"
set format x2 "% g"
set format y2 "% g"
set format z "% g"
set format cb "% g"
set angles radians
set grid nopolar
set grid xtics nomxtics ytics nomytics noztics nomztics \
 nox2tics nomx2tics noy2tics nomy2tics nocbtics nomcbtics
set grid layerdefault   linetype 0 linewidth 1.000,  linetype 0 linewidth 1.000
set key title ""
set key inside left top vertical Right noreverse enhanced autotitles box linetype 1 linewidth 1.000
set key noinvert samplen 4 spacing 1 width 0 height 0 
unset label
unset arrow
set style increment default
unset style line
unset style arrow
set style histogram clustered gap 2 title  offset character 0, 0, 0
unset logscale
set offsets 0, 0, 0, 0
set pointsize 1
set encoding default
unset polar
unset parametric
unset decimalsign
set view 60, 30, 1, 1  
set samples 100, 100
set isosamples 10, 10
set surface
unset contour
set clabel '%8.3g'
set mapping cartesian
set datafile separator whitespace
unset hidden3d
set cntrparam order 4
set cntrparam linear
set cntrparam levels auto 5
set cntrparam points 5
set size ratio 0 1,1
set origin 0,0
set style data points
set style function lines
set xzeroaxis linetype -2 linewidth 1.000
set yzeroaxis linetype -2 linewidth 1.000
set zzeroaxis linetype -2 linewidth 1.000
set x2zeroaxis linetype -2 linewidth 1.000
set y2zeroaxis linetype -2 linewidth 1.000
set ticslevel 0.5
set mxtics default
set mytics default
set mztics default
set mx2tics default
set my2tics default
set mcbtics default
set xtics border in scale 1,0.5 mirror norotate  offset character 0, 0, 0
set xtics
set xtics   ("16B" 0.00000, "64B" 1.00000, "256B" 2.00000, "1KB" 3.00000, "2KB" 4.00000, "4KB" 5.00000, "8KB" 6.00000, "16KB" 7.00000, "32KB" 8.00000, "64KB" 9.00000, "128KB" 10.0000, "256KB" 11.0000, "512KB" 12.0000, "1MB" 13.0000, "2MB" 14.0000, "4MB" 15.0000, "8MB" 16.0000)
set ytics border in scale 1,0.5 mirror norotate  offset character 0, 0, 0
set ytics autofreq
set ztics border in scale 1,0.5 nomirror norotate  offset character 0, 0, 0
set ztics autofreq
set nox2tics
set noy2tics
set cbtics border in scale 1,0.5 mirror norotate  offset character 0, 0, 0
set cbtics autofreq
set title  offset character 0, 0, 0 font "" norotate
set timestamp bottom 
set timestamp "" 
set timestamp  offset character 0, 0, 0 font "" norotate
set rrange [ * : * ] noreverse nowriteback  # (currently [0.00000:10.0000] )
set trange [ * : * ] noreverse nowriteback  # (currently [-5.00000:5.00000] )
set urange [ * : * ] noreverse nowriteback  # (currently [-5.00000:5.00000] )
set vrange [ * : * ] noreverse nowriteback  # (currently [-5.00000:5.00000] )
set xlabel "Encryption batch block size [bytes]" 
set xlabel  offset character 0, 0, 0 font "" textcolor lt -1 norotate
set x2label "" 
set x2label  offset character 0, 0, 0 font "" textcolor lt -1 norotate
set xrange [ * : * ] noreverse nowriteback  # (currently [-10.0000:10.0000] )
set x2range [ * : * ] noreverse nowriteback  # (currently [-10.0000:10.0000] )
set ylabel "Encryption speed [megabytes/seconds]" 
set ylabel  offset character 0, 0, 0 font "" textcolor lt -1 rotate by 90
set y2label "" 
set y2label  offset character 0, 0, 0 font "" textcolor lt -1 rotate by 90
set yrange [ * : * ] noreverse nowriteback  # (currently [-10.0000:10.0000] )
set y2range [ * : * ] noreverse nowriteback  # (currently [-10.0000:10.0000] )
set zlabel "" 
set zlabel  offset character 0, 0, 0 font "" textcolor lt -1 norotate
set zrange [ * : * ] noreverse nowriteback  # (currently [-10.0000:10.0000] )
set cblabel "" 
set cblabel  offset character 0, 0, 0 font "" textcolor lt -1 rotate by 90
set cbrange [ * : * ] noreverse nowriteback  # (currently [-10.0000:10.0000] )
set zero 1e-08
set lmargin  -1
set bmargin  -1
set rmargin  -1
set tmargin  -1
set locale "C"
set pm3d explicit at s
set pm3d scansautomatic
set pm3d interpolate 1,1 flush begin noftriangles nohidden3d corners2color mean
set palette positive nops_allcF maxcolors 0 gamma 1.5 color model RGB 
set palette rgbformulae 7, 5, 15
set colorbox default
set colorbox vertical origin screen 0.9, 0.2, 0 size screen 0.05, 0.6, 0 front bdefault
set loadpath 
set fontpath 
set fit noerrorvariables
set terminal pdf size 15cm,9cm

#set terminal png size 1000,500

set title "Encryption performance with ECB (CUDA)" 
set output "ecb-encrypt_cuda.pdf"

GNUTERM = "wxt"
plot 'bf-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'Blowfish ECB GPU' with linespoints, \
     'bf-ecb_cpu.dat' using ($2/1048576) title 'Blowfish ECB CPU' with linespoints, \
     'idea-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'IDEA ECB GPU' with linespoints, \
     'idea-ecb_cpu.dat' using ($2/1048576) title 'IDEA ECB CPU' with linespoints, \
     'des-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'DES ECB GPU' with linespoints, \
     'des-ecb_cpu.dat' using ($2/1048576) title 'DES ECB CPU' with linespoints, \
     'cast5-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'CAST5 ECB GPU' with linespoints, \
     'cast5-ecb_cpu.dat' using ($2/1048576) title 'CAST5 ECB CPU' with linespoints, \
     'camellia-128-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'Camellia 128 ECB GPU' with linespoints, \
     'camellia-128-ecb_cpu.dat' using ($2/1048576) title 'Camellia 128 ECB CPU' with linespoints

set title "Encryption performance with ECB (OpenCL)" 
set output "ecb-encrypt_opencl.pdf"

plot 'bf-ecb_gpu_opencl.dat' using ($2/1048576) title 'Blowfish GPU' with linespoints, \
     'bf-ecb_cpu.dat' using ($2/1048576) title 'Blowfish CPU' with linespoints, \
     'idea-ecb_gpu_opencl.dat' using ($2/1048576) title 'IDEA GPU' with linespoints, \
     'idea-ecb_cpu.dat' using ($2/1048576) title 'IDEA CPU' with linespoints, \
     'des-ecb_gpu_opencl.dat' using ($2/1048576) title 'DES GPU' with linespoints, \
     'des-ecb_cpu.dat' using ($2/1048576) title 'DES CPU' with linespoints, \
     'cast5-ecb_gpu_opencl.dat' using ($2/1048576) title 'CAST5 GPU' with linespoints, \
     'cast5-ecb_cpu.dat' using ($2/1048576) title 'CAST5 CPU' with linespoints, \
     'camellia-128-ecb_gpu_opencl.dat' using ($2/1048576) title 'Camellia 128 GPU' with linespoints, \
     'camellia-128-ecb_cpu.dat' using ($2/1048576) title 'Camellia 128 CPU' with linespoints

set title "Encryption performance Blowfish-ECB" 
set output "ecb-encrypt_bf.pdf"
plot 'bf-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'Blowfish CUDA' with linespoints, \
     'bf-ecb_gpu_opencl.dat' using ($2/1048576) title 'Blowfish OpenCL' with linespoints, \
     'bf-ecb_cpu.dat' using ($2/1048576) title 'Blowfish CPU' with linespoints

set title "Encryption performance CAST5-ECB" 
set output "ecb-encrypt_cast5.pdf"
plot 'cast5-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'CAST5 CUDA' with linespoints, \
     'cast5-ecb_gpu_opencl.dat' using ($2/1048576) title 'CAST5 OpenCL' with linespoints, \
     'cast5-ecb_cpu.dat' using ($2/1048576) title 'CAST5 CPU' with linespoints

set title "Encryption performance DES-ECB" 
set output "ecb-encrypt_des.pdf"
plot 'des-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'DES CUDA' with linespoints, \
     'des-ecb_gpu_opencl.dat' using ($2/1048576) title 'DES OpenCL' with linespoints, \
     'des-ecb_cpu.dat' using ($2/1048576) title 'DES CPU' with linespoints

set title "Encryption performance IDEA-ECB" 
set output "ecb-encrypt_idea.pdf"
plot 'idea-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'IDEA CUDA' with linespoints, \
     'idea-ecb_gpu_opencl.dat' using ($2/1048576) title 'IDEA OpenCL' with linespoints, \
     'idea-ecb_cpu.dat' using ($2/1048576) title 'IDEA CPU' with linespoints

set title "Encryption performance Camellia-ECB" 
set output "ecb-encrypt_camellia-128.pdf"
plot 'camellia-128-ecb_gpu_cudamrg.dat' using ($2/1048576) title 'Camellia CUDA' with linespoints, \
     'camellia-128-ecb_gpu_opencl.dat' using ($2/1048576) title 'Camellia OpenCL' with linespoints, \
     'camellia-128-ecb_cpu.dat' using ($2/1048576) title 'Camellia CPU' with linespoints
