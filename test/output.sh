# Table for each algorithm
paste rambo-urandom rambo-zero|grep 8192| awk {'OFS=" &\t "; OFMT="%.2f"; print "\\multirow{2}{*}{"$1"}\t& CUDA\t", $5, $11, $18, $24, $24/$11, "\t\\\\ \\cline{2-7}\n\t\t\t& OpenCL", $7, $13, $20, $26, $26/$13, "\t\\\\ \\hline"'}


# Paste 8600 GT and GTX 295
paste nomaxrregcount rambo-zero-combined|grep AES|awk {'OFS=" &\t "; print $3, $5, $7, $18, $20, $11, $13, $24, $26 "\t\\\\ \\hline"'}

# AES Shared vs const
paste rambo-aes-128-constant-urandom rambo-aes-128-constant-zero|grep AES|awk {'OFS=" &\t ";  OFMT="%.2f"; print "& " $3, $5, $11, $18, $24, $24/$11, "\\\\ \\cline{2-7}"'}
