nvcc -o zgemm zgemm_6.cu -lcublas -O3 
# ./zgemm 8 8 8 
#  ./zgemm 16 16 16
#  ./zgemm 64 64 64
echo -e " 128: \n"
./zgemm 128 128 128
echo -e " 256: \n"
./zgemm 256 256 256
echo -e " 512: \n"
./zgemm 512 512 512
echo -e " 1024: \n"
 ./zgemm 1024 1024 1024
 echo -e " 2048: \n"
 ./zgemm 2048 2048 2048
 echo -e " 4096: \n"
 ./zgemm 4096 4096 4096
  echo -e " 8192: \n"
 ./zgemm 8192 8192 8192
#   echo -e " 16384: \n"
#  ./zgemm 16384 16384 16384
# ./zgemm 512 512 512