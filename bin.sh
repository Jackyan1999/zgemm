nvcc -o zgemm zgemm_5.cu  -arch=sm_70 -lcublas
cuobjdump  -sass -arch sm_70 zgemm  >zgemm.sass
