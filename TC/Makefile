
NVCPP=nvc++

GCC_HOME=/home/astrel/install/spack-0.19.1/opt/spack/linux-scientific7-haswell/gcc-4.8.5/gcc-12.2.0-alrdcqhufql6zdnkw2ave2fdgpxadhy6

all: dmma_test.exe

dmma_test.exe: dmma_test.cpp mma_m8n8k4_fp64_sm80.h
	${NVCPP} -o dmma_test.exe -O3 -cuda -gpu=cc86 -gpu=managed -std=c++20 -cudalib=cublas,curand --gcc-toolchain=${GCC_HOME} -I./ dmma_test.cpp 

