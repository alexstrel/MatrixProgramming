
NVCPP=nvc++

GCC_HOME=/opt/spack-0.19.0/opt/spack/linux-ubuntu22.04-haswell/gcc-12.1.0/gcc-12.2.0-ogs53w6jn6kghfvjoaeqa3jmblrgrsne

all: dmma_test.exe

dmma_test.exe: dmma_test.cpp mma_m8n8k4_fp64_sm80.h
	${NVCPP} -o dmma_test.exe -O3 -cuda -gpu=cc86 -gpu=managed -std=c++20 -cudalib=cublas,curand --gcc-toolchain=${GCC_HOME} -I./ dmma_test.cpp 

