/usr/local/bin/clang++ -L/usr/local/cuda/lib64 -lcudart -ldl -lrt -pthread -Iinclude -std=c++11 -O2 --cuda-gpu-arch=sm_70 clang.cu -o test
