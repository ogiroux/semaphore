#/usr/local/bin/clang++ -L/usr/loca/cuda/lib64 -lcudart_static -ldl -lrt -pthread -arch=sm_70 lib/cuda/volta.cubin -lib -o volta.a
/usr/local/bin/clang++ -L/usr/loca/cuda/lib64 -lcudart_static -ldl -lrt -pthread -Iinclude -std=c++11 -O3 --cuda-gpu-arch=sm_70 lib/cuda/semaphore.cu lib/cuda/atomic.cu test.cu volta.a -o test
