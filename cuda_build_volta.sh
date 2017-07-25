nvcc --expt-relaxed-constexpr --relocatable-device-code=true --device-c -arch=sm_70 lib/cuda/volta.cubin -lib -o volta.a
nvcc --expt-relaxed-constexpr --relocatable-device-code=true --expt-extended-lambda -Iinclude -std=c++11 -O3 -arch=compute_70 -code=sm_70 lib/cuda/semaphore.cu lib/cuda/atomic.cu test.cu volta.a -o test
