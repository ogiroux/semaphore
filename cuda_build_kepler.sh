nvcc --expt-relaxed-constexpr --relocatable-device-code=true --expt-extended-lambda -Iinclude -std=c++11 -O3 -arch=compute_30 lib/cuda/semaphore.cu lib/cuda/atomic.cu test.cu -o test
