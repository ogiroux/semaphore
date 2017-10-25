/*

Copyright (c) 2017, NVIDIA Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#warning "Use of all warp threads is TOO iffy without CUDA support for compute_70 or above. Will run 1 thread per warp."
#endif

#include <sys/mman.h>
#include <cuda/launch>
#include <cuda/atomic>
#include <cuda/semaphore>
#include <cassert>
#include <mutex>
#include <thread>
#include <iostream>

#define __test_abi __host__ __device__

template<class T>
using atomic = cuda::experimental::atomic<T>;
using thread = std::thread;
using binary_semaphore = cuda::experimental::binary_semaphore;
using counting_semaphore = cuda::experimental::counting_semaphore;
using condition_variable_atomic = cuda::experimental::condition_variable_atomic;
namespace details = cuda::experimental::details;

#include "test.hpp"

using mutex = binary_semaphore_mutex;

template<class F>
__global__ void run_gpu_thread(uint32_t count, uint32_t count_per_block, F const* f) {

    unsigned int const myIdx = blockIdx.x * count_per_block + threadIdx.x;
    if (myIdx < count)
        (*f)(myIdx);
}

uint32_t cap = 0;
bool use_malloc_managed = true;
uint32_t max_block_count = 0;

void* allocate_raw_bytes(size_t s, bool force = false) { 
    void* ptr = nullptr;
#ifdef __CUDACC__
    if(use_malloc_managed || force) {
        auto const ret = cap < 6 ? cudaHostAlloc(&ptr, s, 0) : cudaMallocManaged(&ptr, s);
        assert(ret == cudaSuccess);
        if(cap >= 6)
            cudaMemAdvise(ptr, s, cudaMemAdviseSetPreferredLocation, 0);
    }
    else
#endif
    {
        ptr = mmap(0, s, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if(ptr == MAP_FAILED) {
            ptr = nullptr;
        }
    }
    assert(ptr != nullptr);
    return ptr;
}

void deallocate_raw_bytes(void* ptr, bool force = false) {
#ifdef __CUDACC__
    if(use_malloc_managed || force)
        cudaFree(ptr);
    else
#endif
    {
        // Leak. Oops.
    }
}

template<class F>
F* start_gpu_threads(uint32_t count, F f) {

    if(!count)
        return nullptr;

    uint32_t const blocks = (std::min)(count, max_block_count);
    uint32_t const threads_per_block = (count / blocks) + (count % blocks ? 1 : 0);

    auto const fptr = new (allocate_raw_bytes(sizeof(F))) F(f);
    assert(uintptr_t(fptr) % alignof(F) == 0);
    run_gpu_thread<F><<<blocks, threads_per_block>>>(count, threads_per_block, fptr);

    return fptr;
}

template<class F>
void stop_gpu_threads(F* fptr) {
    if(nullptr == fptr)
        return;
    auto const ret = cudaDeviceSynchronize();
    assert(ret == cudaSuccess);
    fptr->~F();
    deallocate_raw_bytes(fptr);
}

uint32_t dev = 0;

unsigned int max_gpu_threads() { 

    cudaSetDevice(dev);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    max_block_count = deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor / 1024;
    cap = deviceProp.major;
    if (cap < 7)
        return max_block_count * 32;
    else
        return max_block_count * 32 * 32;
}

#include "driver.cpp"

int main(int argc, char const* argv[]) {

    return driver_main(argc, argv);
}
