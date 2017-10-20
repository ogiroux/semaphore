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

#include <semaphore>
#include <cassert>
#include <mutex>
#include <thread>
#include <iostream>
#include <cstdlib>

#define __test_abi
#define __managed__
template<class T>
using atomic = std::atomic<T>;
using mutex = std::mutex;
using thread = std::thread;
using namespace std::experimental;
template<class F>
int start_gpu_threads(uint32_t count, F f) { assert(!count); return 0; }
void stop_gpu_threads(int) { }
uint32_t max_gpu_threads() { return 0; }
unsigned int dev = 0;
unsigned int cap = 0;

bool use_malloc_managed;

void* allocate_raw_bytes(size_t s) { 
    return malloc(s);
}
void* allocate_bytes(size_t s, size_t a) { 
    a = std::max(a, sizeof(size_t));
    unsigned char* ptr = (unsigned char*)allocate_raw_bytes(a + s + sizeof(size_t));
    unsigned char* target = ptr + sizeof(size_t);
    target += a - uintptr_t(target) % a;
    *(size_t*)(target - sizeof(size_t)) = target - ptr;
    return target;
}
template<class T, class... Args>
T* allocate(Args... args) {
    return new (allocate_bytes(sizeof(T), alignof(T))) T(std::forward<Args>(args)...);
}
void deallocate_raw_bytes(void* ptr) {
    free(ptr); 
}
void deallocate_bytes(void* ptr) { 
    unsigned char* target = (unsigned char*)ptr;
    target -= *(size_t*)(target - sizeof(size_t));
    deallocate_raw_bytes(target); 
}
template<class T>
void deallocate(T* ptr) {
    ptr->~T();
    deallocate_bytes(ptr);
}

#include "driver.cpp"

int main(int argc, char const* argv[]) {

    return driver_main(argc, argv);
}
