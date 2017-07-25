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

#include <cuda/atomic>
#include <cuda/semaphore>

namespace cuda { namespace experimental { inline namespace v1 {

    namespace detail {

        static constexpr int __atomic_lock_table_entry_size =
            sizeof(binary_semaphore) > alignof(binary_semaphore) ? 
            sizeof(binary_semaphore) : alignof(binary_semaphore);
        
        static constexpr int __atomic_lock_table_entry_count = 1024;

        __managed__ unsigned char __atomic_lock_table[__atomic_lock_table_entry_count][__atomic_lock_table_entry_size] = { 0 };

        __host__ __device__ size_t __atomic_lock_table_index(void const* ptr) { 

            return ((uintptr_t)ptr / __atomic_lock_table_entry_size) & (__atomic_lock_table_entry_count - 1);
        }

        __host__ __device__ void __lock_by_address(void const* addr) {

            auto const sem = (binary_semaphore*)&__atomic_lock_table[__atomic_lock_table_index(addr)][0];
            sem->acquire();
        }

        __host__ __device__ void __unlock_by_address(void const* addr) {

            auto const sem = (binary_semaphore*)&__atomic_lock_table[__atomic_lock_table_index(addr)][0];
            sem->release();
        }
    }
    
} } }
