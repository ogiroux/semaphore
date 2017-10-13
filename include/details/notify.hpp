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

// 32.10, waiting and notifying functions

#ifndef __semaphore_cuda
#define __get_condition_variable_atomic_from_ptr(a) (*__condition_variable_atomic_from_ptr(a))
#else
#define __get_condition_variable_atomic_from_ptr(a) condition_variable_atomic()
#endif

template <class T>
__semaphore_abi void atomic_notify_one(const volatile atomic<T>*a)
{
    __get_condition_variable_atomic_from_ptr(a).notify_one(*a);
}

template <class T>
__semaphore_abi void atomic_notify_one(const atomic<T>*a)
{
    __get_condition_variable_atomic_from_ptr(a).notify_one(*a);
}

template <class T>
__semaphore_abi void atomic_notify_all(const volatile atomic<T>* a)
{
    __get_condition_variable_atomic_from_ptr(a).notify_all(*a);
}

template <class T>
__semaphore_abi void atomic_notify_all(const atomic<T>* a)
{
    __get_condition_variable_atomic_from_ptr(a).notify_all(*a);
}

template <class T, class V>
__semaphore_abi void atomic_wait_explicit(const volatile atomic<T>* a, V oldval, std::memory_order order)
{
    __get_condition_variable_atomic_from_ptr(a).wait(*a, oldval, order);
}

template <class T, class V>
__semaphore_abi void atomic_wait_explicit(const atomic<T>* a, V oldval, std::memory_order order)
{
    __get_condition_variable_atomic_from_ptr(a).wait(*a, oldval, order);
}

template <class T, class V>
__semaphore_abi void atomic_wait(const volatile atomic<T>* a, V oldval)
{
    __get_condition_variable_atomic_from_ptr(a).wait(*a, oldval);
}

template <class T, class V>
__semaphore_abi void atomic_wait(const atomic<T>* a, V oldval)
{
    __get_condition_variable_atomic_from_ptr(a).wait(*a, oldval);
}
