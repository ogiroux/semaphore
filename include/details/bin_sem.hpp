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

__semaphore_abi inline void binary_semaphore::release()
{
    if(__stolen)
        __stolen = false;
    else
#ifdef __semaphore_cuda
        __tocket.fetch_add(1, std::memory_order_relaxed);
#else
        __tocket.store(__tocket.load(std::memory_order_relaxed)+1, std::memory_order_relaxed);
#endif
#ifdef __semaphore_fast_path
    count_type old = __valubit;
    if (__semaphore_expect(!__atom.compare_exchange_strong(old, 0, std::memory_order_acquire, std::memory_order_relaxed), 0))
        __release_slow(old);
#else
    __atom.fetch_and(~__valubit, std::memory_order_acquire);
#endif
}

__semaphore_abi inline bool binary_semaphore::try_acquire()
{
    count_type old = 0;
    for (int i = 0;i < 64; ++i) {
        if(__semaphore_expect(__atom.compare_exchange_strong(old = 0, __valubit, std::memory_order_acquire, std::memory_order_relaxed),1))
            return __stolen = true;
        for(; old != 0 && i < 64; ++i) {
            __semaphore_yield();
            old = __atom.load(std::memory_order_relaxed);
        }
    }
    return false;
}

__semaphore_abi inline void binary_semaphore::acquire()
{
    if (__semaphore_expect(try_acquire(), 1))
        return;
    __acquire_slow();
}

template <class Clock, class Duration>
bool binary_semaphore::try_acquire_until(const std::chrono::time_point<Clock, Duration>& abs_time)
{
    if (__semaphore_expect(try_acquire(), 1))
        return true;
    return __acquire_slow_timed(abs_time - Clock::now());
}

template <class Rep, class Period>
bool binary_semaphore::try_acquire_for(const std::chrono::duration<Rep, Period>& rel_time)
{
    if (__semaphore_expect(try_acquire(), 1))
        return true;
    return __acquire_slow_timed(rel_time);
}

__semaphore_abi inline __binary_semaphore_impl_base::__binary_semaphore_impl_base(__count_type desired) : __atom(desired ? 0 : 1), __ticket(0), __tocket(0), __stolen(false)
{
}

__semaphore_abi inline binary_semaphore::binary_semaphore(count_type desired) : __binary_semaphore_impl_base(desired)
{
}

__semaphore_abi inline binary_semaphore::~binary_semaphore()
{
}
