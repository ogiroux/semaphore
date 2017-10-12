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

#ifndef __semaphore_sem

__semaphore_abi inline void counting_semaphore::release(count_type term)
{

#ifdef __semaphore_fast_path
    count_type old = 0, set = term << __shift;
    bool success = atom.compare_exchange_weak(old, set, std::memory_order_release, std::memory_order_relaxed);
    while (__semaphore_expect(!success && !(old & (__contmask | __lockmask)), 0))
    {
        set = old + (term << __shift);
        success = atom.compare_exchange_weak(old, set, std::memory_order_release, std::memory_order_relaxed);
    }
    if (__semaphore_expect(!success, 0))
        __fetch_add_slow(term, old, std::memory_order_release, notify);
#else
    count_type old = atom.fetch_add(term, std::memory_order_release);
#endif
}

__semaphore_abi inline void counting_semaphore::acquire() 
{

    while (__semaphore_expect(!__fetch_sub_if(), 0))
    {
        bool const success = __acquire_fast();
        if (__semaphore_expect(!success, 0))
            __acquire_slow();
    }
}
__semaphore_abi inline bool counting_semaphore::try_acquire() 
{

    return __fetch_sub_if();
}

template <class Clock, class Duration>
bool counting_semaphore::try_acquire_until(std::chrono::time_point<Clock, Duration> const &abs_time)
{

    while (__semaphore_expect(!__fetch_sub_if(), 0))
    {
        bool success = __acquire_fast();
        if (__semaphore_expect(!success, 0))
            success = __acquire_slow_timed(abs_time - Clock::now());
        if (__semaphore_expect(!success, 0))
            return false;
    }
    return true;
}

template <class Rep, class Period>
bool counting_semaphore::try_acquire_for(std::chrono::duration<Rep, Period> const &rel_time)
{

    if (__semaphore_expect(__fetch_sub_if(), 1))
        return true;
    else
        return try_acquire_until(details::__semaphore_clock::now() + rel_time);
}

__semaphore_abi inline counting_semaphore::counting_semaphore(count_type desired) : atom(desired << __shift)
{
}

__semaphore_abi inline counting_semaphore::~counting_semaphore()
{
}

__semaphore_abi inline bool counting_semaphore::__fetch_sub_if()
{

    count_type old = 1 << __shift, set = 0;
    bool retcode = atom.compare_exchange_weak(old, set, std::memory_order_acquire, std::memory_order_relaxed);
    if (__semaphore_expect(!retcode && (old >> __shift) >= 1, 0))
    {
        old &= __valumask;
        set = old - (1 << __shift);
        retcode = atom.compare_exchange_weak(old, set, std::memory_order_acquire, std::memory_order_relaxed);
    }
    if (__semaphore_expect(!retcode && (old >> __shift) >= 1, 0))
        retcode = __fetch_sub_if_slow(old);
    return retcode;
}

__semaphore_abi inline bool counting_semaphore::__acquire_fast()
{

    auto value = (atom.load(std::memory_order_acquire) >> __shift);
    if (__semaphore_expect(value >= 1, 1))
        return true;
    for (int i = 0; i < 32; ++i)
    {
        __semaphore_yield();
        value = (atom.load(std::memory_order_acquire) >> __shift);
        if (__semaphore_expect(value >= 1, 1))
            return true;
    }
    return false;
}

#else //__semaphore_sem

__semaphore_abi inline void counting_semaphore::release(count_type term)
{
    count_type old = __frontbuffer.load(std::memory_order_relaxed);
    while (1)
    {
        old &= ~1;
        if (__frontbuffer.compare_exchange_weak(old, old + (term << 1) + 1, std::memory_order_release, std::memory_order_relaxed))
            break;
    }
    if ((old >> 1) < 0)
    { // was it depleted?
        count_type inc = (min)(-(old >> 1), term);
#ifdef __semaphore_back_buffered
        __backbuffer.fetch_add(inc - 1);
        inc = 1;
#endif //__semaphore_back_buffered
        count_type const result = __semaphore_sem_post(__semaphore, inc);
#ifdef WIN32
        if (!result)
        {
            auto d = GetLastError();
            assert(d == ERROR_SUCCESS);
        }
#endif //WIN32
        assert(result);
    }
    __frontbuffer.fetch_sub(1);
}

__semaphore_abi inline void counting_semaphore::acquire()
{
    __acquire_fast();
    __acquire_slow();
}

template <class Rep, class Period>
bool counting_semaphore::try_acquire_for(std::chrono::duration<Rep, Period> const &rel_time)
{
    __acquire_fast();
    if (__frontbuffer.fetch_sub(2, std::memory_order_acquire) >> 1 > 0)
        return true;
    auto const result = __semaphore_sem_wait_timed(__semaphore, rel_time);
    if (result)
        __backfill();
    return result;
}

template <class Clock, class Duration>
bool counting_semaphore::try_acquire_until(std::chrono::time_point<Clock, Duration> const &abs_time)
{

    return try_acquire_for(abs_time - Clock::now());
}

__semaphore_abi inline bool counting_semaphore::try_acquire()
{
    __acquire_fast();
    if (__frontbuffer.fetch_sub(2, std::memory_order_acquire) >> 1 > 0)
        return true;
    return try_acquire_for(std::chrono::nanoseconds(0));
}

__semaphore_abi inline counting_semaphore::counting_semaphore(count_type desired)
    : __frontbuffer { desired << 1 }
#ifdef __semaphore_back_buffered
    , __backbuffer{ 0 }
#endif //__semaphore_back_buffered
{
    auto const result = __semaphore_sem_init(__semaphore, desired);
#ifdef WIN32
    if (!result)
    {
        auto d = GetLastError();
        assert(d == ERROR_SUCCESS);
    }
#endif //WIN32
    assert(result);
}

__semaphore_abi inline counting_semaphore::~counting_semaphore()
{

    while (__frontbuffer.load(std::memory_order_acquire) & 1)
        ;
    auto const result = __semaphore_sem_destroy(__semaphore);
    assert(result);
}

__semaphore_abi inline void counting_semaphore::__acquire_fast()
{
    if (__semaphore_expect(__frontbuffer.load(std::memory_order_relaxed) > 1, 1))
        return;
    for (int i = 0; i < 32; ++i)
    {
        __semaphore_yield();
        if (__semaphore_expect(__frontbuffer.load(std::memory_order_relaxed) > 1, 1))
            return;
    }
}

__semaphore_abi inline void counting_semaphore::__acquire_slow() {
    if (__frontbuffer.fetch_sub(2, std::memory_order_acquire) >> 1 > 0)
        return;
    count_type const result = __semaphore_sem_wait(__semaphore);
#ifdef WIN32
    if (!result)
    {   
        auto d = GetLastError();
        assert(d == ERROR_SUCCESS);
    }
#endif //WIN32
    assert(result);
    __backfill();
}

void inline counting_semaphore::__acquire_slow_timed(std::chrono::nanoseconds const& rel_time) {
    assert(0);
}

__semaphore_abi inline void counting_semaphore::__backfill()
{
#ifdef __semaphore_back_buffered
    if (__semaphore_expect(__backbuffer.load(std::memory_order_relaxed) == 0, 1))
        return;
    if (__semaphore_expect(__backbuffer.fetch_sub(1, std::memory_order_relaxed) == 0, 0))
    {
        __backbuffer.fetch_add(1, std::memory_order_relaxed); // put back
        return;
    }
    auto const result = __semaphore_sem_post(__semaphore, 1);
    assert(result);
#endif //__semaphore_back_buffered
}

#endif //__semaphore_sem
