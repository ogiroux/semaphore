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

template<class Semaphore>
__semaphore_abi inline void __condition_variable_atomic_impl_base<Semaphore>::__notify()
{
#ifndef __semaphore_cuda
    if (__semaphore_expect(!__reversebuffer.load(std::memory_order_relaxed), 1))
        return;
    (atomic_thread_fence)(std::memory_order_seq_cst);
    int const waiting = __reversebuffer.exchange(0, std::memory_order_relaxed);
    if (__semaphore_expect(waiting, 0))
        __sem.release(waiting);
#endif
}

template<class Semaphore>
template<class A, class Predicate>
__semaphore_abi void __condition_variable_atomic_impl_base<Semaphore>::__wait(A& object, Predicate pred, std::memory_order order)
{
    for (int i = 0; i < 16; ++i, __semaphore_yield())
        if (__semaphore_expect(pred(object.load(order)), 1))
            return;
    details::__semaphore_exponential_backoff b;
    while(1)
    {
#ifndef __semaphore_cuda
        __reversebuffer.fetch_add(1, std::memory_order_relaxed);
        (atomic_thread_fence)(std::memory_order_seq_cst);
        if (__semaphore_expect(pred(object.load(order)), 0))
        {
            auto const waiting = __reversebuffer.exchange(0, std::memory_order_relaxed);
            switch (waiting)
            {
            case 0:
                __sem.acquire(); // uuuuuuuuhhhh, this is really weird for for/until
            case 1:
                break;
            default:
                __sem.release(waiting - 1); break;
            }
            return;
        }
        __sem.__acquire_slow();
#else
        b.sleep();
#endif
        if(pred(object.load(order)))
            return;
    }
}

template<class Semaphore>
template<class A, class Predicate, class Clock, class Duration>
bool __condition_variable_atomic_impl_base<Semaphore>::__wait_until(A& object, Predicate pred, std::chrono::time_point<Clock, Duration> const& abs_time, std::memory_order order)
{
    for (int i = 0; i < 16; ++i, __semaphore_yield())
        if (__semaphore_expect(pred(object.load(order)), 1))
            return true;
    details::__semaphore_exponential_backoff b;
    while(1)
    {
        if(Clock::now() > abs_time)
            return false;
#ifndef __semaphore_cuda
        __reversebuffer.fetch_add(1, std::memory_order_relaxed);
        (atomic_thread_fence)(std::memory_order_seq_cst);
        if (__semaphore_expect(pred(object.load(order)), 0))
        {
            auto const waiting = __reversebuffer.exchange(0, std::memory_order_relaxed);
            switch (waiting)
            {
            case 0:
                __sem.acquire(); // uuuuuuuuhhhh, this is really weird for for/until
            case 1:
                break;
            default:
                __sem.release(waiting - 1); break;
            }
            return true;
        }
        __sem.__acquire_slow_timed(abs_time - Clock::now());
#else
        b.sleep();
#endif
        if(pred(object.load(order)))
            return true;
    }
}

template<class Semaphore>
template<class A, class Predicate, class Rep, class Period>
bool __condition_variable_atomic_impl_base<Semaphore>::__wait_for(A& object, Predicate pred, std::chrono::duration<Rep, Period> const& rel_time, std::memory_order order)
{
    return __wait_until(object, pred, details::__semaphore_clock::now() + rel_time, order);
}

template <class Semaphore>
__semaphore_abi inline __condition_variable_atomic_impl_base<Semaphore>::__condition_variable_atomic_impl_base()
#ifndef __semaphore_cuda
    : __sem(0), __reversebuffer{0}
#endif
{
}

__semaphore_abi inline condition_variable_atomic::condition_variable_atomic() : __condition_variable_atomic_impl_base()
{

}

__semaphore_abi inline condition_variable_atomic::~condition_variable_atomic() 
{
}    

template <class T>
__semaphore_abi void condition_variable_atomic::notify_one(const volatile atomic<T>&) 
{
    __notify(); 
}

template <class T>
__semaphore_abi void condition_variable_atomic::notify_one(const atomic<T>&)
{
    __notify(); 
}

template <class T>
__semaphore_abi void condition_variable_atomic::notify_all(const volatile atomic<T>&)
{
    __notify(); 
}

template <class T>
__semaphore_abi void condition_variable_atomic::notify_all(const atomic<T>&)
{
    __notify(); 
}

template <class T>
__semaphore_abi void condition_variable_atomic::wait(const volatile atomic<T>& object, T old, std::memory_order order)
{ 
    __wait(object, [=] __semaphore_abi (T other) -> bool { return old != other; }, order); 
}

template <class T>
__semaphore_abi void condition_variable_atomic::wait(const atomic<T>& object, T old, std::memory_order order)
{ 
    __wait(object, [=] __semaphore_abi (T other) -> bool { return old != other; }, order); 
}

template <class T, class Predicate>
__semaphore_abi void condition_variable_atomic::wait(const volatile atomic<T>& object, Predicate pred, std::memory_order order)
{ 
    __wait(object, pred, order); 
}

template <class T, class Predicate>
__semaphore_abi void condition_variable_atomic::wait(const atomic<T>& object, Predicate pred, std::memory_order order)
{ 
    __wait(object, pred, order); 
}

template <class T, class Clock, class Duration>
bool condition_variable_atomic::wait_until(const volatile atomic<T>& object, T old, std::chrono::time_point<Clock, Duration> const& abs_time, std::memory_order order)
{ 
    return __wait_until(object, [=] __semaphore_abi (T other) -> bool { return old != other; }, abs_time, order); 
}

template <class T, class Clock, class Duration>
bool condition_variable_atomic::wait_until(const atomic<T>& object, T old, std::chrono::time_point<Clock, Duration> const& abs_time, std::memory_order order)
{ 
    return __wait_until(object, [=] __semaphore_abi (T other) -> bool { return old != other; }, abs_time, order); 
}

template <class T, class Predicate, class Clock, class Duration>
bool condition_variable_atomic::wait_until(const volatile atomic<T>& object, Predicate pred, std::chrono::time_point<Clock, Duration> const& abs_time, std::memory_order order)
{ 
    return __wait_until(object, pred, abs_time, order); 
}

template <class T, class Predicate, class Clock, class Duration>
bool condition_variable_atomic::wait_until(const atomic<T>& object, Predicate pred, std::chrono::time_point<Clock, Duration> const& abs_time, std::memory_order order)
{ 
    return __wait_until(object, pred, abs_time, order); 
}

template <class T, class Rep, class Period>
bool condition_variable_atomic::wait_for(const volatile atomic<T>& object, T old, std::chrono::duration<Rep, Period> const& rel_time, std::memory_order order)
{ 
    return __wait_for(object, [=] __semaphore_abi (T other) -> bool { return old != other; }, rel_time, order); 
}

template <class T, class Rep, class Period>
bool condition_variable_atomic::wait_for(const atomic<T>& object, T old, std::chrono::duration<Rep, Period> const& rel_time, std::memory_order order)
{ 
    return __wait_for(object, [=] __semaphore_abi (T other) -> bool { return old != other; }, rel_time, order); 
}

template <class T, class Predicate, class Rep, class Period>
bool condition_variable_atomic::wait_for(const volatile atomic<T>& object, Predicate pred, std::chrono::duration<Rep, Period> const& rel_time, std::memory_order order)
{ 
    return __wait_for(object, pred, rel_time, order); 
}

template <class T, class Predicate, class Rep, class Period>
bool condition_variable_atomic::wait_for(const atomic<T>& object, Predicate pred, std::chrono::duration<Rep, Period> const& rel_time, std::memory_order order)
{ 
    return __wait_for(object, pred, rel_time, order); 
}
