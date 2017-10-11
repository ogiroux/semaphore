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

#ifndef __semaphore_config__
#define __semaphore_config__

#include <atomic>
#include <chrono>
#include <thread>
#include <cassert>
#include <algorithm>

#ifdef WIN32
#include <windows.h>
#endif //WIN32

#ifdef __linux__
#include <time.h>
#include <cstring>
#include <unistd.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <climits>
#include <semaphore.h>
#endif //__linux__

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif //__APPLE__

#if defined(__CUDA_ARCH__) && !defined(__has_cuda_nanosleep)
    #warning "Use of semaphores is iffy without CUDA support for nanosleep (CUDA version TBD)."
#endif //__CUDA_ARCH__

#ifdef __semaphore_cuda
    #define __semaphore_ns cuda
#else
    #define __semaphore_ns std
#endif //__semaphore_cuda

namespace __semaphore_ns
{
namespace experimental
{
inline namespace v1
{
namespace details
{

#if defined(__semaphore_cuda)

#ifndef __has_cuda_atomic
    #error "CUDA atomic<T> support is required"
#endif //__has_cuda_atomic

#define __semaphore_abi __host__ __device__
#define __semaphore_managed __managed__

__semaphore_abi inline void __semaphore_yield()
{
#if !defined(__CUDA_ARCH__)
    std::this_thread::yield();
#elif defined(__has_cuda_nanosleep)
    __mme_nanosleep(1);
#endif
}
#define __semaphore_expect(c, e) (c)

#else //__semaphore_cuda

#define __semaphore_abi
#define __semaphore_managed

#if defined(__GNUC__)
    #define __semaphore_expect __builtin_expect
#else //__GNUC__
    #define __semaphore_expect(c, e) (c)
#endif //__GNUC__

#ifdef WIN32

#if _WIN32_WINNT >= 0x0602
    #define __semaphore_fast_path
#endif //_WIN32_WINNT
#define __semaphore_sem
#define  __semaphore_sem_t HANDLE

inline bool __semaphore_sem_init(__semaphore_sem_t &sem, int init)
{
    bool const ret = (sem = CreateSemaphore(NULL, init, INT_MAX, NULL)) != NULL;
    assert(ret);
    return ret;
}

inline bool __semaphore_sem_destroy(__semaphore_sem_t &sem)
{
    assert(sem != NULL);
    return CloseHandle(sem) == TRUE;
}

inline bool __semaphore_sem_post(__semaphore_sem_t &sem, int inc)
{
    assert(sem != NULL);
    assert(inc > 0);
    return ReleaseSemaphore(sem, inc, NULL) == TRUE;
}

inline bool __semaphore_sem_wait(__semaphore_sem_t &sem)
{
    assert(sem != NULL);
    return WaitForSingleObject(sem, INFINITE) == WAIT_OBJECT_0;
}

template <class Rep, class Period>
inline bool __semaphore_sem_wait_timed(__semaphore_sem_t &sem, chrono::duration<Rep, Period> const &delta)
{
    assert(sem != NULL);
    return WaitForSingleObject(sem, (DWORD)chrono::duration_cast<chrono::milliseconds>(delta).count()) == WAIT_OBJECT_0;
}

#endif //WIN32

#ifdef __linux__

#define __semaphore_fast_path
#define __semaphore_back_buffered
#define __semaphore_sem
#define __semaphore_sem_t sem_t

template <class Rep, class Period>
timespec __semaphore_to_timespec(chrono::duration<Rep, Period> const &delta)
{
    struct timespec ts;
    ts.tv_sec = static_cast<long>(chrono::duration_cast<chrono::seconds>(delta).count());
    ts.tv_nsec = static_cast<long>(chrono::duration_cast<chrono::nanoseconds>(delta).count());
    return ts;
}

inline bool __semaphore_sem_init(__semaphore_sem_t &sem, int init)
{
    return sem_init(&sem, 0, init) == 0;
}

inline bool __semaphore_sem_destroy(__semaphore_sem_t &sem)
{
    return sem_destroy(&sem) == 0;
}

inline bool __semaphore_sem_post(__semaphore_sem_t &sem, int inc)
{
    assert(inc == 1);
    return sem_post(&sem) == 0;
}

inline bool __semaphore_sem_wait(__semaphore_sem_t &sem)
{
    return sem_wait(&sem) == 0;
}

template <class Rep, class Period>
inline bool __semaphore_sem_wait_timed(__semaphore_sem_t &sem, chrono::duration<Rep, Period> const &delta)
{
    auto const timespec = __semaphore_to_timespec(delta);
    return sem_timedwait(&sem, &timespec) == 0;
}

inline void __semaphore_yield()
{
    sched_yield();
}

#else //__linux__

inline void __semaphore_yield()
{
    this_thread::yield();
}

#endif //__linux__

#ifdef __APPLE__

#define __semaphore_back_buffered
#define __semaphore_sem
#define __semaphore_sem_t dispatch_semaphore_t

inline bool __semaphore_sem_init(__semaphore_sem_t &sem, int init)
{
    return (sem = dispatch_semaphore_create(init)) != NULL;
}

inline bool __semaphore_sem_destroy(__semaphore_sem_t &sem)
{
    assert(sem != NULL);
    dispatch_release(sem);
    return true;
}

inline bool __semaphore_sem_post(__semaphore_sem_t &sem, int inc)
{
    assert(inc == 1);
    dispatch_semaphore_signal(sem);
    return true;
}

inline bool __semaphore_sem_wait(__semaphore_sem_t &sem)
{
    return dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER) == 0;
}

template <class Rep, class Period>
inline bool __semaphore_sem_wait_timed(__semaphore_sem_t &sem, chrono::duration<Rep, Period> const &delta)
{
    return dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, chrono::duration_cast<chrono::nanoseconds>(delta).count())) == 0;
}

#endif //__APPLE__

#endif //__semaphore_cuda

using __semaphore_clock = std::conditional<std::chrono::high_resolution_clock::is_steady,
                                           std::chrono::high_resolution_clock,
                                           std::chrono::steady_clock>::type;

using __semaphore_duration = std::chrono::nanoseconds;

// A simple exponential back-off helper that is designed to cover the space between (1<<__magic_number_3) and __magic_number_4
struct __semaphore_exponential_backoff
{
    static constexpr unsigned max_time = 128*64;
    static constexpr unsigned min_time = 64;
    unsigned time = min_time, time_sum = 0;

    __semaphore_abi void reset() {
        time = min_time;
        time_sum = 0;
    }

    __semaphore_abi void sleep(unsigned maximum = max_time)
    {
        {
            auto const this_time = time > maximum ? maximum : time;
            time_sum += this_time;
#if !defined(__CUDA_ARCH__)
            std::this_thread::sleep_for(std::chrono::nanoseconds(this_time));
#elif defined(__has_cuda_nanosleep)
            __mme_nanosleep(time);
#endif //__CUDA_ARCH__
        }
        time += min_time + (time >> 2);
        if (time > max_time) 
            time = max_time;
    }
};

} //details
} //v1
} //experimental
} //__semaphore_ns

#endif //__semaphore_config__
