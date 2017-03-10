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

#ifndef binary_semaphore_hpp
#define binary_semaphore_hpp

#include <atomic>
#include <chrono>
#include <thread>
#include <cassert>
#include <algorithm>

#if defined(__GNUC__)
    #define __semaphore_expect __builtin_expect
#else
    #define __semaphore_expect(c,e) (c)
#endif

#ifdef WIN32
    #include <windows.h>
    #ifdef min
        #define __semaphore_old_min min
        #undef min
    #endif
    #ifdef max
        #define __semaphore_old_max max
        #undef max
    #endif
    typedef HANDLE __semaphore_sem_t;
    inline bool __semaphore_sem_init(__semaphore_sem_t& sem, int init) { 
        bool const ret = (sem = CreateSemaphore(NULL, init, INT_MAX, NULL)) != NULL;
        assert(ret);
        return ret;
    }
    inline bool __semaphore_sem_destroy(__semaphore_sem_t& sem) { 
        assert(sem != NULL);
        return CloseHandle(sem) == TRUE;
    }
    inline bool __semaphore_sem_post(__semaphore_sem_t& sem, int inc) { 
        assert(sem != NULL);
        assert(inc > 0);
        return ReleaseSemaphore(sem, inc, NULL) == TRUE; 
    }
    inline bool __semaphore_sem_wait(__semaphore_sem_t& sem) { 
        assert(sem != NULL);
        return WaitForSingleObject(sem, INFINITE) == WAIT_OBJECT_0;
    }
    template < class Rep, class Period>
    inline bool __semaphore_sem_wait_timed(__semaphore_sem_t& sem, std::chrono::duration<Rep, Period> const& delta) { 
        assert(sem != NULL);
        return WaitForSingleObject(sem, std::chrono::milliseconds(delta).count()) == WAIT_OBJECT_0;
    }
    #if _WIN32_WINNT >= 0x0602
        #define __semaphore_fast_path
    #endif
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
    template < class Rep, class Period>
    timespec __semaphore_to_timespec(std::chrono::duration<Rep, Period> const& delta) {
        struct timespec ts;
        ts.tv_sec = static_cast<long>(std::chrono::duration_cast<std::chrono::seconds>(delta).count());
        ts.tv_nsec = static_cast<long>(std::chrono::duration_cast<std::chrono::nanoseconds>(delta).count());
        return ts;
    }
    typedef sem_t __semaphore_sem_t;
    inline bool __semaphore_sem_init(__semaphore_sem_t& sem, int init) { 
        return sem_init(&sem, 0, init) == 0; 
    }
    inline bool __semaphore_sem_destroy(__semaphore_sem_t& sem) { 
        return sem_destroy(&sem) == 0; 
    }
    inline bool __semaphore_sem_post(__semaphore_sem_t& sem, int inc) { 
        assert(inc == 1);
        return sem_post(&sem) == 0; 
    }
    inline bool __semaphore_sem_wait(__semaphore_sem_t& sem) { 
        return sem_wait(&sem) == 0;
    }
    template < class Rep, class Period>
    inline bool __semaphore_sem_wait_timed(__semaphore_sem_t& sem, std::chrono::duration<Rep, Period> const& delta) { 
        auto const timespec = __semaphore_to_timespec(delta);
        return sem_timedwait(&sem, &timespec) == 0;
    }
    inline void __semaphore_yield() { 
        sched_yield(); 
    }
    #define __semaphore_fast_path
    #define __semaphore_back_buffered
#else
    inline void __semaphore_yield() { 
        std::this_thread::yield(); 
    }
#endif

#ifdef __APPLE__
    #include <dispatch/dispatch.h>
    typedef dispatch_semaphore_t __semaphore_sem_t;
    inline bool __semaphore_sem_init(__semaphore_sem_t& sem, int init) { 
        return (sem = dispatch_semaphore_create(init)) != NULL; 
    }
    inline bool __semaphore_sem_destroy(__semaphore_sem_t& sem) { 
        assert(sem != NULL);
        dispatch_release(sem); 
        return true;
    }
    inline bool __semaphore_sem_post(__semaphore_sem_t& sem, int inc) { 
        assert(inc == 1);
        dispatch_semaphore_signal(sem);
        return true;
    }
    inline bool __semaphore_sem_wait(__semaphore_sem_t& sem) { 
        return dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER) == 0;
    }
    template < class Rep, class Period>
    inline bool __semaphore_sem_wait_timed(__semaphore_sem_t& sem, std::chrono::duration<Rep, Period> const& delta) { 
        return dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, std::chrono::duration_cast<std::chrono::nanoseconds>(delta).count())) == 0;
    }
    #define __semaphore_back_buffered
#endif

namespace std {
    namespace experimental {
        inline namespace concurrency_v2 {

            using __semaphore_clock = conditional<chrono::high_resolution_clock::is_steady,
                                                  chrono::high_resolution_clock, 
                                                  chrono::steady_clock>::type;

            using __semaphore_duration = chrono::nanoseconds;

            // A simple exponential back-off helper that is designed to cover the space between (1<<__magic_number_3) and __magic_number_4
            class __semaphore_exponential_backoff {
                int microseconds = 50;
            public:
                void sleep(int us = 0) {
                    if (us != 0)
                        microseconds = us;
                    this_thread::sleep_for(chrono::microseconds(microseconds));
                    // Avoiding the use of std::min here, to keep includes minimal
                    auto next_microseconds = microseconds + (microseconds >> 2);
                    microseconds = next_microseconds < 8192 ? next_microseconds : 8192;
                }
            };

            enum class semaphore_notify {
                all, one, none
            };

            typedef struct __binary_semaphore {

                typedef uint8_t __base_t;

                static constexpr __base_t __valubit = 1;
#ifdef __semaphore_fast_path
                static constexpr __base_t __contbit = 2;
                static constexpr __base_t __lockbit = 4;
#endif

#ifdef __semaphore_fast_path
                bool __test_and_set_slow(__base_t old, memory_order order) noexcept;
#endif

                bool __test_and_set(memory_order order) noexcept {
                    __base_t old = 0;
                    bool const success = atom.compare_exchange_weak(old, __valubit, order, memory_order_relaxed);
                    bool retcode = (old & __valubit) == 1;
#ifdef __semaphore_fast_path
                    if (__semaphore_expect(!success && !retcode, 0))
                        retcode = __test_and_set_slow(old, order);
#endif
#ifdef __semaphore_arm
                    if (!retcode) {
                        __asm__ __volatile__(
                            "   dsb\n"
                            "   sev"
                        );
                    }
#endif
                    return retcode;
                }

#ifdef __semaphore_fast_path
                void __release_slow(__base_t old, memory_order order, semaphore_notify notify) noexcept;
#endif            

                inline void release(memory_order order = memory_order_seq_cst, semaphore_notify notify = semaphore_notify::all) noexcept {

#ifdef __semaphore_fast_path
                    __base_t old = __valubit;
                    bool const success = atom.compare_exchange_weak(old, 0, order, memory_order_relaxed);
                    if (__semaphore_expect(!success, 0)) {
                        bool const success2 = ((old & ~__valubit) == 0) && atom.compare_exchange_weak(old, 0, order, memory_order_relaxed);
                        if (__semaphore_expect(!success2, 0))
                            __release_slow(old, order, notify);
                    }
#else
                    atom.store(0, order);
#endif
#ifdef __semaphore_arm
                    __asm__ __volatile__(
                        "   dsb\n"
                        "   sev"
                    );
#endif
                }

                void __wait_slow(memory_order order) noexcept;

                inline bool __wait_fast(memory_order order) noexcept {

                    __base_t old = atom.load(order);
                    __base_t const expectbit = 0; //(set ? __valubit : 0);
                    if (__semaphore_expect(old == expectbit, 1))
                        return true;
#ifdef __semaphore_arm
                    if ((old & __valubit) != expectbit)
                        for (int i = 0; i < 4; ++i) {
                            __base_t const tmp = old;
                            __asm__ __volatile__(
                                "ldrex %0, [%1]\n"
                                "cmp %0, %2\n"
                                "it eq\n"
                                "wfeeq.n\n"
                                "nop.w\n"
                                : "=&r" (old) : "r" (&atom), "r" (tmp) : "cc"
                            );
                            if ((old & __valubit) == expectbit) { atomic_thread_fence(order); break; }
                        }
#endif
                    if ((old & __valubit) != expectbit)
                        for (int i = 0; i < 32; ++i) {
                            __semaphore_yield();
                            old = atom.load(order);
                            if ((old & __valubit) == expectbit) break;
                        }
                    return old == expectbit;
                }

                inline void acquire(memory_order order = memory_order_seq_cst) noexcept {

                    while (__semaphore_expect(__test_and_set(order), 0)) {
                        bool const success = __wait_fast(order);
                        if (__semaphore_expect(!success, 0))
                            __wait_slow(order);
                    }
                }

                bool __wait_slow_timed(chrono::time_point<__semaphore_clock, __semaphore_duration> const& abs_time, memory_order order) noexcept;

                template <class Clock, class Duration>
                bool acquire_until(chrono::time_point<Clock, Duration> const& abs_time, memory_order order = memory_order_seq_cst) {

                    while (__semaphore_expect(__test_and_set(order), 0)) {
                        bool success = __wait_fast(order);
                        if (__semaphore_expect(!success, 0))
                            success = __wait_slow_timed(abs_time, order);
                        if (__semaphore_expect(!success, 0))
                            return false;
                    }
                    return true;
                }
                template <class Rep, class Period>
                bool acquire_for(chrono::duration<Rep, Period> const& rel_time, memory_order order = memory_order_seq_cst) {

                    if(__semaphore_expect(__test_and_set(order), 0))
                        return true;
                    else
                        return acquire_until(__semaphore_clock::now() + rel_time, order);
                }
                inline bool try_acquire(std::memory_order order = std::memory_order_seq_cst) noexcept {

                    return acquire_for(chrono::nanoseconds(0), order);
                }

                __binary_semaphore(bool available) noexcept : atom(!available) { }
                __binary_semaphore() noexcept = default;
                __binary_semaphore(const __binary_semaphore&) = delete;
                __binary_semaphore& operator=(const __binary_semaphore&) = delete;

                mutable atomic<__base_t> atom{ false };

            } binary_semaphore;

            struct __counting_semaphore {

#ifdef __semaphore_fast_path
                static constexpr int __valumask = ~3,
                                     __contmask =  2,
                                     __lockmask =  1,
                                     __shift = 2;
#else
                static constexpr int __valumask = ~0,
                                     __contmask =  0,
                                     __lockmask =  0,
                                     __shift = 0;
#endif

                static constexpr int max() { 

                    return int(unsigned(__valumask) >> __shift); 
                }

                bool __fetch_sub_if_slow(int old, memory_order order) noexcept;

                bool __fetch_sub_if(memory_order order) noexcept {

                    int old = 1 << __shift, set = 0;
                    bool retcode = atom.compare_exchange_weak(old, set, order, memory_order_relaxed);
                    if (__semaphore_expect(!retcode && (old >> __shift) >= 1, 0)) {
                        old &= __valumask;
                        set = old - (1<< __shift);
                        retcode = atom.compare_exchange_weak(old, set, order, memory_order_relaxed);
                    }
                    if (__semaphore_expect(!retcode && (old >> __shift) >= 1, 0))
                        retcode = __fetch_sub_if_slow(old, order);
#ifdef __semaphore_arm
                    if (!retcode) {
                        __asm__ __volatile__(
                            "   dsb\n"
                            "   sev"
                        );
                    }
#endif
                    return retcode;
                }

#ifdef __semaphore_fast_path
                void __fetch_add_slow(int term, int old, memory_order order, semaphore_notify notify) noexcept;
#endif                

                inline void release(int term, memory_order order = memory_order_seq_cst, semaphore_notify notify = semaphore_notify::all) noexcept {

#ifdef __semaphore_fast_path
                    int old = 0, set = term << __shift;
                    bool success = atom.compare_exchange_weak(old, set, order, memory_order_relaxed);
                    while (__semaphore_expect(!success && !(old & (__contmask | __lockmask)), 0)) {
                        set = old + (term << __shift);
                        success = atom.compare_exchange_weak(old, set, order, memory_order_relaxed);
                    }
                    if (__semaphore_expect(!success, 0))
                        __fetch_add_slow(term, old, order, notify);
#else
                    int old = atom.fetch_add(term, order);
#endif

#ifdef __semaphore_arm
                    __asm__ __volatile__(
                        "   dsb\n"
                        "   sev"
                    );
#endif
                }

                bool __wait_slow_timed(int c, chrono::time_point<__semaphore_clock, __semaphore_duration> const& abs_time, memory_order order) noexcept;

                void __wait_slow(memory_order order) noexcept;

                inline bool __wait_fast(memory_order order) noexcept {

                    auto value = (atom.load(order) >> __shift);
                    if (__semaphore_expect(value >= 1, 1))
                        return true;
#ifdef __semaphore_arm
                    for (int i = 0; i < 4; ++i) {
                        auto const tmp = old;
                        __asm__ __volatile__(
                            "ldrex %0, [%1]\n"
                            "cmp %0, %2\n"
                            "it eq\n"
                            "wfeeq.n\n"
                            "nop.w\n"
                            : "=&r" (old) : "r" (&atom), "r" (tmp) : "cc"
                        );
                        value = (atom.load(order) >> __shift);
                        if (__semaphore_expect(value >= 1, 1)) {
                            atomic_thread_fence(order); 
                            return true; 
                        }
                    }
#endif
                    for (int i = 0; i < 32; ++i) {
                        __semaphore_yield();
                        value = (atom.load(order) >> __shift);
                        if (__semaphore_expect(value >= 1, 1)) return true;
                    }
                    return false;
                }

                inline void acquire(int term = 1, memory_order order = memory_order_seq_cst) noexcept {

                    while (__semaphore_expect(!__fetch_sub_if(order), 0)) {
                        bool const success = __wait_fast(order);
                        if (__semaphore_expect(!success, 0))
                            __wait_slow(order);
                    }
                }

                template <class Clock, class Duration>
                bool acquire_until(chrono::time_point<Clock, Duration> const& abs_time, int term = 1, memory_order order = memory_order_seq_cst) {

                    while (__semaphore_expect(!__fetch_sub_if(order), 0)) {
                        bool success = __wait_fast(order);
                        if (__semaphore_expect(!success, 0))
                            success = __wait_slow_timed(term, abs_time, order);
                        if (__semaphore_expect(!success, 0))
                            return false;
                    }
                    return true;
                }
                template <class Rep, class Period>
                bool acquire_for(chrono::duration<Rep, Period> const& rel_time, int term = 1, memory_order order = memory_order_seq_cst) {

                    if (__semaphore_expect(__fetch_sub_if(order), 1))
                        return true;
                    else
                        return acquire_until(__semaphore_clock::now() + chrono::duration_cast<std::chrono::nanoseconds>(rel_time), order);
                }
                inline bool try_acquire(std::memory_order order = std::memory_order_seq_cst) noexcept {

                    return acquire_for(chrono::nanoseconds(0), order);
                }

                __counting_semaphore(int initial) noexcept : atom(initial << __shift) {
                    assert(initial >= 0 && initial <= max());
                }
                __counting_semaphore() noexcept = default;
                __counting_semaphore(const __counting_semaphore&) = delete;
                __counting_semaphore& operator=(const __counting_semaphore&) = delete;

            private:
                atomic<int> __reversebuffer{ 0 };
                atomic<int> atom{ 0 };

                template <class T>
                friend void __atomic_notify_semaphore(atomic<T> const* a, __counting_semaphore* s);

                template <class T, class V, class Fun>
                friend bool __atomic_wait_semaphore(atomic<T> const* a, V oldval, __counting_semaphore* s, memory_order order, Fun fun);
            };

            struct buffered_semaphore {

                static constexpr int max_limit = numeric_limits<int>::max() >> 1;

                inline void release(int term, std::memory_order order = std::memory_order_seq_cst,
                    std::experimental::semaphore_notify notify = std::experimental::semaphore_notify::all) noexcept {

                    assert(term > 0 && term <= max_limit);

                    auto old = __frontbuffer.load(std::memory_order_relaxed);
                    while(1) {
                        old &= ~1;
                        if (__frontbuffer.compare_exchange_weak(old, old + (term << 1) + 1, order, std::memory_order_relaxed))
                            break;
                    }
                    if (old >> 1 < 0) { // was it depleted?
                        auto inc = (std::min)(-(old >> 1), term);
#ifdef __semaphore_back_buffered
                        __backbuffer.fetch_add(inc - 1);
                        inc = 1;
#endif
                        auto const result = __semaphore_sem_post(__semaphore, inc);
#ifdef WIN32
                        if (!result) {
                            auto d = GetLastError();
                            assert(d == ERROR_SUCCESS);
                        }
#endif
                        assert(result);
                    }
                    __frontbuffer.fetch_sub(1);
                }

                inline void __wait_fast() noexcept {

                    if (__semaphore_expect(__frontbuffer.load(std::memory_order_relaxed) > 1, 1))
                        return;
#ifdef __semaphore_arm
                    for (int i = 0; i < 4; ++i) {
                        __base_t const tmp = old;
                        __asm__ __volatile__(
                            "ldrex %0, [%1]\n"
                            "cmp %0, %2\n"
                            "it eq\n"
                            "wfeeq.n\n"
                            "nop.w\n"
                            : "=&r" (old) : "r" (&atom), "r" (tmp) : "cc"
                        );
                        if (__semaphore_expect(__frontbuffer.load(std::memory_order_relaxed) > 1, 1))
                            return;
                    }
#endif
                    for (int i = 0; i < 32; ++i) {
                        __semaphore_yield();
                        if (__semaphore_expect(__frontbuffer.load(std::memory_order_relaxed) > 1, 1))
                            return;
                    }
                }

                inline void __backfill() {
#ifdef __semaphore_back_buffered
                    if (__semaphore_expect(__backbuffer.load(std::memory_order_relaxed) == 0, 1))
                        return;
                    if (__semaphore_expect(__backbuffer.fetch_sub(1, std::memory_order_relaxed) == 0,0)) {
                        __backbuffer.fetch_add(1, std::memory_order_relaxed); // put back
                        return;
                    }
                    auto const result = __semaphore_sem_post(__semaphore, 1);
                    assert(result);
#endif
                }

                inline void acquire(std::memory_order order = std::memory_order_seq_cst) noexcept {

                    __wait_fast();
                    if (__frontbuffer.fetch_sub(2, order) >> 1 > 0)
                        return;
                    auto const result = __semaphore_sem_wait(__semaphore);
#ifdef WIN32
                    if (!result) {
                        auto d = GetLastError();
                        assert(d == ERROR_SUCCESS);
                    }
#endif
                    assert(result);
                    __backfill();
                }
                template <class Rep, class Period>
                bool acquire_for(std::chrono::duration<Rep, Period> const& rel_time, std::memory_order order = std::memory_order_seq_cst) {

                    __wait_fast();
                    if (__frontbuffer.fetch_sub(2, order) >> 1 > 0)
                        return true;
                    auto const result = __semaphore_sem_wait_timed(__semaphore, rel_time);
                    if(result)
                        __backfill();
                    return result;
                }
                template <class Clock, class Duration>
                bool acquire_until(std::chrono::time_point<Clock, Duration> const& abs_time, std::memory_order order = std::memory_order_seq_cst) {
        
                    return acquire_for(abs_time - Clock::now(), order);
                }
                inline bool try_acquire(std::memory_order order = std::memory_order_seq_cst) noexcept {

                    __wait_fast();
                    if (__frontbuffer.fetch_sub(2, order) >> 1 > 0)
                        return true;
                    return acquire_for(chrono::nanoseconds(0), order);
                }
                buffered_semaphore(int initial = 0) noexcept 
                    : __reversebuffer{ 0 }
                    , __frontbuffer{ initial << 1 }
#ifdef __semaphore_back_buffered
                    , __backbuffer{ 0 } 
#endif
                {
                    assert(initial >= 0 && initial <= max_limit);

                    auto const result = __semaphore_sem_init(__semaphore, initial);
#ifdef WIN32
                    if (!result) {
                        auto d = GetLastError();
                        assert(d == ERROR_SUCCESS);
                    }
#endif
                    assert(result);
                }
                ~buffered_semaphore() {

                    while (__frontbuffer.load(std::memory_order_acquire) & 1)
                        ;
                    auto const result = __semaphore_sem_destroy(__semaphore);
                    assert(result);
                }
                buffered_semaphore(const buffered_semaphore&) = delete;
                buffered_semaphore& operator=(const buffered_semaphore&) = delete;

            private:
                __semaphore_sem_t __semaphore;
                std::atomic<int> __reversebuffer;
                std::atomic<int> __frontbuffer;
#ifdef __semaphore_back_buffered
                std::atomic<int> __backbuffer;
#endif

                template <class T>
                friend void __atomic_notify_semaphore(atomic<T> const* a, buffered_semaphore* s);

                template <class T, class V, class Fun>
                friend bool __atomic_wait_semaphore(atomic<T> const* a, V oldval, buffered_semaphore* s, memory_order order, Fun fun);
            };

            //typedef __counting_semaphore counting_semaphore;
            typedef buffered_semaphore counting_semaphore;

            template <class Sem, class Count>
            inline void semaphore_release_count_explicit(Sem* f, Count count, memory_order order) noexcept {
                f->release(count, order);
            }
            template <class Sem, class Count>
            inline void semaphore_release_count(Sem* f, Count count, memory_order order = memory_order_seq_cst) noexcept {
                semaphore_release_count_explicit(f, count, order);
            }
            template <class Sem>
            inline void semaphore_release_explicit(Sem* f, memory_order order) noexcept {
                f->release(order);
            }
            template <class Sem>
            inline void semaphore_release(Sem* f, memory_order order = memory_order_seq_cst) noexcept {
                semaphore_release_explicit(f, order);
            }
            template <class Sem>
            inline void semaphore_acquire_explicit(Sem* f, memory_order order) noexcept {
                f->acquire(order);
            }
            template <class Sem>
            inline void semaphore_acquire(Sem* f, memory_order order = memory_order_seq_cst) noexcept {
                semaphore_acquire_explicit(f, order);
            }
            template <class Sem>
            inline bool semaphore_try_acquire_explicit(Sem* f, memory_order order) noexcept {
                return f->try_acquire(order);
            }
            template <class Sem>
            inline bool semaphore_try_acquire(Sem* f, memory_order order = memory_order_seq_cst) noexcept {
                return semaphore_try_acquire_explicit(f, order);
            }
            template <class Sem, class Rep, class Period>
            inline void semaphore_acquire_for(Sem* f, std::chrono::duration<Rep, Period> const& rel_time, memory_order order = memory_order_seq_cst) noexcept {
                f->acquire_for(rel_time, order);
            }
            template <class Sem, class Clock, class Duration>
            inline void semaphore_acquire_until(Sem* f, std::chrono::time_point<Clock, Duration> const& abs_time, memory_order order = memory_order_seq_cst) noexcept {
                f->acquire_until(abs_time, order);
            }

            template <class T>
            void __atomic_notify_semaphore(atomic<T> const* a, counting_semaphore* s) {

                if (__semaphore_expect(!s->__reversebuffer.load(memory_order_relaxed), 1))
                    return;
                atomic_thread_fence(std::memory_order_seq_cst);
                int const waiting = s->__reversebuffer.exchange(0, memory_order_relaxed);
                if (__semaphore_expect(waiting, 0))
                    s->release(waiting, memory_order_release);
            }
            
            template <class T>
            void atomic_notify_semaphore(atomic<T> const* a, counting_semaphore* s) {

                __atomic_notify_semaphore(a, s);
            }

            template <class T, class V, class Fun>
            bool __atomic_wait_semaphore(atomic<T> const* a, V oldval, counting_semaphore* s, memory_order order, Fun fun) {

                for (int i = 0; i < 128; ++i, __semaphore_yield())
                    if (__semaphore_expect(a->load(order) != oldval, 1))
                        return true;
                do {
                    s->__reversebuffer.fetch_add(1, memory_order_relaxed);
                    atomic_thread_fence(memory_order_seq_cst);
                    if (__semaphore_expect(a->load(order) != oldval, 0)) {
                        int const waiting = s->__reversebuffer.exchange(0, memory_order_relaxed);
                        switch (waiting) {
                        case 0:  s->acquire(memory_order_relaxed); // uuuuuuuuhhhh, this is really weird for for/until
                        case 1:  break;
                        default: s->release(waiting - 1, memory_order_relaxed);
                        }
                        return true;
                    }
                    if(!fun()) return false;
                } while (a->load(order) != oldval);
                return false;
            }

            template <class T, class V>
            void atomic_wait_semaphore_explicit(std::atomic<T> const* a, V oldval, counting_semaphore* s, std::memory_order order) {
                auto const fun = [=]() -> bool { s->acquire(memory_order_relaxed); return true; };
                __atomic_wait_semaphore(a, oldval, s, order, fun);
            }
            template <class T, class V>
            void atomic_wait_semaphore(std::atomic<T> const* a, V oldval, counting_semaphore* s, std::memory_order order = std::memory_order_seq_cst) {
                atomic_wait_semaphore_explicit(a, oldval, s, order);
            }
            template <class T, class V, class Clock, class Duration>
            bool atomic_wait_semaphore_until(std::atomic<T> const* a, V oldval, std::chrono::time_point<Clock, Duration> const& abs_time, counting_semaphore* s, std::memory_order order = std::memory_order_seq_cst) {
                auto const fun = [&]() -> bool { return s->acquire_until(abs_time, memory_order_relaxed); };
                return __atomic_wait_semaphore(a, oldval, s, order, fun);
            }
            template <class T, class V, class Rep, class Period>
            bool atomic_wait_semaphore_for(std::atomic<T> const* a, V oldval, std::chrono::duration<Rep, Period> const& rel_time, counting_semaphore* s, std::memory_order order = std::memory_order_seq_cst) {
                return atomic_wait_semaphore_until(a, oldval, __semaphore_clock::now() + rel_time, s, order);
            }

            struct alignas(64)               __atomic_wait_table_entry { counting_semaphore sem; };
            static constexpr int             __atomic_wait_table_size = 0x10;
            extern __atomic_wait_table_entry __atomic_wait_table[__atomic_wait_table_size];
            inline size_t                    __atomic_wait_table_index(void const* ptr) { return ((uintptr_t)ptr / 64) & 0xF; }

            template <class T>
            void atomic_notify(atomic<T> const* a) {
                atomic_notify_semaphore(a, &__atomic_wait_table[__atomic_wait_table_index(a)].sem);
            }
            template <class T, class V>
            void atomic_wait_explicit(std::atomic<T> const* a, V oldval, std::memory_order order) {
                atomic_wait_semaphore_explicit(a, oldval, &__atomic_wait_table[__atomic_wait_table_index(a)].sem, order);
            }
            template <class T, class V>
            void atomic_wait(std::atomic<T> const* a, V oldval, std::memory_order order = std::memory_order_seq_cst) {
                atomic_wait_explicit(a, oldval, &__atomic_wait_table[__atomic_wait_table_index(a)].sem, order);
            }
            template <class T, class V, class Rep, class Period>
            bool atomic_wait_for(std::atomic<T> const* a, V oldval, std::chrono::duration<Rep, Period> const& rel_time, std::memory_order order = std::memory_order_seq_cst) {
                return atomic_wait_semaphore_for(a, oldval, rel_time, &__atomic_wait_table[__atomic_wait_table_index(a)].sem, order);
            }
            template <class T, class V, class Clock, class Duration>
            bool atomic_wait_until(std::atomic<T> const* a, V oldval, std::chrono::time_point<Clock, Duration> const& abs_time, std::memory_order order = std::memory_order_seq_cst) {
                return atomic_wait_semaphore_until(a, oldval, abs_time, &__atomic_wait_table[__atomic_wait_table_index(a)].sem, order);
            }

        } // namespace concurrency_v2
    } // namespace experimental
} // namespace std

#ifdef __semaphore_old_min
    #define min __semaphore_old_min
#endif
#ifdef __semaphore_old_max
    #define max __semaphore_old_max
#endif

#endif //binary_semaphore_hpp
