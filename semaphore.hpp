/*

Copyright (c) 2014, NVIDIA Corporation
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

#ifndef binary_semaphore_HPP
#define binary_semaphore_HPP

#include <atomic>
#include <chrono>
#include <thread>
#include <cassert>

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
#endif //__linux__

namespace std {
    namespace experimental {
        inline namespace concurrency_v2 {


                using __semaphore_clock = conditional<chrono::high_resolution_clock::is_steady,
                                          chrono::high_resolution_clock, 
                                          chrono::steady_clock>::type;

                using __semaphore_duration = chrono::microseconds;


#ifdef __linux__
    inline void __semaphore_yield() { sched_yield(); }
#else
    inline void __semaphore_yield() { this_thread::yield(); }
#endif

#if defined(__GNUC__)
    #define __semaphore_expect __builtin_expect
#else
    #define __semaphore_expect(c,e) (c)
#endif

#ifdef __linux__
    #define __semaphore_fast_path
#endif

#if defined(WIN32) && _WIN32_WINNT >= 0x0602
    #define __semaphore_fast_path
#endif

            enum class atomic_notify {
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
                bool __test_and_set_slow(__base_t old, memory_order order, atomic_notify notify) noexcept;
#endif

                bool __test_and_set(memory_order order, atomic_notify notify) noexcept {
                    __base_t old = 0;
                    bool const success = atom.compare_exchange_weak(old, __valubit, order, memory_order_relaxed);
                    bool retcode = (old & __valubit) == 1;
#ifdef __semaphore_fast_path
                    if (__semaphore_expect(!success && !retcode, 0))
                        retcode = __test_and_set_slow(old, order, notify);
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


                void release(memory_order order = memory_order_seq_cst, atomic_notify notify = atomic_notify::all) const noexcept;



#ifdef __semaphore_fast_path
                void __release_slow(__base_t old, memory_order order, atomic_notify notify) noexcept;
#endif                

                inline void release(memory_order order = memory_order_seq_cst, atomic_notify notify = atomic_notify::all) noexcept {

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

                template <class A>
                static bool __wait_fast(A& atom, memory_order order) noexcept {

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


                inline void acquire(memory_order order = memory_order_seq_cst, atomic_notify notify = atomic_notify::all) noexcept {

                    while (__semaphore_expect(__test_and_set(order, notify), 0)) {
                        bool const success = __wait_fast(atom, order);
                        if (__semaphore_expect(!success, 0))
                            __wait_slow(order);
                    }
                }

                bool __wait_slow_timed(chrono::time_point<__semaphore_clock, __semaphore_duration> const& abs_time, memory_order order) noexcept;

                template <class Clock, class Duration>
                bool acquire_until(chrono::time_point<Clock, Duration> const& abs_time, memory_order order = memory_order_seq_cst, atomic_notify notify = atomic_notify::all) {

                    while (__semaphore_expect(__test_and_set(order, notify), 0)) {
                        bool success = __wait_fast(atom, order);
                        if (__semaphore_expect(!success, 0))
                            success = __wait_slow_timed(order, abs_time);
                        if (__semaphore_expect(!success, 0))
                            return false;
                    }
                    return true;
                }
                template <class Rep, class Period>
                bool acquire_for(chrono::duration<Rep, Period> const& rel_time, memory_order order = memory_order_seq_cst, atomic_notify notify = atomic_notify::all) {

                    if(__semaphore_expect(__test_and_set(order, notify), 0))
                        return true;
                    else
                        return acquire_until(__semaphore_clock::now() + rel_time, order);
                }

                __binary_semaphore(__base_t init) noexcept : atom(init) { }
                __binary_semaphore() noexcept = default;
                __binary_semaphore(const __binary_semaphore&) = delete;
                __binary_semaphore& operator=(const __binary_semaphore&) = delete;

                mutable atomic<__base_t> atom;

            } binary_semaphore;

            inline void binary_semaphore_release(binary_semaphore* f) noexcept {
                f->release();
            }
            inline void binary_semaphore_release_explicit(binary_semaphore* f, memory_order order) noexcept {
                f->release(order);
            }
            inline void binary_semaphore_release_explicit_notify(binary_semaphore* f, memory_order order, atomic_notify notify) noexcept {
                f->release(order, notify);
            }

            inline void binary_semaphore_acquire(binary_semaphore* f) {
                f->acquire();
            }
            inline void binary_semaphore_acquire_explicit(binary_semaphore* f, memory_order order) {
                f->acquire(order);
            }
            inline void binary_semaphore_acquire_explicit_notify(binary_semaphore* f, memory_order order, atomic_notify notify) {
                f->acquire(order, notify);
            }

        } // namespace concurrency_v2
    } // namespace experimental
} // namespace std

#endif //binary_semaphore_HPP
