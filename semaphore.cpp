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

#include "semaphore.hpp"

namespace std {
    namespace experimental {
        inline namespace concurrency_v2 {

#ifdef __linux__
            // On Linux, we make use of the kernel memory wait operations. These have been available for a long time.
            template <class A>
            inline const void* __semaphore_fixalign(A& a) {
                static_assert(sizeof(A) <= sizeof(int), "Linux only supports 'int' for Futex.");
                return (const void*)((intptr_t)&a & ~(sizeof(int) - 1));
            }
            inline int __semaphore_readint(const void* p) {
                int i;
                memcpy(&i, p, sizeof(int));
                return i;
            }
            template <class A, class V>
            inline void __semaphore_wait(A& a, V v) {
                auto p = __semaphore_fixalign(a);
                auto i = __semaphore_readint(p);
                asm volatile("" ::: "memory");
                if (a.load(memory_order_relaxed) != v) return;
                syscall(SYS_futex, p, FUTEX_WAIT_PRIVATE, i, 0, 0, 0);
            }
            template <class A, class V, class Rep, class Period>
            void __semaphore_wait_timed(A& a, V v, const chrono::duration<Rep, Period>& t) {
                auto p = __semaphore_fixalign(a);
                auto i = __semaphore_readint(p);
                asm volatile("" ::: "memory");
                if (a.load(memory_order_relaxed) != v) return;
                syscall(SYS_futex, p, FUTEX_WAIT_PRIVATE, i, __semaphore_to_timespec(t), 0, 0);
            }
            template <class A>
            inline void __semaphore_wake_one(A& a) {
                syscall(SYS_futex, __semaphore_fixalign(a), FUTEX_WAKE_PRIVATE, 1, 0, 0, 0);
            }
            template <class A>
            inline void __semaphore_wake_all(A& a) {
                syscall(SYS_futex, __semaphore_fixalign(a), FUTEX_WAKE_PRIVATE, INT_MAX, 0, 0, 0);
            }
            template <class A, class V>
            inline void __semaphore_wait(volatile A& a, V v) {
                auto p = __semaphore_fixalign(a);
                auto i = __semaphore_readint(p);
                asm volatile("" ::: "memory");
                if (a.load(memory_order_relaxed) != v) return;
                syscall(SYS_futex, p, FUTEX_WAIT, i, 0, 0, 0);
            }
            template <class A, class V, class Rep, class Period>
            void __semaphore_wait_timed(volatile A& a, V v, const chrono::duration<Rep, Period>& t) {
                auto p = __semaphore_fixalign(a);
                auto i = __semaphore_readint(p);
                asm volatile("" ::: "memory");
                if (a.load(memory_order_relaxed) != v) return;
                syscall(SYS_futex, p, FUTEX_WAIT, i, __semaphore_to_timespec(t), 0, 0);
            }
            template <class A>
            inline void __semaphore_wake_one(volatile A& a) {
                syscall(SYS_futex, __semaphore_fixalign(a), FUTEX_WAKE, 1, 0, 0, 0);
            }
            template <class A>
            inline void __semaphore_wake_all(volatile A& a) {
                syscall(SYS_futex, __semaphore_fixalign(a), FUTEX_WAKE, INT_MAX, 0, 0, 0);
            }
#endif // __linux__

#if defined(WIN32) && _WIN32_WINNT >= 0x0602
            // On Windows, we make use of the kernel memory wait operations as well. These first became available with Windows 8.
            template <class A, class V>
            void __semaphore_wait(A& a, V v) {
                static_assert(sizeof(V) <= 8, "Windows only allows sizes between 1B and 8B for WaitOnAddress.");
                WaitOnAddress((PVOID)&a, (PVOID)&v, sizeof(v), -1);
            }
            template <class A, class V, class Rep, class Period>
            void __semaphore_wait_timed(A& a, V v, chrono::duration<Rep, Period> const& delta) {
                static_assert(sizeof(V) <= 8, "Windows only allows sizes between 1B and 8B for WaitOnAddress.");
                WaitOnAddress((PVOID)&a, (PVOID)&v, sizeof(v), (DWORD)chrono::duration_cast<chrono::milliseconds>(delta).count());
            }
            template <class A>
            inline void __semaphore_wake_one(A& a) {
                WakeByAddressSingle((PVOID)&a);
            }
            template <class A>
            inline void __semaphore_wake_all(A& a) {
                WakeByAddressAll((PVOID)&a);
            }
#endif // defined(WIN32) && _WIN32_WINNT >= 0x0602

#ifdef __semaphore_fast_path
            bool binary_semaphore::__test_and_set_slow(binary_semaphore::__base_t old, memory_order order) noexcept {
                while ((old & __valubit) == 0) {
                    old &= ~__lockbit;
                    if (atom.compare_exchange_weak(old, old + __valubit, order, memory_order_relaxed))
                        return false;
                }
                return true;
            }
#endif

#ifdef __semaphore_fast_path
            void binary_semaphore::__release_slow(binary_semaphore::__base_t old, memory_order order, semaphore_notify notify) noexcept {
                while (1) {
                    old &= (__contbit | __valubit);
                    __base_t const lock = (old & __contbit) ? __lockbit : 0;
                    if (atom.compare_exchange_weak(old, lock, order, memory_order_relaxed)) {
                        if (lock) {
                            switch (notify) {
                            case semaphore_notify::all: __semaphore_wake_all(atom); break;
                            case semaphore_notify::one: __semaphore_wake_one(atom); break;
                            case semaphore_notify::none: break;
                            }
                            atom.fetch_and(~lock, memory_order_relaxed);
                        }
                        break;
                    }
                }
            }
#endif                
            void binary_semaphore::__wait_slow(memory_order order) noexcept {

                __base_t old = atom.load(order);
                __base_t const expectbit = 0; //(set ? __valubit : 0);
                if ((old & __valubit) != expectbit) {
                    __semaphore_exponential_backoff b;
#ifdef __semaphore_fast_path
                    for (int i = 0; i < 2; ++i) {
#else
                    while (1) {
#endif
                        b.sleep();
                        old = atom.load(order);
                        if ((old & __valubit) == expectbit) break;
                    }
                }
#ifdef __semaphore_fast_path
                if ((old & __valubit) != expectbit) {
                    while (1) {
                        old = atom.fetch_or(__contbit, memory_order_relaxed) | __contbit;
                        if ((old & __valubit) == expectbit) break;
                        __semaphore_wait(atom, old);
                        old = atom.load(order);
                        if ((old & __valubit) == expectbit) break;
                    }
                }
                while (old & __lockbit)
                    old = atom.load(memory_order_relaxed);
#endif
            }

            bool binary_semaphore::__wait_slow_timed(chrono::time_point<__semaphore_clock, __semaphore_duration> const& abs_time, memory_order order) noexcept {

                __base_t old = atom.load(order);
                __base_t const expectbit = 0; //(set ? __valubit : 0);
                if ((old & __valubit) != expectbit) {
                    __semaphore_exponential_backoff b;
#ifdef __semaphore_fast_path
                    for (int i = 0; i < 2; ++i) {
#else
                    while (1) {
#endif
                        if (__semaphore_clock::now() > abs_time)
                            return false;
                        b.sleep();
                        old = atom.load(order);
                        if ((old & __valubit) == expectbit)
                            break;
                    }
                }
#ifdef __semaphore_fast_path
                if ((old & __valubit) != expectbit) {
                    while (1) {
                        old = atom.fetch_or(__contbit, memory_order_relaxed) | __contbit;
                        if ((old & __valubit) == expectbit)
                            break;
                        auto const delay = abs_time - __semaphore_clock::now();
                        if (delay < __semaphore_duration(0))
                            return false;
                        __semaphore_wait_timed(atom, old, delay);
                        old = atom.load(order);
                        if ((old & __valubit) == expectbit)
                            break;
                    }
                }
                while (old & __lockbit)
                    old = atom.load(memory_order_relaxed);
#endif
                return true;
            }

            bool __counting_semaphore::__fetch_sub_if_slow(int old, memory_order order) noexcept {

                do {
                    old &= ~__lockmask;
                    if (atom.compare_exchange_weak(old, old - (1 << __shift), order, memory_order_relaxed))
                        return true;
                } while ((old >> __shift) >= 1);

                return false;
            }

#ifdef __semaphore_fast_path
            void __counting_semaphore::__fetch_add_slow(int term, int old, memory_order order, semaphore_notify notify) noexcept {

                assert(term >= 0 && term <= max());

                while (1) {

                    bool const apply_lock = ((old & __contmask) != 0) && (notify != semaphore_notify::none);
                    int const set = ((old & __valumask) + (term<<__shift)) | (apply_lock ? __lockmask : 0);

                    old &= ~__lockmask;
                    if (atom.compare_exchange_weak(old, set, order, memory_order_relaxed)) {
                        if (apply_lock) {
                            switch (notify) {
                            case semaphore_notify::all: __semaphore_wake_all(atom); break;
                            case semaphore_notify::one: __semaphore_wake_one(atom); break;
                            case semaphore_notify::none: break;
                            }
                            atom.fetch_and(~__lockmask, memory_order_relaxed);
                        }
                        break;
                    }
                }
            }

            void __counting_semaphore::__wait_slow(memory_order order) noexcept {

                int old;
                __semaphore_exponential_backoff b;
#ifdef __semaphore_fast_path
                for (int i = 0; i < 2; ++i) {
#else
                while (1) {
#endif
                    b.sleep();
                    old = atom.load(order);
                    if ((old >> __shift) >= 1) goto done;
                }
#ifdef __semaphore_fast_path
                while (1) {
                    old = atom.fetch_or(__contmask, memory_order_relaxed) | __contmask;
                    if ((old >> __shift) >= 1) goto done;
                    __semaphore_wait(atom, old);
                    old = atom.load(order);
                    if ((old >> __shift) >= 1) goto done;
                }
#endif
            done:
#ifdef __semaphore_fast_path
                while (old & __lockmask)
                    old = atom.load(memory_order_relaxed);
#else
                ;
#endif
            }

#endif                

            __atomic_wait_table_entry __atomic_wait_table[__atomic_wait_table_size];

        } // namespace concurrency_v2
    } // namespace experimental
} // namespace std
