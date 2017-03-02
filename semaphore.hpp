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
    template < class Rep, class Period>
    timespec __semaphore_to_timespec(chrono::duration<Rep, Period> const& delta) {
        struct timespec ts;
        ts.tv_sec = static_cast<long>(chrono::duration_cast<chrono::seconds>(delta).count());
        ts.tv_nsec = static_cast<long>(chrono::duration_cast<chrono::nanoseconds>(delta).count());
        return ts;
    }
    #define __semaphore_fast_path
#endif

#if defined(WIN32) && _WIN32_WINNT >= 0x0602
    #define __semaphore_fast_path
#endif

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
                            success = __wait_slow_timed(order, abs_time);
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

                __binary_semaphore(bool available) noexcept : atom(!available) { }
                __binary_semaphore() noexcept = default;
                __binary_semaphore(const __binary_semaphore&) = delete;
                __binary_semaphore& operator=(const __binary_semaphore&) = delete;

                mutable atomic<__base_t> atom{ false };

            } binary_semaphore;

            struct __counting_semaphore {

                typedef int __base_t;
                static constexpr int __base_t_bits = CHAR_BIT * sizeof(int);

#ifdef __semaphore_fast_path
                static constexpr __base_t __valumask = (1 << (__base_t_bits - 2)) - 1,
                                          __contmask = (1 << (__base_t_bits - 2)),
                                          __lockmask = (1 << (__base_t_bits - 1));
#else
                static constexpr __base_t __valumask = ~0,
                                          __contmask = 0,
                                          __lockmask = 0;
#endif
                static constexpr __base_t max = __valumask >> 1;
                static constexpr __base_t __bias = max + 1;
                static constexpr __base_t min = -__bias;

#ifdef __semaphore_fast_path
                template<class Pred>
                bool __fetch_sub_if_slow(Pred pred, __base_t term, __base_t old, memory_order order) noexcept {

                    assert(term >= 0 && term <= max);

                    do {
                        old &= ~__lockmask;
                        if (atom.compare_exchange_weak(old, old - term, order, memory_order_relaxed))
                            return true;
                    } while (pred((old & __valumask) - __bias));

                    return false;
                }
#endif
                template<class Pred>
                bool __fetch_sub_if(Pred pred, __base_t term, memory_order order) noexcept {

                    assert(term >= 0 && term <= max);

                    __base_t old = __bias+term, set = __bias+0;
                    bool retcode = atom.compare_exchange_weak(old, set, order, memory_order_relaxed);
                    if (__semaphore_expect(!retcode && pred((old & __valumask) - __bias), 0)) {
                        old &= __valumask;
                        set = old - term;
                        retcode = atom.compare_exchange_weak(old, set, order, memory_order_relaxed);
                    }
                    if (__semaphore_expect(!retcode && pred((old & __valumask) - __bias), 0))
                        retcode = __fetch_sub_if_slow(pred, term, old, order);
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
                void __fetch_add_slow(__base_t term, __base_t old, memory_order order, semaphore_notify notify) noexcept;
#endif                

                inline void release(__base_t term, memory_order order = memory_order_seq_cst, semaphore_notify notify = semaphore_notify::all) noexcept {

#ifdef __semaphore_fast_path
                    __base_t old = __bias+0, set = __bias+term;
                    bool success = atom.compare_exchange_weak(old, set, order, memory_order_relaxed);
                    while (__semaphore_expect(!success && !(old & (__contmask | __lockmask)), 0)) {
                        set = old + term;
                        success = atom.compare_exchange_weak(old, set, order, memory_order_relaxed);
                    }
                    assert((__bias + max - (old & __valumask)) >= term);
                    if (__semaphore_expect(!success, 0))
                        __fetch_add_slow(term, old, order, notify);
#else
                    old = atom.fetch_add(term, order);
#endif

#ifdef __semaphore_arm
                    __asm__ __volatile__(
                        "   dsb\n"
                        "   sev"
                    );
#endif
                }

                void __wait_slow(__base_t c, memory_order order) noexcept;

                template<class Pred>
                inline bool __wait_fast(Pred pred, memory_order order) noexcept {

                    auto value = (atom.load(order) & __valumask) - __bias;
                    if (__semaphore_expect(pred(value), 1))
                        return true;
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
                        value = (atom.load(order) & __valumask) - __bias;
                        if (__semaphore_expect(pred(value), 1)) {
                            atomic_thread_fence(order); 
                            return true; 
                        }
                    }
#endif
                    for (int i = 0; i < 32; ++i) {
                        __semaphore_yield();
                        value = (atom.load(order) & __valumask) - __bias;
                        if (__semaphore_expect(pred(value), 1)) return true;
                    }
                    return false;
                }

#ifdef __semaphore_fast_path
                void __wait_final(__base_t old) noexcept;
#endif

                template<class Pred>
                void __wait_slow(Pred pred, memory_order order) noexcept {

                    __base_t old;
                    __semaphore_exponential_backoff b;
#ifdef __semaphore_fast_path
                    for (int i = 0; i < 2; ++i) {
#else
                    while (1) {
#endif
                        b.sleep();
                        old = atom.load(order);
                        if (pred((old & __valumask) - __bias)) goto done;
                    }
#ifdef __semaphore_fast_path
                    while (1) {
                        old = atom.fetch_or(__contmask, memory_order_relaxed) | __contmask;
                        if (pred((old & __valumask) - __bias)) goto done;
                        __wait_final(old);
                        old = atom.load(order);
                        if (pred((old & __valumask) - __bias)) goto done;
                    }
#endif
                done:
#ifdef __semaphore_fast_path
                    while (old & __lockmask)
                        old = atom.load(memory_order_relaxed);
#endif
                }

                template<class Pred>
                inline void __wait_if(Pred pred, memory_order order = memory_order_seq_cst) noexcept {

                    bool const success = __wait_fast(pred, order);
                    if (__semaphore_expect(!success, 0))
                        __wait_slow(pred, order);
                }

                template<class Pred>
                inline void acquire_if(Pred pred, __base_t term, memory_order order = memory_order_seq_cst) noexcept {

                    while (__semaphore_expect(!__fetch_sub_if(pred, term, order), 0))
                        __wait_if(pred, order);
                }
                inline void acquire(__base_t term = 1, memory_order order = memory_order_seq_cst) noexcept {

                    acquire_if([=](__base_t value) -> bool { return value >= term; }, term, order);
                }

                bool __wait_slow_timed(__base_t c, chrono::time_point<__semaphore_clock, __semaphore_duration> const& abs_time, memory_order order) noexcept;

                template <class Clock, class Duration>
                bool acquire_until(chrono::time_point<Clock, Duration> const& abs_time, __base_t term = 1, memory_order order = memory_order_seq_cst) {

                    auto const pred = [=](__base_t value) -> bool { return value >= term; };
                    while (__semaphore_expect(!__fetch_sub_if(pred, term, order), 0)) {
                        bool success = __wait_fast(pred, order);
                        if (__semaphore_expect(!success, 0))
                            success = __wait_slow_timed(term, order, abs_time);
                        if (__semaphore_expect(!success, 0))
                            return false;
                    }
                    return true;
                }
                template <class Rep, class Period>
                bool acquire_for(chrono::duration<Rep, Period> const& rel_time, __base_t term = 1, memory_order order = memory_order_seq_cst) {

                    if (__semaphore_expect(__test_and_set(order), 0))
                        return true;
                    else
                        return acquire_until(__semaphore_clock::now() + rel_time, order);
                }

                __counting_semaphore(__base_t initial) noexcept : atom(initial+__bias) {
                    assert(initial >= min && initial <= max);
                }
                __counting_semaphore() noexcept = default;
                __counting_semaphore(const __counting_semaphore&) = delete;
                __counting_semaphore& operator=(const __counting_semaphore&) = delete;

                atomic<__base_t> atom{ __base_t() };

            };



/*

implementations:
* time backoff (binary semaphore Mac)
* futex / Win8+ (binary/counting semaphore Linux/Win8+)
* pthread_sem / Win8+ (counting semaphore Mac/Win7-)


release(mo = sc) or release(no = 1, mo = sc)
acquire(mo = sc)
acquire_until(at, mo = sc)
acquire_for(rt, mo = sc)

// notify/wait may be less efficient (contention management out-of-line) and not destructor-safe.

atomic_notify(a)
atomic_wait(a, old)

*/

/*
typedef LONG NTSTATUS;

typedef NTSTATUS(NTAPI *_NtQuerySemaphore)(
    HANDLE SemaphoreHandle,
    DWORD SemaphoreInformationClass, // Would be SEMAPHORE_INFORMATION_CLASS
    PVOID SemaphoreInformation,      // but this is to much to dump here
    ULONG SemaphoreInformationLength,
    PULONG ReturnLength OPTIONAL
    );

typedef struct _SEMAPHORE_BASIC_INFORMATION {
    ULONG CurrentCount;
    ULONG MaximumCount;
} SEMAPHORE_BASIC_INFORMATION;
*/

            struct buffered_semaphore {

                static constexpr int max_limit = (std::numeric_limits<int>::max)() >> 1;

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
                        auto const inc = min(-(old >> 1), term);
#ifdef WIN32
                        auto const result = ReleaseSemaphore(__semaphore, inc, NULL) == TRUE;
#else
                        __backbuffer.fetch_add(inc - 1);
                        auto const result = sem_post(&__semaphore) == 0;
#endif
                        assert(result);
                    }
                    __frontbuffer.fetch_sub(1);
                }

                inline void __wait_fast() noexcept {

                    if (__semaphore_expect(__frontbuffer.load(std::memory_order_relaxed) >> 1 > 0, 1))
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
                        if (__semaphore_expect(__frontbuffer.load(std::memory_order_relaxed) >> 1 > 0, 1))
                            return;
                    }
#endif
                    for (int i = 0; i < 32; ++i) {
                        __semaphore_yield();
                        if (__semaphore_expect(__frontbuffer.load(std::memory_order_relaxed) >> 1 > 0, 1))
                            return;
                    }
                }

                inline void acquire(std::memory_order order = std::memory_order_seq_cst) noexcept {

                    __wait_fast();
                    if (__frontbuffer.fetch_sub(2, order) >> 1 > 0)
                        return;
#ifdef WIN32
                    auto const result = WaitForSingleObject(__semaphore, INFINITE);
                    assert(result == WAIT_OBJECT_0);
#else
                    auto const result = sem_wait(&__semaphore);
                    assert(result == 0);
                    int value = __backbuffer.load(std::memory_order_relaxed);
                    while (value > 0)  // would it still stay positive?
                        if (__backbuffer.compare_exchange_weak(value, value - 1, order, std::memory_order_relaxed)) { // conserve buffer.
                            auto const result = sem_post(&__semaphore);
                            assert(result == 0);
                            break;
                        }
#endif
                }
                template <class Rep, class Period>
                bool acquire_for(std::chrono::duration<Rep, Period> const& rel_time, std::memory_order order = std::memory_order_seq_cst) {

                    __wait_fast();
                    if (__frontbuffer.fetch_sub(2, order) >> 1 > 0)
                        return;
#ifdef WIN32
                    auto const result = WaitForSingleObject(__semaphore, std::chrono::milliseconds(rel_time).count()) == WAIT_OBJECT_0;
#else
                    auto const timespec __semaphore_to_timespec(rel_time);
                    auto const result = sem_timedwait(&__semaphore, &timespec) == 0;
                    int value = __backbuffer.load(std::memory_order_relaxed);
                    while (value > 0)  // would it still stay positive?
                        if (__backbuffer.compare_exchange_weak(value, value - 1, order, std::memory_order_relaxed)) { // conserve buffer.
                            auto const result = sem_post(&__semaphore);
                            assert(result == 0);
                            break;
                        }
#endif
                    return result;
                }
                template <class Clock, class Duration>
                bool acquire_until(std::chrono::time_point<Clock, Duration> const& abs_time, std::memory_order order = std::memory_order_seq_cst) {
        
                    return acquire_for(abs_time - Clock::now(), term, order);
                }

                buffered_semaphore(int initial = 0) noexcept 
                    : __frontbuffer{ initial << 1 }
#ifndef WIN32
                    , __backbuffer{ 0 } 
#endif
                {
                    assert(initial >= 0 && initial <= max_limit);
#ifdef WIN32
                    __semaphore = CreateSemaphore(NULL, 0, max_limit, NULL);
                    assert(__semaphore != (HANDLE)ERROR_INVALID_HANDLE);
#else
                    auto result = sem_init(&__semaphore, 0, 0);
                    assert(result == 0);
#endif
                }
                ~buffered_semaphore() {

                    while (__frontbuffer.load(std::memory_order_acquire) & 1)
                        ;
#ifdef WIN32
                    CloseHandle(__semaphore);
#else
                    auto const result = sem_destroy(&__semaphore);
                    assert(result == 0);
#endif
                }
                buffered_semaphore(const buffered_semaphore&) = delete;
                buffered_semaphore& operator=(const buffered_semaphore&) = delete;

            private:

#ifdef WIN32
                 HANDLE          __semaphore;
#else
                sem_t            __semaphore;
                std::atomic<int> __backbuffer;
#endif
                std::atomic<int> __frontbuffer;
            };

            //typedef __counting_semaphore counting_semaphore;
            typedef buffered_semaphore counting_semaphore;

            template <class Sem>
            inline void semaphore_release(Sem* f) noexcept {
                f->release();
            }
            template <class Sem>
            inline void semaphore_release_explicit(Sem* f, memory_order order) noexcept {
                f->release(order);
            }
            template <class Sem>
            inline void semaphore_release_explicit_notify(Sem* f, memory_order order, semaphore_notify notify) noexcept {
                f->release(order, notify);
            }
            template <class Sem>
            inline void semaphore_release(Sem* f, typename Sem::__base_t term) noexcept {
                f->release(term);
            }
            template <class Sem>
            inline void semaphore_release_explicit(Sem* f, typename Sem::__base_t term, memory_order order) noexcept {
                f->release(term, order);
            }
            template <class Sem>
            inline void semaphore_release_explicit_notify(Sem* f, typename Sem::__base_t term, memory_order order, semaphore_notify notify) noexcept {
                f->release(term, order, notify);
            }

            template <class Sem>
            inline void semaphore_acquire(Sem* f) {
                f->acquire();
            }
            template <class Sem>
            inline void semaphore_acquire_explicit(Sem* f, memory_order order) {
                f->acquire(order);
            }
            template <class Sem>
            inline void semaphore_acquire(Sem* f, typename Sem::__base_t term) {
                f->acquire(term);
            }
            template <class Sem>
            inline void semaphore_acquire_explicit(Sem* f, typename Sem::__base_t term, memory_order order) {
                f->acquire(term, order);
            }

        } // namespace concurrency_v2
    } // namespace experimental
} // namespace std

#endif //binary_semaphore_HPP
