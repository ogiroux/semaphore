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

#ifndef TEST_HPP
#define TEST_HPP

#include <thread>
#include <mutex>

#include <atomic>
template <bool truly>
struct dumb_mutex {

    dumb_mutex() : locked(false) {
    }

    dumb_mutex(const dumb_mutex&) = delete;
    dumb_mutex& operator=(const dumb_mutex&) = delete;

    void lock() {

        while (1) {
            bool state = false;
            if (locked.compare_exchange_weak(state, true, std::memory_order_acquire))
                return;
            while (locked.load(std::memory_order_relaxed))
                if (!truly)
                    std::this_thread::yield();
        };
    }

    void unlock() {

        locked.store(false, std::memory_order_release);
    }

private:
    std::atomic<bool> locked;
};

#include "semaphore.hpp"
struct alignas(64) binary_semaphore_lock {

    void lock() {

        //while (__semaphore_expect(f.test_and_set(std::memory_order_acquire), 0))
        //    ;                                 //this is the C++17 version
        f.acquire(std::memory_order_acquire);   //this is the C++20 version maybe!
    }

    void unlock() {

        f.release(std::memory_order_release);
    }

private:
    std::experimental::binary_semaphore f;
};

#ifdef WIN32
#include <windows.h>
#include <synchapi.h>
struct srw_mutex {

    srw_mutex() {
        InitializeSRWLock(&_lock);
    }

    void lock() {
        AcquireSRWLockExclusive(&_lock);
    }
    void unlock() {
        ReleaseSRWLockExclusive(&_lock);
    }

private:
    SRWLOCK _lock;
};
#endif

#endif //TEST_HPP


struct latch {
    latch() = delete;
    latch(int c) : a{ c }, c{ c } { }
    latch(const latch&) = delete;
    latch& operator=(const latch&) = delete;
    void arrive() { 
        if (a.fetch_sub(1) == 1)
            f.release(c, std::memory_order_release);
    }
    void wait() { 
        f.acquire(std::memory_order_acquire);
    }
private:
    alignas(64) std::experimental::counting_semaphore f{ 0 };
    alignas(64) std::atomic<int>                      a{ 0 };
    int const                                         c;
};

struct __atomic_wait {

    alignas(64) std::atomic<int>                      f{ 0 };
    alignas(64) std::experimental::counting_semaphore s{ 0 };

} __atomic_wait_table[0xF];

template <class T>
void atomic_notify(std::atomic<T>& a) {

    auto& w = __atomic_wait_table[((uintptr_t)&a / sizeof(__atomic_wait)) & 0xF];
    if (__semaphore_expect(!w.f.load(std::memory_order_relaxed), 1))
        return;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    int const waiting = w.f.exchange(0, std::memory_order_relaxed);
    if (__semaphore_expect(waiting, 0))
        w.s.release(waiting, std::memory_order_release);
}

template <class T, class V>
void atomic_wait(std::atomic<T> const& a, V oldval, std::memory_order order = std::memory_order_seq_cst) {

    for (int i = 0; i < 128; ++i, std::experimental::__semaphore_yield())
        if (__semaphore_expect(a.load(order) != oldval, 1))
            return;
    auto& w = __atomic_wait_table[((uintptr_t)&a / sizeof(__atomic_wait)) & 0xF];
    do {
        w.f.fetch_add(1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (__semaphore_expect(a.load(order) != oldval, 0)) {
            int const waiting = w.f.exchange(0, std::memory_order_relaxed);
            switch (waiting) {
            case 0:  w.s.acquire(std::memory_order_relaxed);
            case 1:  break;
            default: w.s.release(waiting - 1, std::memory_order_relaxed);
            }
            return;
        }
        w.s.acquire(std::memory_order_acquire);
    } while (a.load(order) != oldval);
}

struct binary_semaphore_lock2 {

    void lock() {

        bool old = false;
        while (!f.compare_exchange_strong(old, true, std::memory_order_acquire)) {
            atomic_wait(f, old, std::memory_order_relaxed);
            old = false;
        }
    }

    void unlock() {

        f.store(false, std::memory_order_release);
        atomic_notify(f);
    }

private:
    alignas(64) std::atomic<bool> f{ 0 };
};

// 2 cache lines
struct barrier {
    barrier() = delete;
    barrier(int c) : o(c) { }
    barrier(const barrier&) = delete;
    barrier& operator=(const barrier&) = delete;
    void arrive_and_wait() {
        auto const p = e.load(std::memory_order_relaxed);
        if (a.fetch_add(1, std::memory_order_acq_rel) == o - 1) {
            a.store(0, std::memory_order_relaxed);
            e.store(p + 1, std::memory_order_release);
            atomic_notify(e);
        }
        else
            atomic_wait(e, p, std::memory_order_acquire);
    }
private:
    alignas(64) std::atomic<int> a{ 0 };
    alignas(64) std::atomic<int> e{ 0 };
    int const                    o;
};
