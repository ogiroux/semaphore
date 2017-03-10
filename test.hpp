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
struct binary_semaphore_lock {
    void lock() {
        //while (__semaphore_expect(f.test_and_set(std::memory_order_acquire), 0))
        //    ;                                 //this is the C++17 version
        f.acquire(std::memory_order_acquire);   //this is the C++20 version maybe!
    }
    void unlock() {
        f.release(std::memory_order_release);
    }
private:
    alignas(64) std::experimental::binary_semaphore f{ true };
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

struct binary_semaphore_lock2 {

    void lock() {

        bool old = false;
        while (!f.compare_exchange_strong(old, true, std::memory_order_acquire)) {
            std::experimental::atomic_wait_explicit(&f, old, std::memory_order_relaxed);
            old = false;
        }
    }

    void unlock() {

        f.store(false, std::memory_order_release);
        std::experimental::atomic_notify(&f);
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
            std::experimental::atomic_notify(&e);
        }
        else
            std::experimental::atomic_wait_explicit(&e, p, std::memory_order_acquire);
    }
private:
    alignas(64) std::atomic<int> a{ 0 };
    alignas(64) std::atomic<int> e{ 0 };
    int const                    o;
};
