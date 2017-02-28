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
    latch(int c) : a(c) { }
    latch(const latch&) = delete;
    latch& operator=(const latch&) = delete;
    void arrive() { 
        if (a.fetch_add(-1) == 1) 
            f.release();
    }
    void wait() { 
        f.acquire();
    }
private:
    std::atomic<int> a;
    std::experimental::binary_semaphore f;
};
/*
struct barrier {
    barrier() = delete;
    barrier(int c) : o(c) { }
    barrier(const barrier&) = delete;
    barrier& operator=(const barrier&) = delete;
    void arrive_and_wait() { 
        auto const epoch = f.test();
        if (a.fetch_add(1) == o - 1) {
            a.store(0);
            f.set(!epoch);
        }
        else
            f.acquire(!epoch);
    }
private:
    int const o;
    std::atomic<int> a{0};
    std::experimental::binary_semaphore f;
};
*/
template <class T, class V>
void atomic_notify(std::experimental::binary_semaphore& s,
                   std::atomic<bool>& f,
                   std::atomic<T>& a, 
                   V newval, 
                   std::memory_order order = std::memory_order_seq_cst, //this implementation can't actually make use of this operand
                   std::experimental::atomic_notify notify = std::experimental::atomic_notify::all) {

    a.store(newval); //requires sc
    if (__semaphore_expect(f.load(),0)) { //requires sc
        f.store(false, std::memory_order_relaxed);
        s.release(std::memory_order_release, notify);
    }
}

template <class T, class V>
void atomic_wait(std::experimental::binary_semaphore& s,
                 std::atomic<bool>& f,
                 std::atomic<T> const& a, 
                 V current, 
                 std::memory_order order = std::memory_order_seq_cst) {

    if (__semaphore_expect(a.load(order) == current, 1))
        return;
    for (int i = 0; i < 32; ++i, std::experimental::__semaphore_yield())
        if (a.load(order) != current)
            return;
    while (1) {
        f.store(true, std::memory_order_relaxed);
        s.acquire(std::memory_order_seq_cst, std::experimental::atomic_notify::none); //requires sc
        if (a.load() != current) //requires sc
            return;
    }
}

struct alignas(64) binary_semaphore_lock2 {

    void lock() {

        int old = 0;
        while (!i.compare_exchange_strong(old, 1, std::memory_order_acquire)) {
            atomic_wait(s, f, i, old, std::memory_order_relaxed);
            old = 0;
        }
    }

    void unlock() {

        atomic_notify(s, f, i, 0, std::memory_order_release);
    }

private:
    std::atomic<int> i{0};
    std::atomic<bool> f{false};
    std::experimental::binary_semaphore s;
};
