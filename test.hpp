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

struct alignas(64) test_mutex_ {

    __test_abi void lock() {

        bool state = false;
        if (__semaphore_expect(locked.compare_exchange_weak(state, true, std::memory_order_acquire), 1))
            return;
        for (int i = 0; i < 32; ++i) {
            details::__semaphore_yield();
            state = locked.load(std::memory_order_relaxed);
            if (__semaphore_expect(state == false, 1)) {
                if (__semaphore_expect(locked.compare_exchange_weak(state, true, std::memory_order_acquire), 1))
                    return;
                break;
            }
        }
        details::__semaphore_exponential_backoff b;
        while (!locked.compare_exchange_weak(state, true, std::memory_order_acquire))
            b.sleep();
    }

    __test_abi void unlock() {

        locked.store(false, std::memory_order_release);
    }

private:
    atomic<bool> locked{ false };
};


template<class T, class V>
__test_abi void test_atomic_wait_explicit(atomic<T> const* a, V old, std::memory_order order) {
    for (int i = 0; i < 16; ++i)
        if (__semaphore_expect(a->load(order) != old, 1))
            return;
    unsigned nanoseconds = 8000;
    while (a->load(order) == old) {
        nanoseconds += nanoseconds;
        if (nanoseconds > 256000)
            nanoseconds = 256000;
#if !defined(__CUDA_ARCH__)
        std::this_thread::sleep_for(std::chrono::nanoseconds(nanoseconds));
#elif defined(__has_cuda_nanosleep)
        asm volatile("nanosleep.u32 %0;" ::"r"(nanoseconds):);
#endif
    }
}

struct alignas(64) test_mutex {
    __test_abi void lock() {
        for (int i = 0; i < 16; ++i) {
            bool state = false;
            if (__semaphore_expect(_locked.compare_exchange_weak(state, true, std::memory_order_acquire), 1))
                return;
        }
        while (1) {
            bool state = false;
            if (__semaphore_expect(_locked.compare_exchange_weak(state, true, std::memory_order_acquire), 0))
                return;
            test_atomic_wait_explicit(&_locked, state, std::memory_order_relaxed);
        }
    }
    __test_abi void unlock() {
        _locked.store(false, std::memory_order_release);
    }
    atomic<bool> _locked{ false };
};


struct alignas(64) dumb_mutex {

    __test_abi void lock() {

        while (1) {
            bool state = false;
            if (locked.compare_exchange_weak(state, true, std::memory_order_acquire))
                return;
            while (locked.load(std::memory_order_relaxed))
                ;
        };
    }

    __test_abi void unlock() {

        locked.store(false, std::memory_order_release);
    }

private:
    atomic<bool> locked{ false };
};

struct alignas(64) binary_semaphore_lock {
    __test_abi void lock() {
        f.acquire(std::memory_order_acquire);
    }
    __test_abi void unlock() {
        f.release(std::memory_order_release);
    }
private:
    binary_semaphore f{ true };
};

struct alignas(64) counting_semaphore_lock {
    __test_abi void lock() {
        f.acquire(std::memory_order_acquire);
    }
    __test_abi void unlock() {
        f.release(std::memory_order_release);
    }
private:
    counting_semaphore f{ 1 };
};

struct latch {
    __test_abi latch() = delete;
    __test_abi latch(int c) : a{ c }, c{ c } { }
    __test_abi latch(const latch&) = delete;
    __test_abi latch& operator=(const latch&) = delete;
    __test_abi void arrive() { 
        if (a.fetch_sub(1) == 1)
            f.release(c, std::memory_order_release);
    }
    __test_abi void wait() { 
        f.acquire(std::memory_order_acquire);
    }
private:
    alignas(64) counting_semaphore f{ 0 };
    alignas(64) atomic<int>                      a{ 0 };
    int const                                         c;
};

struct alignas(64) atomic_wait_lock {

    __test_abi void lock() {

        bool old = false;
        atomic_wait_cas_explicit(&f, old, true, std::memory_order_acquire);
//        while (!f.compare_exchange_strong(old, true, std::memory_order_acquire)) {
  //          atomic_wait_explicit(&f, old, std::memory_order_relaxed);
    //        old = false;
      //  }
    }

    __test_abi void unlock() {

        f.store(false, std::memory_order_release);
        atomic_signal(&f);
    }

private:
    alignas(64) atomic<bool> f{ 0 };
};

// 2 cache lines
struct alignas(64) barrier {

    __test_abi barrier() = delete;
    __test_abi constexpr barrier(int c) : __o{ c } { }
    __test_abi barrier(const barrier&) = delete;
    __test_abi barrier& operator=(const barrier&) = delete;
//    __test_abi barrier& operator=(const barrier&) volatile = delete;

    __test_abi void arrive_and_wait() {
        __wait(__arrive());
    }
    __test_abi void arrive_and_drop() {
        __d.fetch_add(1, std::memory_order_relaxed);
        __arrive();
    }

    alignas(64) atomic<int> __a{ 0 };
    alignas(64) atomic<int> __e{ 0 };
    alignas(64) atomic<int> __d{ 0 };
    int                     __o;

    __test_abi int __arrive() {
        auto const p = __e.load(std::memory_order_relaxed);
        if (__a.fetch_add(1, std::memory_order_acq_rel) == __o - 1) {
            __o -= __d.load(std::memory_order_relaxed);
            __d.store(0, std::memory_order_relaxed);
            __a.store(0, std::memory_order_relaxed);
            __e.store(p + 1, std::memory_order_release);
            atomic_signal(&__e);
            return -1;
        }
        return p;
    }
    __test_abi void __wait(int p) {
        if (p < 0)
            return;
        atomic_wait_explicit(&__e, p, std::memory_order_acquire);
        //while (__e.load(std::memory_order_acquire) == p)
            ;
    }
};

#ifdef __NVCC__

namespace cuda { namespace experimental { inline namespace v1 { 

namespace details {
    __test_abi inline void __yield()
    {
#if !defined(__CUDA_ARCH__)
        std::this_thread::yield();
#elif defined(__has_cuda_nanosleep)
        __mme_nanosleep(1);
#endif
    }
    struct __exponential_backoff
    {
        unsigned time = 64;
        __test_abi void reset() {
            time = 64;
        }
        __test_abi void sleep()
        {
#if !defined(__CUDA_ARCH__)
            std::this_thread::sleep_for(std::chrono::nanoseconds(time));
#elif defined(__has_cuda_nanosleep)
            __mme_nanosleep(time);
#endif
            time += 64 + (time >> 2);
            if (time > 128*64) 
                time = 128*64;
        }
    };
}

struct mutex
{
    __test_abi inline void unlock() noexcept {
        tocket.fetch_add(1, std::memory_order_relaxed);
        atom.fetch_and(~__valubit, std::memory_order_release);
    }

    __test_abi inline void lock(std::memory_order order = std::memory_order_seq_cst) noexcept {
        uint32_t old = 0;
        for (int i = 0; i < 64; ++i) {
            if(atom.compare_exchange_weak(old = 0, __valubit, order, std::memory_order_relaxed)) {
                ticket.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            for (; old != 0 && i < 64; ++i, old = atom.load(std::memory_order_relaxed))
                details::__yield();
        }
        __lock_slow(order);
    }

    __test_abi constexpr mutex() noexcept : atom(0), ticket(0), tocket(5) {
    }

    mutex(const mutex&) = delete;
    mutex &operator=(const mutex&) = delete;

private:
    static constexpr uint32_t __valubit = 1;
    static constexpr uint32_t __contbit = 2;

    __test_abi void __lock_slow(std::memory_order order) noexcept
    {
        details::__exponential_backoff b;
        auto old = atom.fetch_add(__contbit, std::memory_order_acquire);
        auto const tick = ticket.fetch_add(1, std::memory_order_relaxed);
        auto tock = tocket.load(std::memory_order_relaxed);
        auto const maxdiff = (std::numeric_limits<uint32_t>::max)() >> 1;
        auto ready = (tock >= tick || tick - tock > maxdiff);
        for (int i = 0; ; ++i) {
            if(i < 64)
                details::__yield();
            else 
                b.sleep();
            if(!ready) {
                tock = tocket.load(std::memory_order_relaxed);
                ready = (tock >= tick || tick - tock > maxdiff);
                if(ready)
                    b.reset();
                else
                    continue;
            }
            old = atom.load(std::memory_order_relaxed);
            while ((old & __valubit) == 0) {
                auto next = old - __contbit + __valubit;
                if (atom.compare_exchange_weak(old, next, order, std::memory_order_relaxed))
                    return;
            }
        }
    }

    mutable atomic<uint32_t> atom;
    mutable atomic<uint32_t> ticket;
    mutable atomic<uint32_t> tocket;
};

}}}

#endif //__NVCC__

#endif //TEST_HPP
