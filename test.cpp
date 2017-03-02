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

#ifdef WIN32
#define _WIN32_WINNT 0x0602
#endif

#include "test.hpp"

#include <map>
#include <string>
#include <atomic>
#include <random>
#include <chrono>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <algorithm>

static int measure_count = 1 << 28;
static double time_target_in_seconds = 5;

#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#include <sys/times.h>
typedef tms cpu_time;
cpu_time get_cpu_time() {
    cpu_time t;
    times(&t);
    return t;
}
double user_time_consumed(cpu_time start, cpu_time end) {
    auto nanoseconds_per_clock_tick = double(1000000000) / sysconf(_SC_CLK_TCK);
    auto clock_ticks_elapsed = end.tms_utime - start.tms_utime;
    return clock_ticks_elapsed * nanoseconds_per_clock_tick;
}
double system_time_consumed(cpu_time start, cpu_time end) {
    auto nanoseconds_per_clock_tick = double(1000000000) / sysconf(_SC_CLK_TCK);
    auto clock_ticks_elapsed = end.tms_stime - start.tms_stime;
    return clock_ticks_elapsed * nanoseconds_per_clock_tick;
}
#endif

#ifdef __linux__
#include <sched.h>
void set_affinity(std::uint64_t cpu) {

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    cpu %= sizeof(int) * 8;
    CPU_SET(cpu, &cpuset);

    sched_setaffinity(0, sizeof(cpuset), &cpuset);
}
#endif

#ifdef __APPLE__
#include <mach/thread_policy.h>
#include <pthread.h>

extern "C" kern_return_t thread_policy_set(thread_t                thread,
    thread_policy_flavor_t  flavor,
    thread_policy_t         policy_info,
    mach_msg_type_number_t  count);

void set_affinity(std::uint64_t cpu) {

    cpu %= sizeof(integer_t) * 8;
    integer_t count = (1 << cpu);
    thread_policy_set(pthread_mach_thread_np(pthread_self()), THREAD_AFFINITY_POLICY, (thread_policy_t)&count, 1);
}
#endif

#ifdef WIN32
static HANDLE self = GetCurrentProcess();
typedef std::pair<FILETIME, FILETIME> cpu_time;
cpu_time get_cpu_time() {
    cpu_time t;
    FILETIME ftime, fsys, fuser;
    GetProcessTimes(self, &ftime, &ftime, &fsys, &fuser);
    memcpy(&t.first, &fsys, sizeof(FILETIME));
    memcpy(&t.second, &fuser, sizeof(FILETIME));
    return t;
}
std::uint64_t make64(std::uint64_t low, std::uint64_t high) {
    return low | (high << 32);
}
std::uint64_t make64(FILETIME ftime) {
    return make64(ftime.dwLowDateTime, ftime.dwHighDateTime);
}
double user_time_consumed(cpu_time start, cpu_time end) {

    double nanoseconds_per_clock_tick = 100; //100-nanosecond intervals
    auto clock_ticks_elapsed = make64(end.second) - make64(start.second);
    return clock_ticks_elapsed * nanoseconds_per_clock_tick;
}
double system_time_consumed(cpu_time start, cpu_time end) {

    double nanoseconds_per_clock_tick = 100; //100-nanosecond intervals
    auto clock_ticks_elapsed = make64(end.first) - make64(start.first);
    return clock_ticks_elapsed * nanoseconds_per_clock_tick;
}
void set_affinity(std::uint64_t cpu) {

    cpu %= sizeof(std::uint64_t) * 8;
    SetThreadAffinityMask(GetCurrentThread(), int(1 << cpu));
}
#endif

#ifdef EXTRA_LOCKS
// On Mac, you can build this in your Git/WebKit directory like so:
// xcrun clang++ -DEXTRA_LOCKS=1 -IPATH_TO_binary_semaphore/binary_semaphore -o LockSpeedTest2 Source/WTF/benchmarks/LockSpeedTest2.cpp -O3 -W -ISource/WTF -ISource/WTF/benchmarks -LWebKitBuild/Release -lWTF -framework Foundation -licucore -std=c++11 -fvisibility=hidden

#include "config.h"

//#include "ToyLocks.h"
#include <thread>
#include <unistd.h>
#include <wtf/CurrentTime.h>
#include <wtf/DataLog.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/ParkingLot.h>
#include <wtf/StdLibExtras.h>
#include <wtf/Threading.h>
#include <wtf/ThreadingPrimitives.h>
#include <wtf/Vector.h>
#include <wtf/WordLock.h>
#include <wtf/text/CString.h>
#endif

using my_clock = std::conditional<std::chrono::high_resolution_clock::is_steady,
    std::chrono::high_resolution_clock, std::chrono::steady_clock>::type;

template <class R>
double compute_work_item_cost(R r) {

    auto start = my_clock::now();

    //perform work
    for (int i = 0; i < measure_count; ++i) r.discard(1);

    auto end = my_clock::now();

    return std::chrono::nanoseconds(end - start).count() / double(measure_count);
}

class run {
    std::atomic<bool> go, stop;
    std::atomic<int> running, iterations;
public:
    run() : go(false), stop(false), running(0), iterations(0) { }
    struct report {
        double wall_time, user_time, system_time;
        std::uint64_t steps;
    };
    template <class F>
    report time(std::ostream& log, int threads, F f, std::uint64_t target_count) {
        std::random_device d;
        for (int i = 0; i < threads; ++i) {
            auto s = d();
            std::thread([&, f, i, s]() mutable {
                std::mt19937 r;
                r.seed(s);
                set_affinity(i);
                running++;
                while (go != true) std::this_thread::yield();
                while (stop != true) {
                    f(i, r);
                    iterations.fetch_add(1, std::memory_order_relaxed);
                }
                running--;
            }).detach();
        }
        while (running != threads) std::this_thread::yield();
        go = true;
        auto cpu_start = get_cpu_time();
        auto start = my_clock::now();
        std::uint64_t it1 = iterations;
        if (threads)
            std::this_thread::sleep_for(std::chrono::milliseconds(uint64_t(1000 * time_target_in_seconds)));
        else {
            std::mt19937 r;
            for (std::uint64_t i = 0; i < target_count; ++i) {
                f(0, r);
                iterations.fetch_add(1, std::memory_order_relaxed);
            }
        }
        std::uint64_t it2 = iterations;
        auto end = my_clock::now();
        auto cpu_end = get_cpu_time();
        log << "Done, canceling threads...\r" << std::flush;
        stop = true;
        while (running != 0) std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::seconds(1));

        report r;
        r.wall_time = double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        r.user_time = user_time_consumed(cpu_start, cpu_end);
        r.system_time = system_time_consumed(cpu_start, cpu_end);
        r.steps = (it2 - it1);

        return r;
    }
};

template <class F>
double do_run(std::ostream& csv, std::ostream& log, std::string const& what, int threads, F f, int count, double cost)
{
    log << "Measuring " << what << " (" << threads << "T, " << time_target_in_seconds
        << "s, step = " << count << " x " << cost << " ns)" << std::endl;

    auto expected = count * cost;
    auto r = run().time(log, threads, f, std::uint64_t(time_target_in_seconds * 1E9 / expected));
    log << std::right << std::setfill(' ') << std::setw(20) <<
        "total progress : " << r.steps << " steps in " << r.wall_time << "ns" << std::endl;

    auto wall_time = r.wall_time / r.steps;
    auto user_time = r.user_time / r.steps;
    auto system_time = r.system_time / r.steps;
    auto cpu_time = user_time + system_time;
    /*
    std::cout << std::right << std::setfill(' ') << std::setw(20) <<
    "expected : " << expected << " ns/step" << std::endl;

    std::cout << std::right << std::setfill(' ') << std::setw(20) <<
    "real : " << wall_time << " ns/step (" << wall_time / expected * 100 << "%)" << std::endl;

    std::cout << std::right << std::setfill(' ') << std::setw(20) <<
    "cpu : " << cpu_time << " ns/step (" << cpu_time / expected * 100 << "%)" << std::endl;
    */
    log << std::right << std::setfill(' ') << std::setw(20) <<
        "[user : " << user_time / cpu_time * 100 << "%]" << std::endl;
    log << std::right << std::setfill(' ') << std::setw(20) <<
        "[system : " << system_time / cpu_time * 100 << "%]" << std::endl;

    log << std::endl;

    csv << "\"" << what << "\"," << threads << ',' << r.steps << ',' << r.wall_time << ',' << expected << ','
        << wall_time / expected << ',' << cpu_time / expected << ',' << user_time / cpu_time << ',' << system_time / cpu_time << std::endl;

    return wall_time / count;
}

int main(int, const char *[]) {

    //    time_target_in_seconds = 0.5;

#ifdef EXTRA_LOCKS
    WTF::initializeThreading();
#endif

    std::ostringstream nullstream;
    std::ofstream csv("output.csv");
    if (!csv) {
        std::cout << "ERROR: could not open the output file." << std::endl;
        return 0;
    }

    csv << "name, threads, steps, time, expected, real/expected, cpu/expected, user/cpu, system/cpu" << std::endl;

    std::mt19937 r;
    std::mutex m1;
    typedef binary_semaphore_lock test_mutex_1;
    test_mutex_1 m2;

#ifdef EXTRA_LOCKS
#define HAS_LOCK_2
    typedef WTF::Lock test_mutex_2;
    //#else
    //    typedef webkit_mutex test_mutex_2;
#endif

#define HAS_LOCK_2
typedef binary_semaphore_lock2 test_mutex_2;

#ifdef HAS_LOCK_2
    test_mutex_2 m3;
#endif

    auto const N = std::thread::hardware_concurrency();
    assert(N <= 1024);

    // CONTROL FOR 1-THREAD

    std::cout << "Warming up...\r" << std::flush;
    compute_work_item_cost(r);
    auto cost = compute_work_item_cost(r);
    auto target_count = int(5E1 / cost);
    cost = do_run(nullstream, std::cout, "CONTROL run for 1-thread", 1, [=](int, std::mt19937&) mutable {
        for (int i = 0; i < target_count; ++i) r.discard(1);
    }, target_count, cost);
    std::cout << "Adjusting cost to " << cost << " ns/iteration (targeting " << target_count << " iterations/step).\n";
    std::cout << std::endl;

    // SINGLE THREADED

    auto std_single_threaded = do_run(csv, std::cout, "std::mutex single-threaded", 1, [=, &m1](int, std::mt19937&) mutable {
        { std::unique_lock<std::mutex>(m1); }
        for (int i = 0; i < target_count; ++i) r.discard(1);
    }, target_count, cost);
    auto ttas_single_threaded = do_run(csv, std::cout, "ttas_mutex single-threaded", 1, [=, &m2](int, std::mt19937&) mutable {
        { std::unique_lock<test_mutex_1>(m2); }
        for (int i = 0; i < target_count; ++i) r.discard(1);
    }, target_count, cost);
#ifdef HAS_LOCK_2
    std_single_threaded = do_run(csv, std::cout, "trial_mutex single-threaded", 1, [=, &m2](int, std::mt19937&) mutable {
        { std::unique_lock<test_mutex_2>(m3); }
        for (int i = 0; i < target_count; ++i) r.discard(1);
    }, target_count, cost);
#endif

    // CONTROL FOR N-THREAD

    auto cost_n = do_run(nullstream, std::cout, "CONTROL for uncontended N-thread", N, [=](int, std::mt19937&) mutable {
        for (int i = 0; i < target_count; ++i) r.discard(1);
    }, target_count, cost);
    if (cost_n < cost)
        std::cout << "NOTE: Based purely on these numbers, your system appears to have hyper-threads enabled.\n";
    cost = cost_n;
    std::cout << "Adjusting cost to " << cost << " ns/iteration (targeting " << target_count << " iterations/step).\n";
    std::cout << std::endl;

    //

    std::mutex m1N[1024];
    test_mutex_1 m2N[1024];
#ifdef HAS_LOCK_2
    test_mutex_2 m3N[1024];
#endif


    // NO CONTENTION

    auto std_no_contention = do_run(csv, std::cout, "std::mutex no contention", N, [=, &m1N](int i, std::mt19937&) mutable {
        auto& m = m1N[i];
        { std::unique_lock<std::mutex> l(m); }
        for (int i = 0; i < target_count; ++i) r.discard(1);
    }, target_count, cost);
    auto ttas_no_contention = do_run(csv, std::cout, "ttas_mutex no contention", N, [=, &m2N](int i, std::mt19937&) mutable {
        auto& m = m2N[i];
        { std::unique_lock<test_mutex_1> l(m); }
        for (int i = 0; i < target_count; ++i) r.discard(1);
    }, target_count, cost);
#ifdef HAS_LOCK_2
    std_no_contention = do_run(csv, std::cout, "trial_mutex no contention", N, [=, &m3N](int i, std::mt19937&) mutable {
        auto& m = m3N[i];
        { std::unique_lock<test_mutex_2> l(m); }
        for (int i = 0; i < target_count; ++i) r.discard(1);
    }, target_count, cost);
#endif

    // LOW-P CONTENTION

    {
        std::random_device d;
        if (d.entropy() == 0)
            std::cout << "NOTE: the system randomness source claims to have no entropy, low-p tests may not operate correctly." << std::endl;
        std::cout << std::endl;
    }
    auto mask = (4 << std::ilogb(N)) - 1;
    auto std_rare = do_run(csv, std::cout, "std::mutex rare contention", N, [=, &m1N](int, std::mt19937& dr) mutable {
        auto& m = m1N[dr() & mask];
        { std::unique_lock<std::mutex> l(m); }
        for (int i = 0; i < target_count; ++i) r.discard(1);
    }, target_count, cost);
    auto ttas_rare = do_run(csv, std::cout, "ttas_mutex rare contention", N, [=, &m2N](int, std::mt19937& dr) mutable {
        auto& m = m2N[dr() & mask];
        { std::unique_lock<test_mutex_1> l(m); }
        for (int i = 0; i < target_count; ++i) r.discard(1);
    }, target_count, cost);
#ifdef HAS_LOCK_2
    std_rare = do_run(csv, std::cout, "trial_mutex rare contention", N, [=, &m3N](int, std::mt19937& dr) mutable {
        auto& m = m3N[dr() & mask];
        { std::unique_lock<test_mutex_2> l(m); }
        for (int i = 0; i < target_count; ++i) r.discard(1);
    }, target_count, cost);
#endif

    // SHORTEST

    auto shortest_count = 1;
    auto std_shortest = do_run(csv, std::cout, "std::mutex shortest sections", N, [=, &m1](int, std::mt19937&) mutable {
        std::unique_lock<std::mutex> l(m1);
        r.discard(1);
    }, shortest_count, cost);
    auto ttas_shortest = do_run(csv, std::cout, "ttas_mutex shortest sections", N, [=, &m2](int, std::mt19937&) mutable {
        std::unique_lock<test_mutex_1> l(m2);
        r.discard(1);
    }, shortest_count, cost);
#ifdef HAS_LOCK_2
    std_shortest = do_run(csv, std::cout, "trial_mutex shortest sections", N, [=, &m3](int, std::mt19937&) mutable {
        std::unique_lock<test_mutex_2> l(m3);
        r.discard(1);
    }, shortest_count, cost);
#endif

    // SHORT

    auto short_count = int(2E2 / cost);
    auto std_short = do_run(csv, std::cout, "std::mutex short sections", N, [=, &m1](int, std::mt19937&) mutable {
        std::unique_lock<std::mutex> l(m1);
        for (int i = 0; i < short_count; ++i) r.discard(1);
    }, short_count, cost);
    auto ttas_short = do_run(csv, std::cout, "ttas_mutex short sections", N, [=, &m2](int, std::mt19937&) mutable {
        std::unique_lock<test_mutex_1> l(m2);
        for (int i = 0; i < short_count; ++i) r.discard(1);
    }, short_count, cost);
#ifdef HAS_LOCK_2
    std_short = do_run(csv, std::cout, "trial_mutex short sections", N, [=, &m3](int, std::mt19937&) mutable {
        std::unique_lock<test_mutex_2> l(m3);
        for (int i = 0; i < short_count; ++i) r.discard(1);
    }, short_count, cost);
#endif

    // LONG

    auto long_count = int(1E7 / cost);
    auto std_long = do_run(csv, std::cout, "std::mutex long sections", N, [=, &m1](int, std::mt19937&) mutable {
        std::unique_lock<std::mutex> l(m1);
        for (int i = 0; i < long_count; ++i) r.discard(1);
    }, long_count, cost);
    auto ttas_long = do_run(csv, std::cout, "ttas_mutex long sections", N, [=, &m2](int, std::mt19937&) mutable {
        std::unique_lock<test_mutex_1> l(m2);
        for (int i = 0; i < long_count; ++i) r.discard(1);
    }, long_count, cost);
#ifdef HAS_LOCK_2
    std_long = do_run(csv, std::cout, "trial_mutex long sections", N, [=, &m3](int, std::mt19937&) mutable {
        std::unique_lock<test_mutex_2> l(m3);
        for (int i = 0; i < long_count; ++i) r.discard(1);
    }, long_count, cost);
#endif

    //

    std::cout << "\n\n == REPORT == \n\n";
    std::cout << "single-thread, " << std_single_threaded / ttas_single_threaded << std::endl;
    std::cout << "uncontended, " << std_no_contention / ttas_no_contention << std::endl;
    std::cout << "rare, " << std_rare / ttas_rare << std::endl;
    std::cout << "shortest, " << std_shortest / ttas_shortest << std::endl;
    std::cout << "short, " << std_short / ttas_short << std::endl;
    std::cout << "long, " << std_long / ttas_long << std::endl;

    return 0;
}
