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

#include "harness.cpp"

#include "test.hpp"

struct atomic_flag_lock {

  void lock() {
     while(!flag.test_and_set(std::memory_order_acquire))
      ;
  }
  void unlock() {
     flag.clear(std::memory_order_release);
  }

  std::atomic_flag flag = ATOMIC_VAR_INIT(false);
};


template<class F>
int driver_main(int argc, char const* argv[], F && f) {

#ifdef __test_wtf
    WTF::initializeThreading();
#endif

    static const std::string onlycpu_s = "--cpu", 
    onlygpu_s = "--gpu",
    usehmm_s = "--hmm",
    onlyscenario_s = "--scenario", 
    onlylock_s = "--lock",
    device_s = "--device",
    onlyloop_s = "--loop";

    for(int i = 1; i < argc; ++i)
        if(argv[i] == onlygpu_s) onlygpu = std::stoi(argv[++i]);
        else if(argv[i] == onlycpu_s) onlycpu = std::stoi(argv[++i]);
        else if(argv[i] == onlyloop_s) onlyloop = std::stoi(argv[++i]);
        else if(argv[i] == device_s) dev = std::stoi(argv[++i]);
        else if(argv[i] == onlyscenario_s) onlyscenario = argv[++i];
        else if(argv[i] == onlylock_s) onlylock = argv[++i];
        else if(argv[i] == usehmm_s) use_malloc_managed = false;
        else {
            std::cout << "ERROR, unknown argument: " << argv[i] << std::endl; 
            return -1;
        }

    print_headers();

    run_calibration();

    uint32_t count = 0; 
    double product = 1.0;

    run_and_report_mutex_scenarios(atomic_flag_lock, count, product);

    run_and_report_mutex_scenarios(mutex, count, product);
    run_and_report_mutex_scenarios(binary_semaphore_mutex, count, product);
    if(!onlylock.empty()) {
#ifdef __test_wtf
        run_and_report_mutex_scenarios(WTF::Lock, count, product);
#endif
#ifdef HAS_UNFAIR_LOCK
        run_and_report_mutex_scenarios(unfair_lock, count, product);
#endif
        run_and_report_mutex_scenarios(counting_semaphore_mutex, count, product);
        run_and_report_mutex_scenarios(poor_mutex, count, product);
        run_and_report_mutex_scenarios(mutex, count, product);
    }

    f(count, product);

    std::cout << "== total : " << std::fixed << std::setprecision(0) << 10000/std::pow(product, 1.0/count) << " lockmarks ==" << std::endl;
    return 0;
}
