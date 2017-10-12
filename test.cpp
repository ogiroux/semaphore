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

#include "test_defs.cpp"

int main(int argc, char const* argv[]) {

#ifdef __test_wtf
    WTF::initializeThreading();
#endif

  static const std::string onlycpu_s = "--cpu", 
    onlygpu_s = "--gpu",
    onlyscenario_s = "--scenario", 
    onlylock_s = "--lock",
    device_s = "--device";

  for(int i = 1; i < argc; ++i) {
    if(argv[i] == onlygpu_s) onlygpu = std::stoi(argv[++i]);
    else if(argv[i] == onlycpu_s) onlycpu = std::stoi(argv[++i]);
    else if(argv[i] == device_s) dev = std::stoi(argv[++i]);
    else if(argv[i] == onlyscenario_s) onlyscenario = argv[++i];
    else if(argv[i] == onlylock_s) onlylock = argv[++i];
    else {
        std::cout << "ERROR, unknown argument: " << argv[i] << std::endl; 
        return -1;
    }
  }

  atomic<int> a = ATOMIC_VAR_INIT(1);

  a.load();
  a.store(1);
  a.exchange(1);
  a.fetch_add(0);
  a.fetch_or(0);
  a.fetch_and(1);
  a.fetch_xor(0);
  int old = 1;
  a.compare_exchange_strong(old, 1);
  a.compare_exchange_weak(old,1);

  volatile atomic<int> va = ATOMIC_VAR_INIT(1);

  va.load();
  va.store(1);
  va.exchange(1);
  va.fetch_add(0);
  va.fetch_or(0);
  va.fetch_and(1);
  va.fetch_xor(0);
  va.compare_exchange_strong(old, 1);
  va.compare_exchange_weak(old,1);

  atomic_notify_one(&va);
  atomic_notify_one(&a);
  atomic_notify_all(&va);
  atomic_notify_all(&a);

  atomic_wait(&va, 0);
  atomic_wait_explicit(&va, 0, std::memory_order_acquire);
  atomic_wait(&a, 0);
  atomic_wait_explicit(&a, 0, std::memory_order_acquire);

  condition_variable_atomic c;
  
  c.notify_one(va);
  c.notify_one(a);
  c.notify_all(va);
  c.notify_all(a);

  c.wait(va, 0);
  c.wait(a, 0);
  c.wait(va, 0, std::memory_order_acquire);
  c.wait(a, 0, std::memory_order_acquire);

  c.wait_until(va, 0, std::chrono::high_resolution_clock::now());
  c.wait_until(a, 0, std::chrono::high_resolution_clock::now());
  c.wait_until(va, 0, std::chrono::high_resolution_clock::now(), std::memory_order_acquire);
  c.wait_until(a, 0, std::chrono::high_resolution_clock::now(), std::memory_order_acquire);

  c.wait_for(va, 0, std::chrono::nanoseconds(1));
  c.wait_for(a, 0, std::chrono::nanoseconds(1));
  c.wait_for(va, 0, std::chrono::nanoseconds(1), std::memory_order_acquire);
  c.wait_for(a, 0, std::chrono::nanoseconds(1), std::memory_order_acquire);

  auto l = [] __test_abi (int v) -> bool { return v == 1; };

  c.wait(va, l);
  c.wait(a, l);
  c.wait(va, l, std::memory_order_acquire);
  c.wait(a, l, std::memory_order_acquire);

  c.wait_until(va, l, std::chrono::high_resolution_clock::now());
  c.wait_until(a, l, std::chrono::high_resolution_clock::now());
  c.wait_until(va, l, std::chrono::high_resolution_clock::now(), std::memory_order_acquire);
  c.wait_until(a, l, std::chrono::high_resolution_clock::now(), std::memory_order_acquire);

  c.wait_for(va, l, std::chrono::nanoseconds(1));
  c.wait_for(a, l, std::chrono::nanoseconds(1));
  c.wait_for(va, l, std::chrono::nanoseconds(1), std::memory_order_acquire);
  c.wait_for(a, l, std::chrono::nanoseconds(1), std::memory_order_acquire);

  binary_semaphore b(1);

  b.acquire();
  b.release();
  b.try_acquire();
  b.release();
  b.try_acquire_until(std::chrono::high_resolution_clock::now());
  b.release();
  b.try_acquire_for(std::chrono::nanoseconds(1));
  b.release();
  
  counting_semaphore cs(4);
  
  cs.acquire();
  cs.try_acquire();
  cs.try_acquire_until(std::chrono::high_resolution_clock::now());
  cs.try_acquire_for(std::chrono::nanoseconds(1));
  cs.release(3);
  cs.release();

  print_headers();

  run_calibration();

  uint32_t count = 0; 
  double product = 1.0;

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
  std::cout << "== total : " << std::fixed << std::setprecision(0) << 10000/std::pow(product, 1.0/count) << " lockmarks ==" << std::endl;
}
