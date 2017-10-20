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

struct __binary_semaphore_impl_base {

    using __count_type = uint32_t; // see 33.7.2.1
    
    __semaphore_abi __binary_semaphore_impl_base(__count_type);
  
    static constexpr __count_type __valubit = 1;
    static constexpr __count_type __lockbit = 2;
    static constexpr __count_type __slowbit = 4;
    static constexpr __count_type __contbit = 8;
  
  #ifdef __semaphore_fast_path
      __semaphore_abi void __release_slow(__count_type old);
  #endif
  
      __semaphore_abi void __acquire_slow();
  
      bool __acquire_slow_timed(std::chrono::nanoseconds const&);
  
    atomic<__count_type> __atom;
    atomic<__count_type> __ticket;
    atomic<__count_type> __tocket;
    bool __stolen;    
};

struct __counting_semaphore_impl_base {

#ifndef __semaphore_sem
  using __count_type = uint32_t; // see 33.7.3.1
#else //__semaphore_sem
  using __count_type = int32_t; // see 33.7.3.1
#endif //__semaphore_sem

  __semaphore_abi __counting_semaphore_impl_base(__count_type);

#ifdef __semaphore_fast_path
  static constexpr __count_type __valumask = ~3,
                                __contmask = 2,
                                __lockmask = 1,
                                __shift = 2;
#else //__semaphore_fast_path
  static constexpr __count_type __valumask = ~0,
                                __contmask = 0,
                                __lockmask = 0,
                                __shift = 0;
#endif //__semaphore_fast_path

#ifndef __semaphore_sem
  __semaphore_abi bool __fetch_sub_if_slow(__count_type old);
  __semaphore_abi bool __fetch_sub_if();
#ifdef __semaphore_fast_path
  __semaphore_abi void __fetch_add_slow(__count_type term, __count_type old);
#endif //__semaphore_fast_path

  __semaphore_abi void __acquire_slow();
  bool __acquire_slow_timed(std::chrono::nanoseconds const&);
  __semaphore_abi inline bool __acquire_fast();

  atomic<__count_type> atom;

#else //__semaphore_sem

  __semaphore_abi inline void __acquire_fast();
  __semaphore_abi inline void __acquire_slow();
  inline void __acquire_slow_timed(std::chrono::nanoseconds const&);
  __semaphore_abi inline void __backfill();
  __semaphore_sem_t __semaphore;
  atomic<__count_type> __frontbuffer;
#ifdef __semaphore_back_buffered
  atomic<__count_type> __backbuffer;
#endif //__semaphore_back_buffered

#endif //__semaphore_sem

  friend class condition_variable_atomic;
};

template<class Semaphore>
struct alignas(64) __condition_variable_atomic_impl_base {

  __host__ __device__ __condition_variable_atomic_impl_base();

  __semaphore_abi void __notify();
  template <class A, class Predicate>
    __semaphore_abi void __wait(A& object, Predicate pred, std::memory_order order);
  template <class A, class Predicate, class Clock, class Duration>
    bool __wait_until(A& object, Predicate pred, std::chrono::time_point<Clock, Duration> const& abs_time, std::memory_order order);
  template <class A, class Predicate, class Rep, class Period>
    bool __wait_for(A& object, Predicate pred, std::chrono::duration<Rep, Period> const& rel_time, std::memory_order order);

#ifndef __semaphore_cuda
  Semaphore                     __sem;
  atomic<typename Semaphore::count_type> __reversebuffer{ 0 };
#endif
};
