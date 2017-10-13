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

  __semaphore_abi __condition_variable_atomic_impl_base();

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
