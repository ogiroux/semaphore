
__semaphore_abi inline void binary_semaphore::release()
{
    if(__stolen)
        __stolen = false;
    else
#ifdef __semaphore_cuda
        __tocket.fetch_add(1, std::memory_order_relaxed);
#else
        __tocket.store(__tocket.load(std::memory_order_relaxed)+1, std::memory_order_relaxed);
#endif
#ifdef __semaphore_fast_path
    count_type old = __valubit;
    if (__semaphore_expect(!__atom.compare_exchange_strong(old, 0, std::memory_order_acquire, std::memory_order_relaxed), 0))
        __release_slow(old);
#else
    __atom.fetch_and(~__valubit, std::memory_order_acquire);
#endif
}

__semaphore_abi inline bool binary_semaphore::try_acquire()
{
    count_type old = 0;
    for (int i = 0;i < 64; ++i) {
        if(__semaphore_expect(__atom.compare_exchange_strong(old = 0, __valubit, std::memory_order_acquire, std::memory_order_relaxed),1))
            return __stolen = true;
        for(; old != 0 && i < 64; ++i) {
            __semaphore_yield();
            old = __atom.load(std::memory_order_relaxed);
        }
    }
    return false;
}

__semaphore_abi inline void binary_semaphore::acquire()
{
    if (__semaphore_expect(try_acquire(), 1))
        return;
    __acquire_slow();
}

template <class Clock, class Duration>
__semaphore_abi bool binary_semaphore::try_acquire_until(const std::chrono::time_point<Clock, Duration>& abs_time)
{
    if (__semaphore_expect(try_acquire(), 1))
        return true;
    return __acquire_slow_timed(std::chrono::time_point_cast<details::__semaphore_duration,Clock,Duration>(abs_time));
}

template <class Rep, class Period>
__semaphore_abi bool binary_semaphore::try_acquire_for(const std::chrono::duration<Rep, Period>& rel_time)
{
    if (__semaphore_expect(try_acquire(), 1))
        return true;
    return __acquire_slow_timed(details::__semaphore_clock::now() + rel_time);
}

__semaphore_abi inline binary_semaphore::binary_semaphore(count_type desired) : __atom(desired ? 0 : 1), __ticket(0), __tocket(0), __stolen(false)
{
}

__semaphore_abi inline binary_semaphore::~binary_semaphore()
{
}
