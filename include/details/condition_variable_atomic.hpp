
__semaphore_abi inline void condition_variable_atomic::__notify()
{
#ifndef __semaphore_cuda
    if (__semaphore_expect(!__reversebuffer.load(std::memory_order_relaxed), 1))
        return;
    (atomic_thread_fence)(std::memory_order_seq_cst);
    int const waiting = __reversebuffer.exchange(0, std::memory_order_relaxed);
    if (__semaphore_expect(waiting, 0))
        __sem.release(waiting);
#endif
}

template <class A, class Predicate>
__semaphore_abi void condition_variable_atomic::__wait(A& object, Predicate pred, std::memory_order order)
{
    for (int i = 0; i < 16; ++i, details::__semaphore_yield())
        if (__semaphore_expect(pred(object.load(order)), 1))
            return;
    details::__semaphore_exponential_backoff b;
    while(1)
    {
#ifndef __semaphore_cuda
        __reversebuffer.fetch_add(1, std::memory_order_relaxed);
        (atomic_thread_fence)(std::memory_order_seq_cst);
        if (__semaphore_expect(pred(object.load(order)), 0))
        {
            auto const waiting = __reversebuffer.exchange(0, std::memory_order_relaxed);
            switch (waiting)
            {
            case 0:
                __sem.acquire(); // uuuuuuuuhhhh, this is really weird for for/until
            case 1:
                break;
            default:
                __sem.release(waiting - 1); break;
            }
            return;
        }
        __sem.__acquire_slow();
#else
        b.sleep();
#endif
        if(pred(object.load(order)))
            return;
    }
}

template <class A, class Predicate, class Clock, class Duration>
__semaphore_abi bool condition_variable_atomic::__wait_until(A& object, Predicate pred, std::chrono::time_point<Clock, Duration> const& abs_time, std::memory_order order)
{
    for (int i = 0; i < 16; ++i, details::__semaphore_yield())
        if (__semaphore_expect(pred(object.load(order)), 1))
            return true;
    details::__semaphore_exponential_backoff b;
    while(1)
    {
        if(details::__semaphore_clock::now() > abs_time)
            return false;
#ifndef __semaphore_cuda
        __reversebuffer.fetch_add(1, std::memory_order_relaxed);
        (atomic_thread_fence)(std::memory_order_seq_cst);
        if (__semaphore_expect(pred(object.load(order)), 0))
        {
            auto const waiting = __reversebuffer.exchange(0, std::memory_order_relaxed);
            switch (waiting)
            {
            case 0:
                __sem.acquire(); // uuuuuuuuhhhh, this is really weird for for/until
            case 1:
                break;
            default:
                __sem.release(waiting - 1); break;
            }
            return true;
        }
        __sem.__acquire_slow_timed(abs_time);
#else
        b.sleep();
#endif
        if(pred(object.load(order)))
            return true;
    }
}

template <class A, class Predicate, class Rep, class Period>
__semaphore_abi bool condition_variable_atomic::__wait_for(A& object, Predicate pred, std::chrono::duration<Rep, Period> const& rel_time, std::memory_order order)
{
    return __wait_until(object, pred, details::__semaphore_clock::now() + rel_time, order);
}

__semaphore_abi inline condition_variable_atomic::condition_variable_atomic()
#ifndef __semaphore_cuda
    : __sem(0), __reversebuffer{0}
#endif
{
}

__semaphore_abi inline condition_variable_atomic::~condition_variable_atomic() 
{
}    

template <class T>
__semaphore_abi void condition_variable_atomic::notify_one(const volatile atomic<T>&) 
{
    __notify(); 
}

template <class T>
__semaphore_abi void condition_variable_atomic::notify_one(const atomic<T>&)
{
    __notify(); 
}

template <class T>
__semaphore_abi void condition_variable_atomic::notify_all(const volatile atomic<T>&)
{
    __notify(); 
}

template <class T>
__semaphore_abi void condition_variable_atomic::notify_all(const atomic<T>&)
{
    __notify(); 
}

template <class T>
__semaphore_abi void condition_variable_atomic::wait(const volatile atomic<T>& object, T old, std::memory_order order)
{ 
    __wait(object, [=] __semaphore_abi (T other) -> bool { return old != other; }, order); 
}

template <class T>
__semaphore_abi void condition_variable_atomic::wait(const atomic<T>& object, T old, std::memory_order order)
{ 
    __wait(object, [=] __semaphore_abi (T other) -> bool { return old != other; }, order); 
}

template <class T, class Predicate>
__semaphore_abi void condition_variable_atomic::wait(const volatile atomic<T>& object, Predicate pred, std::memory_order order)
{ 
    __wait(object, pred, order); 
}

template <class T, class Predicate>
__semaphore_abi void condition_variable_atomic::wait(const atomic<T>& object, Predicate pred, std::memory_order order)
{ 
    __wait(object, pred, order); 
}

template <class T, class Clock, class Duration>
__semaphore_abi bool condition_variable_atomic::wait_until(const volatile atomic<T>& object, T old, std::chrono::time_point<Clock, Duration> const& abs_time, std::memory_order order)
{ 
    return __wait_until(object, [=] __semaphore_abi (T other) -> bool { return old != other; }, abs_time, order); 
}

template <class T, class Clock, class Duration>
__semaphore_abi bool condition_variable_atomic::wait_until(const atomic<T>& object, T old, std::chrono::time_point<Clock, Duration> const& abs_time, std::memory_order order)
{ 
    return __wait_until(object, [=] __semaphore_abi (T other) -> bool { return old != other; }, abs_time, order); 
}

template <class T, class Predicate, class Clock, class Duration>
__semaphore_abi bool condition_variable_atomic::wait_until(const volatile atomic<T>& object, Predicate pred, std::chrono::time_point<Clock, Duration> const& abs_time, std::memory_order order)
{ 
    return __wait_until(object, pred, abs_time, order); 
}

template <class T, class Predicate, class Clock, class Duration>
__semaphore_abi bool condition_variable_atomic::wait_until(const atomic<T>& object, Predicate pred, std::chrono::time_point<Clock, Duration> const& abs_time, std::memory_order order)
{ 
    return __wait_until(object, pred, abs_time, order); 
}

template <class T, class Rep, class Period>
__semaphore_abi bool condition_variable_atomic::wait_for(const volatile atomic<T>& object, T old, std::chrono::duration<Rep, Period> const& rel_time, std::memory_order order)
{ 
    return __wait_for(object, [=] __semaphore_abi (T other) -> bool { return old != other; }, rel_time, order); 
}

template <class T, class Rep, class Period>
__semaphore_abi bool condition_variable_atomic::wait_for(const atomic<T>& object, T old, std::chrono::duration<Rep, Period> const& rel_time, std::memory_order order)
{ 
    return __wait_for(object, [=] __semaphore_abi (T other) -> bool { return old != other; }, rel_time, order); 
}

template <class T, class Predicate, class Rep, class Period>
__semaphore_abi bool condition_variable_atomic::wait_for(const volatile atomic<T>& object, Predicate pred, std::chrono::duration<Rep, Period> const& rel_time, std::memory_order order)
{ 
    return __wait_for(object, pred, rel_time, order); 
}

template <class T, class Predicate, class Rep, class Period>
__semaphore_abi bool condition_variable_atomic::wait_for(const atomic<T>& object, Predicate pred, std::chrono::duration<Rep, Period> const& rel_time, std::memory_order order)
{ 
    return __wait_for(object, pred, rel_time, order); 
}
