
// 32.10, waiting and notifying functions

template <class T>
__semaphore_abi void atomic_notify_one(const volatile atomic<T>*a)
{
    condition_variable_atomic::
    __from_ptr(a)->
    notify_one(*a);
}

template <class T>
__semaphore_abi void atomic_notify_one(const atomic<T>*a)
{
    condition_variable_atomic::
    __from_ptr(a)->
    notify_one(*a);
}

template <class T>
__semaphore_abi void atomic_notify_all(const volatile atomic<T>* a)
{
    condition_variable_atomic::
    __from_ptr(a)->
    notify_all(*a);
}

template <class T>
__semaphore_abi void atomic_notify_all(const atomic<T>* a)
{
    condition_variable_atomic::
    __from_ptr(a)->
    notify_all(*a);
}

template <class T, class V>
__semaphore_abi void atomic_wait_explicit(const volatile atomic<T>* a, V oldval, std::memory_order order)
{
    condition_variable_atomic::
    __from_ptr(a)->
    wait(*a, oldval, order);
}

template <class T, class V>
__semaphore_abi void atomic_wait_explicit(const atomic<T>* a, V oldval, std::memory_order order)
{
    condition_variable_atomic::
    __from_ptr(a)->
    wait(*a, oldval, order);
}

template <class T, class V>
__semaphore_abi void atomic_wait(const volatile atomic<T>* a, V oldval)
{
    condition_variable_atomic::
    __from_ptr(a)->
    wait(*a, oldval);
}

template <class T, class V>
__semaphore_abi void atomic_wait(const atomic<T>* a, V oldval)
{
    condition_variable_atomic::
    __from_ptr(a)->
    wait(*a, oldval);
}
