
#ifndef __semaphore_cuda
extern __semaphore_abi condition_variable_atomic *__condition_variable_atomic_from_ptr(void const *ptr);
extern __semaphore_abi condition_variable_atomic *__condition_variable_atomic_from_ptr(void const volatile *ptr);
#endif

#include "details/bin_sem.hpp"
#include "details/int_sem.hpp"
#include "details/con_var.hpp"
#include "details/notify.hpp"
