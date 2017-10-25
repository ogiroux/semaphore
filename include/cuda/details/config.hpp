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

#ifndef __CUDA_ATOMIC_CONFIG_HPP__
#define __CUDA_ATOMIC_CONFIG_HPP__

#include <cstdint>

#if !defined(__CUDACC__)

#error "This file defines CUDA operations; you must use CUDA to use it."

#elif !defined(__CUDA_ARCH__)

#define __mme_fence_signal_ abort
#define __mme_fence_sc_ abort
#define __mme_fence_ abort

#elif __CUDA_ARCH__ < 700

#ifdef __CUDA_ARCH__
    #warning "Use of atomic<T> is iffy without CUDA support for compute_70 or above."
    #if __CUDA_ARCH__ < 600
        #warning "Use of system-scoped atomic operations is iffy without CUDA support for compute_60  or above."
        #define DOTSYS
    #else
        #define DOTSYS ".sys"
    #endif
#else
    #define DOTSYS ".ERROR"
#endif

#define __mme_fence_signal_() asm volatile("":::"memory")
#define __mme_fence_sc_() asm volatile("membar.sys;":::"memory")
#define __mme_fence_() asm volatile("membar.sys;":::"memory")

#define __mme_load_acquire_8_as_32(ptr,ret) asm volatile("ld.volatile.b8 %0, [%1]; membar.sys;" : "=r"(ret) : "l"(ptr) : "memory")
#define __mme_load_relaxed_8_as_32(ptr,ret) asm volatile("ld.volatile.b8 %0, [%1];" : "=r"(ret) : "l"(ptr) : "memory")
#define __mme_store_release_8_as_32(ptr,desired) asm volatile("membar.sys; st.volatile.b8 [%0], %1;" :: "l"(ptr), "r"(desired) : "memory")
#define __mme_store_relaxed_8_as_32(ptr,desired) asm volatile("st.volatile.b8 [%0], %1;" :: "l"(ptr), "r"(desired) : "memory")

#define __mme_load_acquire_16(ptr,ret) asm volatile("ld.volatile.b16 %0, [%1]; membar.sys;" : "=h"(ret) : "l"(ptr) : "memory")
#define __mme_load_relaxed_16(ptr,ret) asm volatile("ld.volatile.b16 %0, [%1];" : "=h"(ret) : "l"(ptr) : "memory")
#define __mme_store_release_16(ptr,desired) asm volatile("membar.sys; st.volatile.b16 [%0], %1;" :: "l"(ptr), "h"(desired) : "memory")
#define __mme_store_relaxed_16(ptr,desired) asm volatile("st.volatile.b16 [%0], %1;" :: "l"(ptr), "h"(desired) : "memory")

#define __mme_load_acquire_32(ptr,ret) asm volatile("ld.volatile.b32 %0, [%1]; membar.sys;" : "=r"(ret) : "l"(ptr) : "memory")
#define __mme_load_relaxed_32(ptr,ret) asm volatile("ld.volatile.b32 %0, [%1];" : "=r"(ret) : "l"(ptr) : "memory")
#define __mme_store_release_32(ptr,desired) asm volatile("membar.sys; st.volatile.b32 [%0], %1;" :: "l"(ptr), "r"(desired) : "memory")
#define __mme_store_relaxed_32(ptr,desired) asm volatile("st.volatile.b32 [%0], %1;" :: "l"(ptr), "r"(desired) : "memory")
#define __mme_exch_release_32(ptr,old,desired) asm volatile("membar.sys; atom.exch" DOTSYS ".b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(desired) : "memory")
#define __mme_exch_acquire_32(ptr,old,desired) asm volatile("atom.exch" DOTSYS ".b32 %0, [%1], %2; membar.sys;" : "=r"(old) : "l"(ptr), "r"(desired) : "memory")
#define __mme_exch_acq_rel_32(ptr,old,desired) asm volatile("membar.sys; atom.exch" DOTSYS ".b32 %0, [%1], %2; membar.sys;" : "=r"(old) : "l"(ptr), "r"(desired) : "memory")
#define __mme_exch_relaxed_32(ptr,old,desired) asm volatile("atom.exch" DOTSYS ".b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(desired) : "memory")
#define __mme_cas_release_32(ptr,old,expected,desired) asm volatile("membar.sys; atom.cas" DOTSYS ".b32 %0, [%1], %2, %3;" : "=r"(old) : "l"(ptr), "r"(expected), "r"(desired) : "memory")
#define __mme_cas_acquire_32(ptr,old,expected,desired) asm volatile("atom.cas" DOTSYS ".b32 %0, [%1], %2, %3; membar.sys;" : "=r"(old) : "l"(ptr), "r"(expected), "r"(desired) : "memory")
#define __mme_cas_acq_rel_32(ptr,old,expected,desired) asm volatile("membar.sys; atom.cas" DOTSYS ".b32 %0, [%1], %2, %3; membar.sys;" : "=r"(old) : "l"(ptr), "r"(expected), "r"(desired) : "memory")
#define __mme_cas_relaxed_32(ptr,old,expected,desired) asm volatile("atom.cas" DOTSYS ".b32 %0, [%1], %2, %3;" : "=r"(old) : "l"(ptr), "r"(expected), "r"(desired) : "memory")
#define __mme_add_release_32(ptr,old,addend) asm volatile("membar.sys; atom.add" DOTSYS ".u32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(addend) : "memory")
#define __mme_add_acquire_32(ptr,old,addend) asm volatile("atom.add" DOTSYS ".u32 %0, [%1], %2; membar.sys;" : "=r"(old) : "l"(ptr), "r"(addend) : "memory")
#define __mme_add_acq_rel_32(ptr,old,addend) asm volatile("membar.sys; atom.add" DOTSYS ".u32 %0, [%1], %2; membar.sys;" : "=r"(old) : "l"(ptr), "r"(addend) : "memory")
#define __mme_add_relaxed_32(ptr,old,addend) asm volatile("atom.add" DOTSYS ".u32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(addend) : "memory")
#define __mme_and_release_32(ptr,old,andend) asm volatile("membar.sys; atom.and" DOTSYS ".b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(andend) : "memory")
#define __mme_and_acquire_32(ptr,old,andend) asm volatile("atom.and" DOTSYS ".b32 %0, [%1], %2; membar.sys;" : "=r"(old) : "l"(ptr), "r"(andend) : "memory")
#define __mme_and_acq_rel_32(ptr,old,andend) asm volatile("membar.sys; atom.and" DOTSYS ".b32 %0, [%1], %2; membar.sys;" : "=r"(old) : "l"(ptr), "r"(andend) : "memory")
#define __mme_and_relaxed_32(ptr,old,andend) asm volatile("atom.and" DOTSYS ".b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(andend) : "memory")
#define __mme_or_release_32(ptr,old,orend) asm volatile("membar.sys; atom.or" DOTSYS ".b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(orend) : "memory")
#define __mme_or_acquire_32(ptr,old,orend) asm volatile("atom.or" DOTSYS ".b32 %0, [%1], %2; membar.sys;" : "=r"(old) : "l"(ptr), "r"(orend) : "memory")
#define __mme_or_acq_rel_32(ptr,old,orend) asm volatile("membar.sys; atom.or" DOTSYS ".b32 %0, [%1], %2; membar.sys;" : "=r"(old) : "l"(ptr), "r"(orend) : "memory")
#define __mme_or_relaxed_32(ptr,old,orend) asm volatile("atom.or" DOTSYS ".b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(orend) : "memory")
#define __mme_xor_release_32(ptr,old,xorend) asm volatile("membar.sys; atom.xor" DOTSYS ".b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(xorend) : "memory")
#define __mme_xor_acquire_32(ptr,old,xorend) asm volatile("atom.xor" DOTSYS ".b32 %0, [%1], %2; membar.sys;" : "=r"(old) : "l"(ptr), "r"(xorend) : "memory")
#define __mme_xor_acq_rel_32(ptr,old,xorend) asm volatile("membar.sys; atom.xor" DOTSYS ".b32 %0, [%1], %2; membar.sys;" : "=r"(old) : "l"(ptr), "r"(xorend) : "memory")
#define __mme_xor_relaxed_32(ptr,old,xorend) asm volatile("atom.xor" DOTSYS ".b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(xorend) : "memory")

#define __mme_load_acquire_64(ptr,ret) asm volatile("ld.volatile.b64 %0, [%1]; membar.sys;" : "=l"(ret) : "l"(ptr) : "memory")
#define __mme_load_relaxed_64(ptr,ret) asm volatile("ld.volatile.b64 %0, [%1];" : "=l"(ret) : "l"(ptr) : "memory")
#define __mme_store_release_64(ptr,desired) asm volatile("membar.sys; st.volatile.b64 [%0], %1;" :: "l"(ptr), "l"(desired) : "memory")
#define __mme_store_relaxed_64(ptr,desired) asm volatile("st.volatile.b64 [%0], %1;" :: "l"(ptr), "l"(desired) : "memory")
#define __mme_exch_release_64(ptr,old,desired) asm volatile("membar.sys; atom.exch" DOTSYS ".b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(desired) : "memory")
#define __mme_exch_acquire_64(ptr,old,desired) asm volatile("atom.exch" DOTSYS ".b64 %0, [%1], %2; membar.sys;" : "=l"(old) : "l"(ptr), "l"(desired) : "memory")
#define __mme_exch_acq_rel_64(ptr,old,desired) asm volatile("membar.sys; atom.exch" DOTSYS ".b64 %0, [%1], %2; membar.sys;" : "=l"(old) : "l"(ptr), "l"(desired) : "memory")
#define __mme_exch_relaxed_64(ptr,old,desired) asm volatile("atom.exch" DOTSYS ".b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(desired) : "memory")
#define __mme_cas_release_64(ptr,old,expected,desired) asm volatile("membar.sys; atom.cas" DOTSYS ".b64 %0, [%1], %2, %3;" : "=l"(old) : "l"(ptr), "l"(expected), "l"(desired) : "memory")
#define __mme_cas_acquire_64(ptr,old,expected,desired) asm volatile("atom.cas" DOTSYS ".b64 %0, [%1], %2, %3; membar.sys;" : "=l"(old) : "l"(ptr), "l"(expected), "l"(desired) : "memory")
#define __mme_cas_acq_rel_64(ptr,old,expected,desired) asm volatile("membar.sys; atom.cas" DOTSYS ".b64 %0, [%1], %2, %3; membar.sys;" : "=l"(old) : "l"(ptr), "l"(expected), "l"(desired) : "memory")
#define __mme_cas_relaxed_64(ptr,old,expected,desired) asm volatile("atom.cas" DOTSYS ".b64 %0, [%1], %2, %3;" : "=l"(old) : "l"(ptr), "l"(expected), "l"(desired) : "memory")
#define __mme_add_release_64(ptr,old,addend) asm volatile("membar.sys; atom.add" DOTSYS ".u64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(addend) : "memory")
#define __mme_add_acquire_64(ptr,old,addend) asm volatile("atom.add" DOTSYS ".u64 %0, [%1], %2; membar.sys;" : "=l"(old) : "l"(ptr), "l"(addend) : "memory")
#define __mme_add_acq_rel_64(ptr,old,addend) asm volatile("membar.sys; atom.add" DOTSYS ".u64 %0, [%1], %2; membar.sys;" : "=l"(old) : "l"(ptr), "l"(addend) : "memory")
#define __mme_add_relaxed_64(ptr,old,addend) asm volatile("atom.add" DOTSYS ".u64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(addend) : "memory")
#define __mme_and_release_64(ptr,old,andend) asm volatile("membar.sys; atom.and" DOTSYS ".b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(andend) : "memory")
#define __mme_and_acquire_64(ptr,old,andend) asm volatile("atom.and" DOTSYS ".b64 %0, [%1], %2; membar.sys;" : "=l"(old) : "l"(ptr), "l"(andend) : "memory")
#define __mme_and_acq_rel_64(ptr,old,andend) asm volatile("membar.sys; atom.and" DOTSYS ".b64 %0, [%1], %2; membar.sys;" : "=l"(old) : "l"(ptr), "l"(andend) : "memory")
#define __mme_and_relaxed_64(ptr,old,andend) asm volatile("atom.and" DOTSYS ".b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(andend) : "memory")
#define __mme_or_release_64(ptr,old,orend) asm volatile("membar.sys; atom.or" DOTSYS ".b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(orend) : "memory")
#define __mme_or_acquire_64(ptr,old,orend) asm volatile("atom.or" DOTSYS ".b64 %0, [%1], %2; membar.sys;" : "=l"(old) : "l"(ptr), "l"(orend) : "memory")
#define __mme_or_acq_rel_64(ptr,old,orend) asm volatile("membar.sys; atom.or" DOTSYS ".b64 %0, [%1], %2; membar.sys;" : "=l"(old) : "l"(ptr), "l"(orend) : "memory")
#define __mme_or_relaxed_64(ptr,old,orend) asm volatile("atom.or" DOTSYS ".b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(orend) : "memory")
#define __mme_xor_release_64(ptr,old,xorend) asm volatile("membar.sys; atom.xor" DOTSYS ".b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(xorend) : "memory")
#define __mme_xor_acquire_64(ptr,old,xorend) asm volatile("atom.xor" DOTSYS ".b64 %0, [%1], %2; membar.sys;" : "=l"(old) : "l"(ptr), "l"(xorend) : "memory")
#define __mme_xor_acq_rel_64(ptr,old,xorend) asm volatile("membar.sys; atom.xor" DOTSYS ".b64 %0, [%1], %2; membar.sys;" : "=l"(old) : "l"(ptr), "l"(xorend) : "memory")
#define __mme_xor_relaxed_64(ptr,old,xorend) asm volatile("atom.xor" DOTSYS ".b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(xorend) : "memory")

#else // __CUDA_ARCH__ >= 700

#define __mme_fence_signal_() asm volatile("":::"memory")
#define __mme_fence_sc_() asm volatile("fence.sc.sys;":::"memory")
#define __mme_fence_() asm volatile("fence.sys;":::"memory")

#define __mme_load_acquire_8_as_32(ptr,ret) asm volatile("ld.acquire.sys.b8 %0, [%1];" : "=r"(ret) : "l"(ptr) : "memory")
#define __mme_load_relaxed_8_as_32(ptr,ret) asm volatile("ld.relaxed.sys.b8 %0, [%1];" : "=r"(ret) : "l"(ptr) : "memory")
#define __mme_store_release_8_as_32(ptr,desired) asm volatile("st.release.sys.b8 [%0], %1;" :: "l"(ptr), "r"(desired) : "memory")
#define __mme_store_relaxed_8_as_32(ptr,desired) asm volatile("st.relaxed.sys.b8 [%0], %1;" :: "l"(ptr), "r"(desired) : "memory")

#define __mme_load_acquire_16(ptr,ret) asm volatile("ld.acquire.sys.b16 %0, [%1];" : "=h"(ret) : "l"(ptr) : "memory")
#define __mme_load_relaxed_16(ptr,ret) asm volatile("ld.relaxed.sys.b16 %0, [%1];" : "=h"(ret) : "l"(ptr) : "memory")
#define __mme_store_release_16(ptr,desired) asm volatile("st.release.sys.b16 [%0], %1;" :: "l"(ptr), "h"(desired) : "memory")
#define __mme_store_relaxed_16(ptr,desired) asm volatile("st.relaxed.sys.b16 [%0], %1;" :: "l"(ptr), "h"(desired) : "memory")

#define __mme_load_acquire_32(ptr,ret) asm volatile("ld.acquire.sys.b32 %0, [%1];" : "=r"(ret) : "l"(ptr) : "memory")
#define __mme_load_relaxed_32(ptr,ret) asm volatile("ld.relaxed.sys.b32 %0, [%1];" : "=r"(ret) : "l"(ptr) : "memory")
#define __mme_store_release_32(ptr,desired) asm volatile("st.release.sys.b32 [%0], %1;" :: "l"(ptr), "r"(desired) : "memory")
#define __mme_store_relaxed_32(ptr,desired) asm volatile("st.relaxed.sys.b32 [%0], %1;" :: "l"(ptr), "r"(desired) : "memory")
#define __mme_exch_release_32(ptr,old,desired) asm volatile("atom.exch.release.sys.b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(desired) : "memory")
#define __mme_exch_acquire_32(ptr,old,desired) asm volatile("atom.exch.acquire.sys.b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(desired) : "memory")
#define __mme_exch_acq_rel_32(ptr,old,desired) asm volatile("atom.exch.acq_rel.sys.b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(desired) : "memory")
#define __mme_exch_relaxed_32(ptr,old,desired) asm volatile("atom.exch.relaxed.sys.b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(desired) : "memory")
#define __mme_cas_release_32(ptr,old,expected,desired) asm volatile("atom.cas.release.sys.b32 %0, [%1], %2, %3;" : "=r"(old) : "l"(ptr), "r"(expected), "r"(desired) : "memory")
#define __mme_cas_acquire_32(ptr,old,expected,desired) asm volatile("atom.cas.acquire.sys.b32 %0, [%1], %2, %3;" : "=r"(old) : "l"(ptr), "r"(expected), "r"(desired) : "memory")
#define __mme_cas_acq_rel_32(ptr,old,expected,desired) asm volatile("atom.cas.acq_rel.sys.b32 %0, [%1], %2, %3;" : "=r"(old) : "l"(ptr), "r"(expected), "r"(desired) : "memory")
#define __mme_cas_relaxed_32(ptr,old,expected,desired) asm volatile("atom.cas.relaxed.sys.b32 %0, [%1], %2, %3;" : "=r"(old) : "l"(ptr), "r"(expected), "r"(desired) : "memory")
#define __mme_add_release_32(ptr,old,addend) asm volatile("atom.add.release.sys.u32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(addend) : "memory")
#define __mme_add_acquire_32(ptr,old,addend) asm volatile("atom.add.acquire.sys.u32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(addend) : "memory")
#define __mme_add_acq_rel_32(ptr,old,addend) asm volatile("atom.add.acq_rel.sys.u32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(addend) : "memory")
#define __mme_add_relaxed_32(ptr,old,addend) asm volatile("atom.add.relaxed.sys.u32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(addend) : "memory")
#define __mme_and_release_32(ptr,old,andend) asm volatile("atom.and.release.sys.b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(andend) : "memory")
#define __mme_and_acquire_32(ptr,old,andend) asm volatile("atom.and.acquire.sys.b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(andend) : "memory")
#define __mme_and_acq_rel_32(ptr,old,andend) asm volatile("atom.and.acq_rel.sys.b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(andend) : "memory")
#define __mme_and_relaxed_32(ptr,old,andend) asm volatile("atom.and.relaxed.sys.b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(andend) : "memory")
#define __mme_or_release_32(ptr,old,orend) asm volatile("atom.or.release.sys.b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(orend) : "memory")
#define __mme_or_acquire_32(ptr,old,orend) asm volatile("atom.or.acquire.sys.b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(orend) : "memory")
#define __mme_or_acq_rel_32(ptr,old,orend) asm volatile("atom.or.acq_rel.sys.b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(orend) : "memory")
#define __mme_or_relaxed_32(ptr,old,orend) asm volatile("atom.or.relaxed.sys.b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(orend) : "memory")
#define __mme_xor_release_32(ptr,old,xorend) asm volatile("atom.xor.release.sys.b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(xorend) : "memory")
#define __mme_xor_acquire_32(ptr,old,xorend) asm volatile("atom.xor.acquire.sys.b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(xorend) : "memory")
#define __mme_xor_acq_rel_32(ptr,old,xorend) asm volatile("atom.xor.acq_rel.sys.b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(xorend) : "memory")
#define __mme_xor_relaxed_32(ptr,old,xorend) asm volatile("atom.xor.relaxed.sys.b32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(xorend) : "memory")

#define __mme_load_acquire_64(ptr,ret) asm volatile("ld.acquire.sys.b64 %0, [%1];" : "=l"(ret) : "l"(ptr) : "memory")
#define __mme_load_relaxed_64(ptr,ret) asm volatile("ld.relaxed.sys.b64 %0, [%1];" : "=l"(ret) : "l"(ptr) : "memory")
#define __mme_store_release_64(ptr,desired) asm volatile("st.release.sys.b64 [%0], %1;" :: "l"(ptr), "l"(desired) : "memory")
#define __mme_store_relaxed_64(ptr,desired) asm volatile("st.relaxed.sys.b64 [%0], %1;" :: "l"(ptr), "l"(desired) : "memory")
#define __mme_exch_release_64(ptr,old,desired) asm volatile("atom.exch.release.sys.b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(desired) : "memory")
#define __mme_exch_acquire_64(ptr,old,desired) asm volatile("atom.exch.acquire.sys.b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(desired) : "memory")
#define __mme_exch_acq_rel_64(ptr,old,desired) asm volatile("atom.exch.acq_rel.sys.b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(desired) : "memory")
#define __mme_exch_relaxed_64(ptr,old,desired) asm volatile("atom.exch.relaxed.sys.b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(desired) : "memory")
#define __mme_cas_release_64(ptr,old,expected,desired) asm volatile("atom.cas.release.sys.b64 %0, [%1], %2, %3;" : "=l"(old) : "l"(ptr), "l"(expected), "l"(desired) : "memory")
#define __mme_cas_acquire_64(ptr,old,expected,desired) asm volatile("atom.cas.acquire.sys.b64 %0, [%1], %2, %3;" : "=l"(old) : "l"(ptr), "l"(expected), "l"(desired) : "memory")
#define __mme_cas_acq_rel_64(ptr,old,expected,desired) asm volatile("atom.cas.acq_rel.sys.b64 %0, [%1], %2, %3;" : "=l"(old) : "l"(ptr), "l"(expected), "l"(desired) : "memory")
#define __mme_cas_relaxed_64(ptr,old,expected,desired) asm volatile("atom.cas.relaxed.sys.b64 %0, [%1], %2, %3;" : "=l"(old) : "l"(ptr), "l"(expected), "l"(desired) : "memory")
#define __mme_add_release_64(ptr,old,addend) asm volatile("atom.add.release.sys.u64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(addend) : "memory")
#define __mme_add_acquire_64(ptr,old,addend) asm volatile("atom.add.acquire.sys.u64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(addend) : "memory")
#define __mme_add_acq_rel_64(ptr,old,addend) asm volatile("atom.add.acq_rel.sys.u64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(addend) : "memory")
#define __mme_add_relaxed_64(ptr,old,addend) asm volatile("atom.add.relaxed.sys.u64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(addend) : "memory")
#define __mme_and_release_64(ptr,old,andend) asm volatile("atom.and.release.sys.b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(andend) : "memory")
#define __mme_and_acquire_64(ptr,old,andend) asm volatile("atom.and.acquire.sys.b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(andend) : "memory")
#define __mme_and_acq_rel_64(ptr,old,andend) asm volatile("atom.and.acq_rel.sys.b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(andend) : "memory")
#define __mme_and_relaxed_64(ptr,old,andend) asm volatile("atom.and.relaxed.sys.b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(andend) : "memory")
#define __mme_or_release_64(ptr,old,orend) asm volatile("atom.or.release.sys.b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(orend) : "memory")
#define __mme_or_acquire_64(ptr,old,orend) asm volatile("atom.or.acquire.sys.b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(orend) : "memory")
#define __mme_or_acq_rel_64(ptr,old,orend) asm volatile("atom.or.acq_rel.sys.b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(orend) : "memory")
#define __mme_or_relaxed_64(ptr,old,orend) asm volatile("atom.or.relaxed.sys.b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(orend) : "memory")
#define __mme_xor_release_64(ptr,old,xorend) asm volatile("atom.xor.release.sys.b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(xorend) : "memory")
#define __mme_xor_acquire_64(ptr,old,xorend) asm volatile("atom.xor.acquire.sys.b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(xorend) : "memory")
#define __mme_xor_acq_rel_64(ptr,old,xorend) asm volatile("atom.xor.acq_rel.sys.b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(xorend) : "memory")
#define __mme_xor_relaxed_64(ptr,old,xorend) asm volatile("atom.xor.relaxed.sys.b64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(xorend) : "memory")

#define __mme_load_relaxed_mmio_8_as_32(ptr,ret) __mme_load_relaxed_mmio_8_as_32_(ptr, reinterpret_cast<uint32_t&>(ret))
#define __mme_store_relaxed_mmio_8_as_32(ptr,desired) __mme_store_relaxed_mmio_8_as_32_(ptr,reinterpret_cast<uint32_t const&>(desired))
#define __mme_load_relaxed_mmio_16(ptr,ret) __mme_load_relaxed_mmio_16_(ptr,reinterpret_cast<uint16_t&>(ret))
#define __mme_store_relaxed_mmio_16(ptr,desired) __mme_store_relaxed_mmio_16_(ptr,reinterpret_cast<uint16_t const&>(desired))
#define __mme_load_relaxed_mmio_32(ptr,ret) __mme_load_relaxed_mmio_32_(ptr,reinterpret_cast<uint32_t&>(ret))
#define __mme_store_relaxed_mmio_32(ptr,desired) __mme_store_relaxed_mmio_32_(ptr,reinterpret_cast<uint32_t const&>(desired))
#define __mme_exch_relaxed_mmio_32(ptr,old,desired) __mme_exch_relaxed_mmio_32_(ptr,reinterpret_cast<uint32_t&>(old),reinterpret_cast<uint32_t const&>(desired))
#define __mme_cas_relaxed_mmio_32(ptr,old,expected,desired) __mme_cas_relaxed_mmio_32_(ptr,reinterpret_cast<uint32_t&>(old),reinterpret_cast<uint32_t const&>(expected),reinterpret_cast<uint32_t const&>(desired))
#define __mme_add_relaxed_mmio_32(ptr,old,addend) __mme_add_relaxed_mmio_32_(ptr,reinterpret_cast<uint32_t&>(old),reinterpret_cast<uint32_t const&>(addend))
#define __mme_and_relaxed_mmio_32(ptr,old,andend) __mme_and_relaxed_mmio_32_(ptr,reinterpret_cast<uint32_t&>(old),reinterpret_cast<uint32_t const&>(andend))
#define __mme_or_relaxed_mmio_32(ptr,old,orend) __mme_or_relaxed_mmio_32_(ptr,reinterpret_cast<uint32_t&>(old),reinterpret_cast<uint32_t const&>(orend))
#define __mme_xor_relaxed_mmio_32(ptr,old,xorend) __mme_xor_relaxed_mmio_32_(ptr,reinterpret_cast<uint32_t&>(old),reinterpret_cast<uint32_t const&>(xorend))
#define __mme_load_relaxed_mmio_64(ptr,ret) __mme_load_relaxed_mmio_64_(ptr,reinterpret_cast<uint32_t&>(ret))
#define __mme_store_relaxed_mmio_64(ptr,desired) __mme_store_relaxed_mmio_64_(ptr,reinterpret_cast<uint32_t const&>(desired))
#define __mme_exch_relaxed_mmio_64(ptr,old,desired) __mme_exch_relaxed_mmio_64_(ptr,reinterpret_cast<uint32_t&>(old),reinterpret_cast<uint32_t const&>(desired))
#define __mme_cas_relaxed_mmio_64(ptr,old,expected,desired) __mme_cas_relaxed_mmio_64_(ptr,reinterpret_cast<uint32_t&>(old),reinterpret_cast<uint32_t const&>(expected),reinterpret_cast<uint32_t const&>(desired))
#define __mme_add_relaxed_mmio_64(ptr,old,addend) __mme_add_relaxed_mmio_64_(ptr,reinterpret_cast<uint32_t&>(old),reinterpret_cast<uint32_t const&>(addend))
#define __mme_and_relaxed_mmio_64(ptr,old,andend) __mme_and_relaxed_mmio_64_(ptr,reinterpret_cast<uint32_t&>(old),reinterpret_cast<uint32_t const&>(andend))
#define __mme_or_relaxed_mmio_64(ptr,old,orend) __mme_or_relaxed_mmio_64_(ptr,reinterpret_cast<uint32_t&>(old),reinterpret_cast<uint32_t const&>(orend))
#define __mme_xor_relaxed_mmio_64(ptr,old,xorend) __mme_xor_relaxed_mmio_64_(ptr,reinterpret_cast<uint32_t&>(old),reinterpret_cast<uint32_t const&>(xorend))

#ifndef __clang__

namespace cuda { namespace experimental { inline namespace v1 { namespace details {

__device__ void __mme_nanosleep(uint32_t) noexcept;
__device__ void __mme_load_relaxed_mmio_8_as_32_(void const volatile*, uint32_t&) noexcept;
__device__ void __mme_store_relaxed_mmio_8_as_32_(void volatile*, uint32_t&) noexcept;
__device__ void __mme_load_relaxed_mmio_16_(void const volatile*, uint16_t&) noexcept;
__device__ void __mme_store_relaxed_mmio_16_(void volatile*, uint16_t const&) noexcept;
__device__ void __mme_load_relaxed_mmio_32_(void const volatile*, uint32_t&) noexcept;
__device__ void __mme_store_relaxed_mmio_32_(void volatile*, uint32_t const&) noexcept;
__device__ void __mme_exch_relaxed_mmio_32_(void volatile*, uint32_t&, uint32_t const&) noexcept;
__device__ void __mme_cas_relaxed_mmio_32_(void volatile*, uint32_t&, uint32_t const&, uint32_t const& desired) noexcept;
__device__ void __mme_add_relaxed_mmio_32_(void volatile*, uint32_t&, uint32_t const&) noexcept;
__device__ void __mme_and_relaxed_mmio_32_(void volatile*, uint32_t&, uint32_t const&) noexcept;
__device__ void __mme_or_relaxed_mmio_32_(void volatile*, uint32_t&, uint32_t const&) noexcept;
__device__ void __mme_xor_relaxed_mmio_32_(void volatile*, uint32_t&, uint32_t const&) noexcept;
__device__ void __mme_load_relaxed_mmio_64_(void const volatile*, uint32_t&) noexcept;
__device__ void __mme_store_relaxed_mmio_64_(void volatile*, uint32_t const&) noexcept;
__device__ void __mme_exch_relaxed_mmio_64_(void volatile*, uint32_t&, uint32_t const&) noexcept;
__device__ void __mme_cas_relaxed_mmio_64_(void volatile*, uint32_t&, uint32_t const&, uint32_t const& desired) noexcept;
__device__ void __mme_add_relaxed_mmio_64_(void volatile*, uint32_t&, uint32_t const&) noexcept;
__device__ void __mme_and_relaxed_mmio_64_(void volatile*, uint32_t&, uint32_t const&) noexcept;
__device__ void __mme_or_relaxed_mmio_64_(void volatile*, uint32_t&, uint32_t const&) noexcept;
__device__ void __mme_xor_relaxed_mmio_64_(void volatile*, uint32_t&, uint32_t const&) noexcept;

#define __has_cuda_mmio
#define __has_cuda_nanosleep

}}}}

#endif //__clang__

#endif // __CUDACC__

#ifndef __has_cuda_nanosleep
#define __has_cuda_nanosleep
__device__ static inline void __mme_nanosleep(uint32_t) noexcept { }
#endif //__has_cuda_nanosleep

//Jesus, this stuff ->
#ifndef __has_cuda_mmio
#define __has_cuda_mmio
#define __mme_load_relaxed_mmio_8_as_32(ptr,ret)            __mme_load_relaxed_32(ptr,ret)
#define __mme_store_relaxed_mmio_8_as_32(ptr,desired)       __mme_store_relaxed_32(ptr,desired)
#define __mme_load_relaxed_mmio_16(ptr,ret)                 __mme_load_relaxed_16(ptr,ret)
#define __mme_store_relaxed_mmio_16(ptr,desired)            __mme_store_relaxed_16(ptr,desired)
#define __mme_load_relaxed_mmio_32(ptr,ret)                 __mme_load_relaxed_32(ptr,ret)
#define __mme_store_relaxed_mmio_32(ptr,desired)            __mme_store_relaxed_32(ptr,desired)
#define __mme_exch_relaxed_mmio_32(ptr,old,desired)         __mme_exch_relaxed_32(ptr,old,desired)
#define __mme_cas_relaxed_mmio_32(ptr,old,expected,desired) __mme_cas_relaxed_32(ptr,old,expected,desired)
#define __mme_add_relaxed_mmio_32(ptr,old,addend)           __mme_add_relaxed_32(ptr,old,addend)
#define __mme_and_relaxed_mmio_32(ptr,old,andend)           __mme_and_relaxed_32(ptr,old,andend)
#define __mme_or_relaxed_mmio_32(ptr,old,orend)             __mme_or_relaxed_32(ptr,old,orend)
#define __mme_xor_relaxed_mmio_32(ptr,old,xorend)           __mme_xor_relaxed_32(ptr,old,xorend)
#define __mme_load_relaxed_mmio_64(ptr,ret)                 __mme_load_relaxed_64(ptr,ret)
#define __mme_store_relaxed_mmio_64(ptr,desired)            __mme_store_relaxed_64(ptr,desired)
#define __mme_exch_relaxed_mmio_64(ptr,old,desired)         __mme_exch_relaxed_64(ptr,old,desired)
#define __mme_cas_relaxed_mmio_64(ptr,old,expected,desired) __mme_cas_relaxed_64(ptr,old,expected,desired)
#define __mme_add_relaxed_mmio_64(ptr,old,addend)           __mme_add_relaxed_64(ptr,old,addend)
#define __mme_and_relaxed_mmio_64(ptr,old,andend)           __mme_and_relaxed_64(ptr,old,andend)
#define __mme_or_relaxed_mmio_64(ptr,old,orend)             __mme_or_relaxed_64(ptr,old,orend)
#define __mme_xor_relaxed_mmio_64(ptr,old,xorend)           __mme_xor_relaxed_64(ptr,old,xorend)
#endif //__has_cuda_mmio

#define __has_cuda_atomic

#endif //__CUDA_ATOMIC_CONFIG_HPP__
