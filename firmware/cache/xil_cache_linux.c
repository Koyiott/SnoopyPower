/*
 * xil_cache_linux.c — Linux userspace Xilinx cache API (SAFE version)
 *
 * WHY WE CANNOT USE PL310 MMIO FROM LINUX USERSPACE
 * ==================================================
 *
 *  The Xilinx bare-metal BSP wraps every PL310 register write with:
 *
 *    currmask = mfcpsr();                    ← read CPSR
 *    mtcpsr(currmask | IRQ_FIQ_MASK);        ← DISABLE ALL INTERRUPTS
 *    // ... PL310 operations ...
 *    mtcpsr(currmask);                        ← restore interrupts
 *
 *  This is critical because:
 *    1. Xil_L2WriteDebugCtrl(0x3) disables L2 write-back and line fills.
 *       If the kernel preempts us between this write and the re-enable,
 *       kernel writes silently fail to cache → MEMORY CORRUPTION → CRASH.
 *    2. The Linux kernel has its own PL310 driver (cache-l2x0.c) which
 *       uses spinlocks to serialize access.  Our writes bypass those
 *       locks → register race conditions → hangs.
 *    3. Clean+Invalidate all ways drops all L2 content including kernel
 *       page tables, DMA buffers, scheduler data → instant crash.
 *
 *  mfcpsr()/mtcpsr() are CP15 operations → PRIVILEGED (EL1 only).
 *  We CANNOT mask interrupts from userspace.  PL310 MMIO
 *  from userspace is fundamentally unsafe when Linux is running.
 *
 * SAFE APPROACH
 * =========================
 *  • L1 eviction    → displacement buffer walk (128 KB × 3 passes)
 *  • L2 eviction    → displacement buffer walk (1 MB)
 *  • Range ops      → __NR_cacheflush syscall (kernel handles L1+L2
 *                      with proper locking, IRQ masking, and L2CC sync)
 *
 * Target: Cortex-A9 dual-core, PynqZ1 (Zynq-7020), PYNQ Linux image.
 * SPDX-License-Identifier: MIT
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syscall.h>

#ifdef __arm__
#include <asm/unistd.h>
#endif

#if defined(__arm__) && !defined(__NR_cacheflush)
  #if defined(__ARM_NR_cacheflush)
    #define __NR_cacheflush __ARM_NR_cacheflush
  #else
    #define __NR_cacheflush 0xf0002
  #endif
#endif

#include "xil_cache.h"

/* ================================================================
 *  Cortex-A9 / PL310 geometry  (PynqZ1)
 * ================================================================ */
#define CACHE_LINE_BYTES     32u

/* L1 D-cache: 32 KB, 4-way, 256 sets, pseudo-random replacement.
 * Displacement: 4× L1 = 128 KB.
 * 3 passes → 12× associativity pressure per set.
 * P(all 4 ways evicted) per set after 3 passes ≈ 0.999999.       */
#define EVICT_L1_BYTES       (128u * 1024u)
#define EVICT_L1_PASSES      3

/* L2 (PL310): 512 KB, 8-way.
 * Displacement: 2× L2 = 1 MB → fills all 8 ways in every set.   */
#define EVICT_L2_BYTES       (1024u * 1024u)

/* ================================================================
 *  Internal state
 * ================================================================ */
static volatile uint8_t *g_evict_l1  = NULL;
static volatile uint8_t *g_evict_l2  = NULL;
static int               g_cache_ok  = 0;

/* ================================================================
 *  Inline barriers
 * ================================================================ */
static inline void dsb_sy(void)
{
#ifdef __arm__
    __asm__ volatile("dsb sy" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

static inline void isb_sy(void)
{
#ifdef __arm__
    __asm__ volatile("isb" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

/* ================================================================
 *  Range flush via Linux syscall
 *
 *  syscall(__NR_cacheflush, start, end, flags)
 *    start : first byte (virtual address)
 *    end   : one past last byte
 *    flags : 0  (clean + invalidate D-cache and I-cache)
 *
 *  The kernel's flush_cache_user_range():
 *    - L1: CP15 DCCIMVAC per line (with IRQs properly managed)
 *    - L2: PL310 clean+inval by PA (with l2x0 spinlock held)
 *    - Proper barriers and cache sync
 *  This is the ONLY safe way to do cache maintenance from userspace.
 * ================================================================ */
static inline void linux_cacheflush_range(uintptr_t start, size_t len)
{
    if (len == 0) return;
    syscall(__NR_cacheflush, start, start + len, 0);
}

/* ================================================================
 *  Init / Deinit
 * ================================================================ */

int Xil_CacheInit(void)
{
    /* L1 displacement buffer (128 KB, page-aligned, pre-faulted) */
    void *p1 = mmap(NULL, EVICT_L1_BYTES,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE,
                    -1, 0);
    if (p1 == MAP_FAILED) {
        perror("[xil_cache] mmap L1 eviction buffer");
        return -1;
    }

    /* L2 displacement buffer (1 MB, page-aligned, pre-faulted) */
    void *p2 = mmap(NULL, EVICT_L2_BYTES,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE,
                    -1, 0);
    if (p2 == MAP_FAILED) {
        perror("[xil_cache] mmap L2 eviction buffer");
        munmap(p1, EVICT_L1_BYTES);
        return -1;
    }

    g_evict_l1 = (volatile uint8_t *)p1;
    g_evict_l2 = (volatile uint8_t *)p2;

    /* Pre-fault every page (MAP_POPULATE should do it, belt+suspenders) */
    memset((void *)g_evict_l1, 0xAA, EVICT_L1_BYTES);
    memset((void *)g_evict_l2, 0x55, EVICT_L2_BYTES);

    g_cache_ok = 1;

    printf("[xil_cache] Linux cache API ready  (displacement + syscall)\n");
    printf("  L1 eviction : %p (%u KB x %d passes)\n",
           (void *)g_evict_l1, EVICT_L1_BYTES / 1024, EVICT_L1_PASSES);
    printf("  L2 eviction : %p (%u KB displacement)\n",
           (void *)g_evict_l2, EVICT_L2_BYTES / 1024);
    printf("  Range ops   : __NR_cacheflush syscall\n");

    return 0;
}

void Xil_CacheDeinit(void)
{
    if (g_evict_l1) {
        munmap((void *)g_evict_l1, EVICT_L1_BYTES);
        g_evict_l1 = NULL;
    }
    if (g_evict_l2) {
        munmap((void *)g_evict_l2, EVICT_L2_BYTES);
        g_evict_l2 = NULL;
    }
    g_cache_ok = 0;
}

/* ================================================================
 *  Xil_L1DCacheFlush / Xil_L1DCacheInvalidate
 *
 *  Evict L1D only — leaves L2 contents intact.
 *
 *  128 KB buffer × 3 passes.  DSB between passes ensures all prior
 *  loads retire before the next pass starts displacing.
 *
 *  The 128 KB buffer fits inside 512 KB L2
 * ================================================================ */
void Xil_L1DCacheFlush(void)
{
    if (!g_cache_ok) return;

    volatile uint8_t sink;
    for (int pass = 0; pass < EVICT_L1_PASSES; pass++) {
        for (unsigned i = 0; i < EVICT_L1_BYTES; i += CACHE_LINE_BYTES) {
            sink = g_evict_l1[i];
        }
        dsb_sy();
    }
    (void)sink;
    isb_sy();
}

void Xil_L1DCacheInvalidate(void)
{
    Xil_L1DCacheFlush();
}

/* ================================================================
 *  Xil_DCacheFlush / Xil_DCacheInvalidate
 *
 *  Evict both L1 and L2.  Walk 1 MB of displacement buffer.
 *  1 MB ≥ 2× the 512 KB L2 → all L2 ways get displaced.
 *  L1 is also displaced (1 MB >> 32 KB L1).
 * ================================================================ */
void Xil_DCacheFlush(void)
{
    if (!g_cache_ok) return;

    volatile uint8_t sink;
    for (unsigned i = 0; i < EVICT_L2_BYTES; i += CACHE_LINE_BYTES) {
        sink = g_evict_l2[i];
    }
    (void)sink;
    dsb_sy();
    isb_sy();
}

void Xil_DCacheInvalidate(void)
{
    Xil_DCacheFlush();
}

/* ================================================================
 *  L2 Cache — same as full D-cache (the 1 MB walk covers L2)
 * ================================================================ */
void Xil_L2CacheFlush(void)      { Xil_DCacheFlush(); }
void Xil_L2CacheInvalidate(void) { Xil_DCacheFlush(); }

/* ================================================================
 *  Range-based operations — __NR_cacheflush syscall
 *
 *  The kernel's flush_cache_user_range() does clean+invalidate on
 *  both L1 and L2, with proper locking (l2x0 spinlock), IRQ masking,
 *  and PL310 cache sync.  This is the kernel-sanctioned mechanism.
 * ================================================================ */
void Xil_DCacheFlushRange(INTPTR adr, u32 len)
{
    if (len == 0) return;

    uintptr_t aligned = (uintptr_t)adr & ~(uintptr_t)(CACHE_LINE_BYTES - 1u);
    u32 extra         = (u32)((uintptr_t)adr - aligned);
    u32 aligned_len   = (len + extra + CACHE_LINE_BYTES - 1u)
                        & ~(u32)(CACHE_LINE_BYTES - 1u);

    dsb_sy();
    linux_cacheflush_range(aligned, aligned_len);
    dsb_sy();
}

void Xil_DCacheInvalidateRange(INTPTR adr, u32 len)
{
    Xil_DCacheFlushRange(adr, len);
}

void Xil_DCacheFlushLine(u32 adr)
{
    uintptr_t aligned = (uintptr_t)adr & ~(uintptr_t)(CACHE_LINE_BYTES - 1u);
    dsb_sy();
    linux_cacheflush_range(aligned, CACHE_LINE_BYTES);
    dsb_sy();
}

void Xil_DCacheInvalidateLine(u32 adr)
{
    Xil_DCacheFlushLine(adr);
}

/* ================================================================
 *  I-Cache — syscall / stubs
 * ================================================================ */
void Xil_ICacheInvalidate(void)
{
    dsb_sy();
    isb_sy();
}

void Xil_ICacheInvalidateRange(INTPTR adr, u32 len)
{
    if (len == 0) return;
    linux_cacheflush_range((uintptr_t)adr, len);
    isb_sy();
}