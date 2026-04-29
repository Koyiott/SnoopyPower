#ifndef MEMBENCH_H
#define MEMBENCH_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <inttypes.h>
#include "include/pmu.h"

#define _GNU_SOURCE
#include <unistd.h>
#include <sys/syscall.h>


#ifdef __arm__
#include <asm/unistd.h> /* for __NR_cacheflush on ARM */
#endif

// Fallback definition for ARM architecture
#if defined(__arm__) && !defined(__NR_cacheflush)
    #if defined(__ARM_NR_cacheflush)
        #define __NR_cacheflush __ARM_NR_cacheflush
    #else
        // The standard ARM private syscall for cacheflush is 0xf0002
        #define __NR_cacheflush 0xf0002
    #endif
#endif

/* ---- geometry & sizing ---- */
#define ITERATIONS   4096
#define LOOP_REPS    10

/* 256 MiB arena (64M 32-bit words) */
#define ARRAY_SIZE   (1024u * 1024u * 64u)
#define ARENA_BYTES  (ARRAY_SIZE * sizeof(uint32_t))

/* one-line safety margin used in some patterns */
#define P7_SAFETY_MARGIN 1

/* ---------------- L1 geometry (Cortex-A9) ---------------- */
#define L1_SET_COUNT    256U
#define L1_LINE_BYTES   32U
#define L1_SET_STRIDE   0x2000U
#define L1_INDEX_SHIFT  5

/* ---------------- L2 geometry (PL310) -------------------- */
#define L2_LINE_BYTES   32U
#define L2_SET_COUNT    4096U
#define L2_SET_STRIDE   (L2_LINE_BYTES * L2_SET_COUNT)   /* 128 KiB */

/* ---------------- DDR addressing (info only) ------------- */
#define DDR_BASE_PHYS   0x00100000UL
#define COL_SHIFT       5
#define COL_BITS        7
#define BANK_SHIFT      (COL_SHIFT + COL_BITS)
#define BANK_BITS       2
#define ROW_SHIFT       (BANK_SHIFT + BANK_BITS)
#define ROW_BITS        (32 - ROW_SHIFT)
#define COL_MASK        ((1u << COL_BITS)  - 1u)
#define BANK_MASK       ((1u << BANK_BITS) - 1u)
#define ROW_MASK        ((1u << ROW_BITS)  - 1u)


#ifndef CACHELINE_SIZE
#define CACHELINE_SIZE    32U
#endif

#define CACHELINE_MASK        (CACHELINE_SIZE - 1U)
#define CACHELINE_ALIGN_DOWN(p)  ((uintptr_t)(p) & ~((uintptr_t)CACHELINE_MASK))
#define CACHELINE_ALIGN_UP(n)    (((size_t)(n) + CACHELINE_MASK) & ~((size_t)CACHELINE_MASK))

/* ---------------- arena binding (Linux) ------------------ */
/* In Linux, we DON'T allocate 256 MiB in .bss.
 * We bind mem_array to your mmap()'d shared window. */
extern uint32_t *mem_array;
void membench_bind(void *base, size_t bytes);

/* For legacy code that used MEM_BASE */
#define MEM_BASE ((uintptr_t)mem_array)

/* ---------------- LCG RNG (deterministic, legacy) -------- */
uint32_t lcg_rand(void);
const uint8_t random_det_value(void);

/* ================================================================
 *  OS-quality random address & value  (membench_rand.c)
 *
 *  Uses /dev/urandom + anonymous mmap arena (2 MiB).
 *  Call membench_rand_init() once at startup, _deinit() at exit.
 * ================================================================ */
int               membench_rand_init(void);
void              membench_rand_deinit(void);
volatile uint8_t *rand_target_addr(void);     /* random cache-line-aligned addr in arena */
uint8_t           rand_write_byte(volatile uint8_t *addr);  /* write random byte, flush to DRAM */
uint8_t           rand_byte(void);            /* just one random byte                       */

/* ---------------- address helpers (deterministic, legacy) - */
uint8_t *random_addr_ddr(void);
uint8_t *random_addr_ddr_same_bank(uint32_t bank_id);
uint8_t *random_addr_in_l2_set(size_t set_raw);
uint8_t *random_addr_in_l1_set(size_t set_raw);

/* ---------------- memory access (deterministic, legacy) --- */
uint8_t write_random_byte_dram(const uint8_t *addr);

void warmup_target_addrs(void);

/* ---------------- timed access primitive ------------ */
void warmup_probe_icache(void);
void streaming_triggered_ldrb(const uint8_t *base);
void streaming_triggered_ldrb_unpriv(const uint8_t *base);
void membench_set_probe_unpriv(int enabled);
void membench_set_probe_barriers(int enabled);

/* Select single-tap (1) or double-tap (0) for timed loads */
void membench_set_single_tap(int enable);

/* enqueue the timed load (FIFO callback wrapper)       */
void membench_queue_timed_load(int fifo_end, const uint8_t *ptr);

/* patterns */
void *cache_hit_l1_given_ptr (int fifo_end, const uint8_t *target_ptr);
void *cache_hit_l2_given_ptr (int fifo_end, const uint8_t *target_ptr);
void *cache_miss_given_ptr   (int fifo_end, const uint8_t *target_ptr);
void *memory_hit_noL1_given_ptr(int fifo_end, const uint8_t *target_ptr);

/* dispatcher used by sca.c */
void memory_bench_run(int pattern_id,
                      size_t cache_hit_index,
                      int end,
                      const uint8_t *direct_ptr);  /* optional */

#endif /* MEMBENCH_H */
