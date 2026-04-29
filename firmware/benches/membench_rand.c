/*
 * membench_rand.c — Cryptographic-quality random address & value generation
 *
 * Uses /dev/urandom for all randomness (no LCG, no xorshift).
 * Allocates a large anonymous mmap() arena at init so that random
 * addresses naturally spread across all L1 sets (256) and L2 sets (4096).
 *
 * Arena: 2 MiB anonymous, page-aligned, read/write.
 *        — covers all 4096 L2 sets (set stride = 128 KiB)
 *        — covers all 256  L1 sets  (set stride = 8 KiB)
 *
 * Call membench_rand_init() once at startup (from hw_init).
 * Call membench_rand_deinit() once at shutdown (from hw_deinit).
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syscall.h>

#include "include/membench.h"
#include "include/xil_cache.h"

/* ================================================================
 *  Arena configuration
 * ================================================================ */
#define RAND_ARENA_BYTES   (2u * 1024u * 1024u)   /* 2 MiB */
#define RAND_CACHE_LINE    32u

static volatile uint8_t *g_rand_arena  = NULL;
static size_t             g_rand_arena_sz = 0;
static int                g_urandom_fd = -1;


/* ================================================================
 *  Core: read exactly `len` bytes from /dev/urandom
 *
 *  This is the ONLY source of randomness in the file.
 *  /dev/urandom never blocks, is seeded from HW entropy on boot,
 *  and is the standard Linux choice for non-blocking crypto random.
 * ================================================================ */
static int rand_fill(void *buf, size_t len)
{
    if (g_urandom_fd < 0)
        return -1;

    uint8_t *p   = (uint8_t *)buf;
    size_t   rem = len;
    while (rem > 0) {
        ssize_t n = read(g_urandom_fd, p, rem);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("[membench_rand] read /dev/urandom");
            return -1;
        }
        p   += (size_t)n;
        rem -= (size_t)n;
    }
    return 0;
}


/* ================================================================
 *  Init / Deinit
 * ================================================================ */

int membench_rand_init(void)
{
    /* 1. Open /dev/urandom (kept open for the process lifetime) */
    g_urandom_fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (g_urandom_fd < 0) {
        perror("[membench_rand] open /dev/urandom");
        return -1;
    }

    /* 2. Allocate the anonymous arena (2 MiB, page-aligned) */
    void *p = mmap(NULL, RAND_ARENA_BYTES,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE,
                   -1, 0);
    if (p == MAP_FAILED) {
        perror("[membench_rand] mmap arena");
        close(g_urandom_fd);
        g_urandom_fd = -1;
        return -1;
    }

    g_rand_arena    = (volatile uint8_t *)p;
    g_rand_arena_sz = RAND_ARENA_BYTES;

    /* 3. Pre-fault and randomize the whole arena content so that
     *    even the initial DRAM-resident data is unpredictable.    */
    rand_fill((void *)g_rand_arena, RAND_ARENA_BYTES);

    printf("[membench_rand] arena %p .. %p (%u KiB), /dev/urandom fd=%d\n",
           (void *)g_rand_arena,
           (void *)(g_rand_arena + RAND_ARENA_BYTES),
           RAND_ARENA_BYTES / 1024, g_urandom_fd);

    return 0;
}

void membench_rand_deinit(void)
{
    if (g_rand_arena) {
        munmap((void *)g_rand_arena, g_rand_arena_sz);
        g_rand_arena    = NULL;
        g_rand_arena_sz = 0;
    }
    if (g_urandom_fd >= 0) {
        close(g_urandom_fd);
        g_urandom_fd = -1;
    }
}


/* ================================================================
 *  rand_target_addr — pick a random cache-line-aligned address
 *                     inside the 2 MiB arena.
 *
 *  Returns a fresh random pointer every call.
 *  The address is 32-byte aligned (Cortex-A9 cache line).
 *  The caller can add a sub-line offset if desired.
 * ================================================================ */
volatile uint8_t *rand_target_addr(void)
{
    if (!g_rand_arena) {
        fprintf(stderr, "[membench_rand] arena not initialized!\n");
        return NULL;
    }

    /* Draw 4 random bytes → uniform offset within arena */
    uint32_t raw;
    rand_fill(&raw, sizeof(raw));

    /* Align down to cache-line boundary */
    uint32_t max_offset = (uint32_t)(g_rand_arena_sz - RAND_CACHE_LINE);
    uint32_t offset = (raw % (max_offset / RAND_CACHE_LINE)) * RAND_CACHE_LINE;

    return g_rand_arena + offset;
}


/* ================================================================
 *  rand_write_byte — write a random byte to `addr`, flush to DRAM.
 *
 *  Uses /dev/urandom for the value.
 *  Returns the byte written (for logging / ground-truth labels).
 * ================================================================ */
uint8_t rand_write_byte(volatile uint8_t *addr)
{
    /* 1. Draw one random byte */
    uint8_t val;
    rand_fill(&val, 1);

    /* 2. Compiler + memory barrier */
    __asm__ volatile("dsb sy" ::: "memory");

    /* 3. Store into cache (will be dirty in L1) */
    *addr = val;

    /* 4. Barrier */
    __asm__ volatile("dsb sy" ::: "memory");

    /* 5. Flush the containing cache line to DRAM via Xilinx cache API.
     *    Xil_DCacheFlushRange handles alignment internally.         */
    Xil_DCacheFlushRange((INTPTR)addr, RAND_CACHE_LINE);

    /* 6. Final barrier */
    __asm__ volatile("dsb sy" ::: "memory");

    return val;
}


/* ================================================================
 *  rand_read_byte — convenience: read one random byte from
 *                   /dev/urandom (not tied to any address).
 * ================================================================ */
uint8_t rand_byte(void)
{
    uint8_t val;
    rand_fill(&val, 1);
    return val;
}