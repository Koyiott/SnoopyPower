#include "include/membench.h"
#include "xfifo.h"
#include <stdint.h>

/* Arena pointer bound at runtime to the mmap()'d region */
uint32_t *mem_array = NULL;

void membench_bind(void *base, size_t bytes)
{
    (void)bytes;
    mem_array = (uint32_t *)base;
}

static inline void dsb_sy(void) { __asm__ volatile("dsb sy" ::: "memory"); }
static inline void isb_sy(void) { __asm__ volatile("isb" ::: "memory"); }

static int g_probe_use_unpriv = 0;
static int g_probe_use_barriers = 1;
static volatile uint8_t g_probe_warmup_byte = 0xAA;
static volatile const uint8_t *g_target_ptr = NULL;

/* Pass target address via CP15 TPIDRURW (c13,c0,2): no D-cache traffic. */
static inline void store_target_in_cp15_ordered(volatile const uint8_t *addr)
{
    __asm__ volatile("mcr p15, 0, %0, c13, c0, 2" :: "r"(addr));
    __asm__ volatile("isb" ::: "memory");
}

static inline void store_target_in_cp15_relaxed(volatile const uint8_t *addr)
{
    __asm__ volatile("mcr p15, 0, %0, c13, c0, 2" :: "r"(addr));
}

/* Low-jitter one-byte read with fences around it (privileged profile). */
__attribute__((naked,noinline))
void streaming_triggered_ldrb(const uint8_t *base)
{
    __asm__ __volatile__ (
        "eor    r3, r3, r3       \n"
        "add    r3, r0, r3       \n"
        "mov    r1, #100          \n"
        "1: subs r1, r1, #1      \n"
        "   bne  1b              \n"
        "dsb    sy               \n"
        "isb                     \n"
        "ldrb   r2, [r3]         \n"
        "dsb    sy               \n"
        "mov    r1, #100          \n"
        "2: subs r1, r1, #1      \n"
        "   bne  2b              \n"
        "dsb    sy               \n"
        "isb                     \n"
        "bx     lr               \n"
    );
}

/* Low-jitter one-byte read for unprivileged probe profile. */
__attribute__((naked,noinline))
void streaming_triggered_ldrb_unpriv(const uint8_t *base)
{
    __asm__ __volatile__ (
        "mov    r1, #100              \n\t"
        "1:                          \n\t"
        "   subs   r1, r1, #1        \n\t"
        "   bne    1b                \n\t"
        "eor    r12, r12, r12        \n\t"
        "add    r3, r0, r12          \n\t"
        "ldrb   r2, [r3]             \n\t"
        "mov    r1, #8               \n\t"
        "mov    r12, r2              \n\t"
        "2:                          \n\t"
        "   orr    r12, r12, r2      \n\t"
        "   eor    r12, r12, r3      \n\t"
        "   mul    r12, r2, r12      \n\t"
        "   subs   r1, r1, #1        \n\t"
        "   bne    2b                \n\t"
        "eor    r1, r12, r12         \n\t"
        "orr    r1, r1, #100       \n\t"
        "3:                          \n\t"
        "   subs   r1, r1, #1        \n\t"
        "   bne    3b                \n\t"
        "bx     lr                   \n\t"
        :::
    );
}

/* FIFO instance (defined in main_linux.c) — needed by queue functions below. */
extern XFIFO fifo_inst;

/* FIFO action wrappers: fetch target ptr from CP15 and tail-branch. */
__attribute__((naked, noinline))
static void memory_bench_function_priv(void)
{
    __asm__ __volatile__(
        "mrc    p15, 0, r0, c13, c0, 2 \n"
        "b      streaming_triggered_ldrb\n"
    );
}

__attribute__((naked, noinline))
static void memory_bench_function_unpriv(void)
{
    __asm__ __volatile__(
        "mrc    p15, 0, r0, c13, c0, 2 \n"
        "b      streaming_triggered_ldrb_unpriv\n"
    );
}

void membench_set_probe_unpriv(int enabled)
{
    g_probe_use_unpriv = (enabled != 0);
}

void membench_set_probe_barriers(int enabled)
{
    g_probe_use_barriers = (enabled != 0);
}

/* Warm both probe callbacks in I-cache to avoid first-run I-side jitter. */
void warmup_probe_icache(void)
{
    store_target_in_cp15_ordered(&g_probe_warmup_byte);
    dsb_sy();
    isb_sy();
    memory_bench_function_priv();
    memory_bench_function_unpriv();
    dsb_sy();
    isb_sy();
}

/* Enqueue timed load: address passed via CP15, not D-cache. */
void membench_queue_timed_load(int fifo_end, const uint8_t *ptr)
{
    g_target_ptr = ptr;
    if (!g_target_ptr) {
        return;
    }

    if (g_probe_use_barriers) {
        store_target_in_cp15_ordered(g_target_ptr);
    } else {
        store_target_in_cp15_relaxed(g_target_ptr);
    }

    if (g_probe_use_barriers) {
        dsb_sy();
        isb_sy();
    }

    fifo_inst.Mode = XFIFO_MODE_SW;
    XFIFO_Write(
        &fifo_inst, fifo_end,
        g_probe_use_unpriv
            ? (XFIFO_WrAction)memory_bench_function_unpriv
            : (XFIFO_WrAction)memory_bench_function_priv
    );

    if (g_probe_use_barriers) {
        dsb_sy();
        isb_sy();
    }
}
