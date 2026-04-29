#include "include/membench.h"
#include "include/shared_region.h"
#include "include/xil_cache.h"

static inline void dsb_sy(void) { __asm__ volatile("dsb sy" ::: "memory"); }
static inline void isb_sy(void) { __asm__ volatile("isb" ::: "memory"); }

extern void Xil_L1DCacheFlush(void);

void *cache_hit_l1_given_ptr(int fifo_end, const uint8_t *target_ptr)
{
    dsb_sy();
    Xil_DCacheFlush();
    dsb_sy();
    isb_sy();

    __asm__ volatile("ldrb r6, [%0]" :: "r"(target_ptr) : "r6", "memory");
    dsb_sy();

    membench_queue_timed_load(fifo_end, target_ptr);
    return (void *)target_ptr;
}

void *cache_hit_l2_given_ptr(int fifo_end, const uint8_t *target_ptr)
{
    dsb_sy();
    Xil_DCacheFlush();
    dsb_sy();
    isb_sy();

    __asm__ volatile("ldrb r6, [%0]" :: "r"(target_ptr) : "r6", "memory");
    dsb_sy();

    Xil_L1DCacheFlush();
    dsb_sy();

    membench_queue_timed_load(fifo_end, target_ptr);
    return (void *)target_ptr;
}

void *cache_miss_given_ptr(int fifo_end, const uint8_t *target_ptr)
{
    dsb_sy();
    Xil_DCacheFlush();
    dsb_sy();
    isb_sy();

    membench_queue_timed_load(fifo_end, target_ptr);
    return (void *)target_ptr;
}

void *memory_hit_noL1_given_ptr(int fifo_end, const uint8_t *target_ptr)
{
    dsb_sy();
    isb_sy();
    membench_queue_timed_load(fifo_end, target_ptr);
    dsb_sy();
    isb_sy();
    return (void *)target_ptr;
}

static void *pattern_l1_hit_unpriv(int fifo_end, const uint8_t *ptr)
{
    void *ret;
    membench_set_probe_unpriv(1);
    membench_set_probe_barriers(1);
    ret = cache_hit_l1_given_ptr(fifo_end, ptr);
    membench_set_probe_unpriv(0);
    membench_set_probe_barriers(1);
    return ret;
}

static void *pattern_l2_hit_unpriv(int fifo_end, const uint8_t *ptr)
{
    void *ret;
    membench_set_probe_unpriv(1);
    membench_set_probe_barriers(1);
    ret = cache_hit_l2_given_ptr(fifo_end, ptr);
    membench_set_probe_unpriv(0);
    membench_set_probe_barriers(1);
    return ret;
}

static void *pattern_dram_miss_unpriv(int fifo_end, const uint8_t *ptr)
{
    void *ret;
    membench_set_probe_unpriv(1);
    membench_set_probe_barriers(1);
    ret = cache_miss_given_ptr(fifo_end, ptr);
    membench_set_probe_unpriv(0);
    membench_set_probe_barriers(1);
    return ret;
}

void memory_bench_run(int pattern_id, size_t cache_hit_index,
                      int end, const uint8_t *direct_ptr)
{
    const uint8_t *ptr;

    switch (pattern_id) {
    case 1:
    case 2:
    case 3:
    case 14:
    case 15:
    case 16: {
        volatile uint8_t *target_ptr = rand_target_addr();
        if (!target_ptr) {
            printf("Error: rand_target_addr() failed.\n");
            return;
        }

        (void)rand_write_byte(target_ptr);
        ptr = (const uint8_t *)target_ptr;

        if (pattern_id == 1) {
            cache_hit_l1_given_ptr(end, ptr);
        } else if (pattern_id == 2) {
            cache_hit_l2_given_ptr(end, ptr);
        } else if (pattern_id == 3) {
            cache_miss_given_ptr(end, ptr);
        } else if (pattern_id == 14) {
            pattern_l1_hit_unpriv(end, ptr);
        } else if (pattern_id == 15) {
            pattern_l2_hit_unpriv(end, ptr);
        } else {
            pattern_dram_miss_unpriv(end, ptr);
        }
        break;
    }

    case 10: {
        ptr = direct_ptr ? direct_ptr : (const uint8_t *)deterministic_target_ptr((int)cache_hit_index);
        if (!ptr) {
            printf("Error: deterministic_target_ptr() failed.\n");
            return;
        }
        memory_hit_noL1_given_ptr(end, ptr);
        break;
    }

    default:
        printf("Unknown pattern ID: %d\n", pattern_id);
        break;
    }
}
