#include "pmu.h"
#include <stdint.h>


void pmu_enable_user_access(void) {
    asm volatile (
        "MRC p15, 0, r0, c9, c14, 0\n"
        "ORR r0, r0, #1\n"
        "MCR p15, 0, r0, c9, c14, 0\n"
        :::"r0"
    );
}

void pmu_init(uint32_t enable_mask) {
    asm volatile (
        "MRC p15, 0, r0, c9, c12, 0\n"
        "ORR r0, r0, #(1 << 0)\n"
        "ORR r0, r0, #(1 << 1)\n"
        "ORR r0, r0, #(1 << 2)\n"
        "MCR p15, 0, r0, c9, c12, 0\n"
        "MCR p15, 0, %0, c9, c12, 1\n"
        :
        : "r"(enable_mask)
        : "r0"
    );
}

void pmu_configure_event_counter(uint32_t counter_idx, uint32_t event_code) {
    asm volatile (
        "MCR p15, 0, %0, c9, c12, 5\n"
        "MCR p15, 0, %1, c9, c13, 1\n"
        :
        : "r"(counter_idx), "r"(event_code)
    );
}

uint32_t pmu_read_event_counter(uint32_t counter_idx) {
    asm volatile (
        "MCR p15, 0, %0, c9, c12, 5\n" :: "r"(counter_idx)
    );
    uint32_t val;
    asm volatile (
        "MRC p15, 0, %0, c9, c13, 2\n" : "=r"(val)
    );
    return val;
}

uint32_t pmu_read_cycle_counter(void) {
    uint32_t val;
    asm volatile ("MRC p15, 0, %0, c9, c13, 0\n" : "=r"(val));
    return val;
}

static inline int actlr_smp(uint32_t v){ return (v >> 6) & 1; } // bit6 = SMP

void pmu_setup_peerL1_ctr(unsigned idx) {
    pmu_enable_user_access();
    pmu_init(1u << idx);
    pmu_configure_event_counter(idx, PMU_EVT_COHERENT_LINEFILL_HIT);
}