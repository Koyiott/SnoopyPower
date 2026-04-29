#ifndef PMU_H
#define PMU_H

#include <stdint.h>

#define PMU_EVT_COHERENT_LINEFILL_HIT 0x51  /* Peer-L1 snoop via SCU */

void pmu_enable_user_access(void);
void pmu_init(uint32_t enable_mask);
void pmu_configure_event_counter(uint32_t counter_idx, uint32_t event_code);
uint32_t pmu_read_event_counter(uint32_t counter_idx);
uint32_t pmu_read_cycle_counter(void);

void pmu_setup_peerL1_ctr(unsigned idx);

#endif // PMU_H
