#pragma once
#include <stdint.h>
static inline uint32_t Xil_In32(uintptr_t a){ return *(volatile uint32_t*)a; }
static inline void     Xil_Out32(uintptr_t a, uint32_t v){ *(volatile uint32_t*)a = v; }

