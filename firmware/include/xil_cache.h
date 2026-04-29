/* xil_cache.h — Linux userspace adaptation of the Xilinx Cortex-A9 cache API
 *
 * SAFE APPROACH (no direct PL310 register access):
 *
 *   L1 (CP15):  Privileged (EL1 only) — displacement buffer walk.
 *   L2 (PL310): Privileged IRQ masking required — displacement buffer walk.
 *   Range ops:  → __NR_cacheflush syscall (kernel handles L1+L2 safely).
 *
 * Call Xil_CacheInit()   once at startup  (from hw_init).
 * Call Xil_CacheDeinit() once at shutdown (from hw_deinit).
 *
 * Target: Cortex-A9, PynqZ1 (Zynq-7020), PYNQ Linux image.
 * SPDX-License-Identifier: MIT
 */
#ifndef XIL_CACHE_H
#define XIL_CACHE_H

#include <stdint.h>
#include <stddef.h>
#include "xil_types.h"

/* Xilinx upstream defines INTPTR as s32 on 32-bit ARM.
 * Some stripped-down xil_types.h only have UINTPTR — provide fallback. */
#ifndef INTPTR
typedef int INTPTR;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Lifecycle (Linux-specific) ---- */
int  Xil_CacheInit(void);
void Xil_CacheDeinit(void);

/* ---- D-Cache: full cache ---- */
void Xil_DCacheFlush(void);
void Xil_DCacheInvalidate(void);

/* ---- D-Cache: range ---- */
void Xil_DCacheFlushRange(INTPTR adr, u32 len);
void Xil_DCacheInvalidateRange(INTPTR adr, u32 len);
void Xil_DCacheFlushLine(u32 adr);
void Xil_DCacheInvalidateLine(u32 adr);

/* ---- L1 D-Cache only ---- */
void Xil_L1DCacheFlush(void);
void Xil_L1DCacheInvalidate(void);

/* ---- L2 Cache ---- */
void Xil_L2CacheFlush(void);
void Xil_L2CacheInvalidate(void);

/* ---- I-Cache ---- */
void Xil_ICacheInvalidate(void);
void Xil_ICacheInvalidateRange(INTPTR adr, u32 len);

/* ---- Enable/Disable (no-ops — kernel manages caches on Linux) ---- */
static inline void Xil_DCacheEnable(void)    { /* no-op */ }
static inline void Xil_DCacheDisable(void)   { /* no-op */ }
static inline void Xil_ICacheEnable(void)    { /* no-op */ }
static inline void Xil_ICacheDisable(void)   { /* no-op */ }
static inline void Xil_L1DCacheEnable(void)  { /* no-op */ }
static inline void Xil_L1DCacheDisable(void) { /* no-op */ }

#ifdef __cplusplus
}
#endif

#endif /* XIL_CACHE_H */