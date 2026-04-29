/* shared_region.h — common shared DDR window layout (both cores / Linux userspace)
 *
 * Window:  .shared_ddr = 0x3FF0_0000 .. 0x3FFF_FFFF (1 MiB)
 * Policy:  We start targets at +0x1000 to avoid metadata at the very beginning.
 * Notes:   Table holds PHYSICAL addresses; we translate using the bound mapping.
 */
#ifndef SHARED_REGION_H
#define SHARED_REGION_H

#include <stdint.h>
#include <stddef.h>   /* size_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------- Window configuration ---------------- */
#define SHARED_BASE_1M     0x3FF00000u
#define SHARED_SIZE_1M     0x00100000u  /* 1 MiB */
#define SHARED_END_1M_EXCL (SHARED_BASE_1M + SHARED_SIZE_1M)

/* Start targets a bit into the window to avoid any metadata page */
#define TARGET_REGION_BASE (SHARED_BASE_1M + 0x00001000u)  /* +4 KiB */

/* Optional: typical Cortex-A9 D-cache line is 32B */
#define A9_DCACHE_LINE_BYTES 32u

/* ---------------- Target address table (PHYSICAL) ---------------- */
static const uintptr_t TARGET_ADDRS_RANDOM_SPREAD[64] = {
    /* hidden special #1 */
    TARGET_REGION_BASE + 0x00000100u,

    /* spread set */
    TARGET_REGION_BASE + 0x00000560u,
    TARGET_REGION_BASE + 0x00000A40u,
    TARGET_REGION_BASE + 0x00000F40u,
    TARGET_REGION_BASE + 0x00001280u,
    TARGET_REGION_BASE + 0x000015E0u,
    TARGET_REGION_BASE + 0x000018C0u,
    TARGET_REGION_BASE + 0x00001AA0u,
    TARGET_REGION_BASE + 0x00001D20u,
    TARGET_REGION_BASE + 0x00002160u,
    TARGET_REGION_BASE + 0x00002460u,
    TARGET_REGION_BASE + 0x000027A0u,
    TARGET_REGION_BASE + 0x00002BA0u,
    TARGET_REGION_BASE + 0x00002E40u,
    TARGET_REGION_BASE + 0x000032C0u,
    TARGET_REGION_BASE + 0x000034E0u,
    TARGET_REGION_BASE + 0x000038A0u,

    /* hidden special #2 */
    TARGET_REGION_BASE + 0x00003B20u,

    TARGET_REGION_BASE + 0x00003DA0u,
    TARGET_REGION_BASE + 0x00003F60u,
    TARGET_REGION_BASE + 0x000042C0u,
    TARGET_REGION_BASE + 0x00004500u,
    TARGET_REGION_BASE + 0x000048E0u,
    TARGET_REGION_BASE + 0x00004BE0u,
    TARGET_REGION_BASE + 0x00005120u,
    TARGET_REGION_BASE + 0x00005220u,
    TARGET_REGION_BASE + 0x00005940u,
    TARGET_REGION_BASE + 0x00005A60u,
    TARGET_REGION_BASE + 0x000061E0u,
    TARGET_REGION_BASE + 0x000065A0u,
    TARGET_REGION_BASE + 0x00006AE0u,
    TARGET_REGION_BASE + 0x00006C20u,
    TARGET_REGION_BASE + 0x00007120u,
    TARGET_REGION_BASE + 0x000074C0u,
    TARGET_REGION_BASE + 0x00007BA0u,
    TARGET_REGION_BASE + 0x00007E80u,
    TARGET_REGION_BASE + 0x000083E0u,
    TARGET_REGION_BASE + 0x00008620u,
    TARGET_REGION_BASE + 0x00008C20u,
    TARGET_REGION_BASE + 0x000092C0u,
    TARGET_REGION_BASE + 0x00009760u,
    TARGET_REGION_BASE + 0x00009F60u,
    TARGET_REGION_BASE + 0x0000A1E0u,
    TARGET_REGION_BASE + 0x0000A9A0u,
    TARGET_REGION_BASE + 0x0000AC60u,
    TARGET_REGION_BASE + 0x0000B540u,
    TARGET_REGION_BASE + 0x0000B9A0u,
    TARGET_REGION_BASE + 0x0000C120u,
    TARGET_REGION_BASE + 0x0000C4E0u,
    TARGET_REGION_BASE + 0x0000CD80u,
    TARGET_REGION_BASE + 0x0000D120u,
    TARGET_REGION_BASE + 0x0000DA20u,
    TARGET_REGION_BASE + 0x0000DBE0u,
    TARGET_REGION_BASE + 0x0000E420u,
    TARGET_REGION_BASE + 0x0000E6C0u,
    TARGET_REGION_BASE + 0x0000EFE0u,
    TARGET_REGION_BASE + 0x0000F320u,
    TARGET_REGION_BASE + 0x0000F760u,
    TARGET_REGION_BASE + 0x0000FAE0u,
    TARGET_REGION_BASE + 0x0000FEA0u,
    TARGET_REGION_BASE + 0x0000FF20u,
    TARGET_REGION_BASE + 0x0000FF40u,
    TARGET_REGION_BASE + 0x0000FF80u,
    TARGET_REGION_BASE + 0x0000FFC0u
};

#define TARGET_ADDRS_COUNT  (sizeof(TARGET_ADDRS_RANDOM_SPREAD)/sizeof(TARGET_ADDRS_RANDOM_SPREAD[0]))

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(TARGET_ADDRS_COUNT == 64, "TARGET_ADDRS_RANDOM_SPREAD must have 64 entries");
#endif

/* ---------------- Bound mapping (set once at init) ---------------- */
extern volatile uint8_t *g_shared_base;
extern size_t            g_shared_size;

/* Bind the mapping once (call this in hw_init after mmio_map of shared window) */
static inline void shared_region_bind(void *base, size_t size) {
    g_shared_base = (volatile uint8_t *)base;
    g_shared_size = size;
}

/* Quick range check helper for PHYS address */
static inline int shared_phys_in_range(uintptr_t phys) {
    return (phys >= (uintptr_t)SHARED_BASE_1M) && (phys < (uintptr_t)SHARED_END_1M_EXCL);
}

/* deterministic_target_ptr — 1-arg version (index only)
 * Uses the globally bound shared mapping (g_shared_base/g_shared_size),
 * set once via shared_region_bind() during hw_init().
 */
static inline volatile uint8_t *deterministic_target_ptr(int targeted_index)
{
    /* Ensure the shared window is bound */
    if (!g_shared_base || g_shared_size < SHARED_SIZE_1M)
        return NULL;

    /* Select entry (wrap over table size) */
    const size_t n   = TARGET_ADDRS_COUNT;
    const size_t idx = (size_t)targeted_index % n;
    const uintptr_t phys = TARGET_ADDRS_RANDOM_SPREAD[idx];

    /* Sanity checks vs the 1 MiB window */
    if (phys < (uintptr_t)SHARED_BASE_1M || phys >= (uintptr_t)SHARED_END_1M_EXCL)
        return NULL;

    const uintptr_t offset = phys - (uintptr_t)SHARED_BASE_1M;
    if (offset >= g_shared_size)
        return NULL;

    /* Translate PHYS → VIRT using the bound base */
    return g_shared_base + offset;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SHARED_REGION_H */
