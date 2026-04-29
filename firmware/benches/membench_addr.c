#include "include/membench.h"
#include "include/shared_region.h"
#include "include/xil_cache.h"

/* local 32-bit xorshift used only in this file */
static inline uint32_t prng32(void)
{
    static uint32_t s = 0xCAFEBABE;
    s ^= s << 13;  s ^= s >> 17;  s ^= s << 5;
    return s;
}

/* pack <row,bank,col32> to arena address */
static inline uint8_t *pack_phys(uint32_t row, uint32_t bank, uint32_t col32)
{
    uintptr_t off = (((uintptr_t)row   << ROW_SHIFT)  |
                     ((uintptr_t)bank  << BANK_SHIFT) |
                     ((uintptr_t)col32 << COL_SHIFT)) & (ARENA_BYTES - 1);

    uint64_t addr64 = (uint64_t)MEM_BASE + off;
    if (addr64 >= (uint64_t)MEM_BASE + ARENA_BYTES)
        addr64 -= ARENA_BYTES;

    return (uint8_t *)(uintptr_t)addr64;
}

/* Random DDR address (32B aligned) */
uint8_t *random_addr_ddr(void)
{
    uint32_t row   = prng32() & ROW_MASK;
    uint32_t bank  = prng32() & BANK_MASK;
    uint32_t col32 = prng32() & COL_MASK;
    return pack_phys(row, bank, col32);
}

/* Random DDR address constrained to bank */
uint8_t *random_addr_ddr_same_bank(uint32_t bank_id)
{
    bank_id &= BANK_MASK;
    uint32_t row   = prng32() & ROW_MASK;
    uint32_t col32 = prng32() & COL_MASK;
    return pack_phys(row, bank_id, col32);
}

/* Choose an address that maps to the given L2 set (0..4095). */
uint8_t *random_addr_in_l2_set(size_t set_raw)
{
    /* if user passed 0, randomize a usable baseline within arena */
    set_raw = (set_raw == 0)
        ? (lcg_rand() % (ARRAY_SIZE - P7_SAFETY_MARGIN))
        : set_raw;

    const uint32_t set_idx = (uint32_t)set_raw & 0xFFFu;

    uintptr_t base = (uintptr_t)mem_array;
    unsigned  cur  = (unsigned)((base >> 5) & 0xFFFu);      /* set of base */
    uintptr_t index_off = ((set_idx - cur) & 0xFFFu) << 5;  /* 32×delta    */

    while (1) {
        /* max tag so (index_off + tag*stride + line) stays in arena */
        uint32_t max_tag = (ARENA_BYTES - L2_LINE_BYTES - index_off) / L2_SET_STRIDE;
        if (max_tag == 0) max_tag = 1;

        uintptr_t off = index_off
                      + (uintptr_t)(lcg_rand() % max_tag) * L2_SET_STRIDE
                      + (uintptr_t)(lcg_rand() & (L2_LINE_BYTES - 1));

        off &= (ARENA_BYTES - 1);
        uintptr_t addr = base + off;

        if (((addr >> 5) & 0xFFFu) == set_idx) {
            /* optional trace */
            // printf("random_addr_in_l2_set: want %u -> %p (set %u)\n",
            //        set_idx, (void*)addr, (unsigned)((addr >> 5) & 0xFFFu));
            return (uint8_t *)addr;
        }
    }
}

uint8_t *random_addr_in_l1_set(size_t set_raw)
{
    //printf("random_addr_in_l1_set: set_raw = %zu\n", set_raw);

    /* 1. choose target index -------------------------------------- */
    set_raw = (set_raw == 0)
        ? (lcg_rand() % (ARRAY_SIZE - P7_SAFETY_MARGIN))
        : set_raw;

    const uint32_t set_idx = (uint32_t)set_raw & 0xFF;   /* 0-255 */

    uintptr_t base = (uintptr_t)mem_array;
    unsigned  cur  = (base >> 5) & 0xFF;

    uintptr_t index_off = ((set_idx - cur) & 0xFF) << 5; /* 32 × delta */

    while (1) {
        uint32_t max_tag = (ARENA_BYTES - L1_LINE_BYTES - index_off)
                           / L1_SET_STRIDE;
        if (max_tag == 0) max_tag = 1;

        uintptr_t off  = index_off
                       + (uintptr_t)(lcg_rand() % max_tag) * L1_SET_STRIDE
                       + (uintptr_t)(lcg_rand() & (L1_LINE_BYTES - 1));

        off &= ARENA_BYTES - 1;                 /* wrap */

        uintptr_t addr = base + off;

        if (((addr >> 5) & 0xFF) == set_idx) {
            /* --- NOW it is safe to print ------------------------- */
            printf("random_addr_in_l1_set: want %u  ->  %p (set %u)\n",
                   set_idx, (void *)addr, (unsigned)((addr >> 5) & 0xFF));
            return (uint8_t *)addr; // uint8_t * is the only C type that promises “this exact byte address, no hidden scaling, no alignment assumptions.”
        }
    }
}

uint8_t write_random_byte_dram(const uint8_t *addr)
{
    // STATIC VALUE SELECTION:
    const uint8_t static_val = 0x63;

    // 1. Pre-write barrier (Data Synchronization Barrier)
    __asm__ volatile("dsb sy" ::: "memory");

    // 2. Write into L1 Cache (Dirty)
    *(volatile uint8_t *)addr = static_val;

    // 3. Post-write barrier
    __asm__ volatile("dsb sy" ::: "memory");

    // 4. Flush to DRAM via Xilinx cache API (clean+invalidate L1+L2)
    Xil_DCacheFlushRange((INTPTR)addr, CACHELINE_SIZE);

    // 5. Final barrier to ensure completion
    __asm__ volatile("dsb sy" ::: "memory");

    return static_val;
}

/**
 * Warm up L2 cache with all predefined target addresses.
 *
 * Reads one byte from each target address to bring the line into cache,
 * then flushes L1 to ensure lines are visible in shared L2.
 */
void warmup_target_addrs(void)
{
    size_t table_size = sizeof(TARGET_ADDRS_RANDOM_SPREAD) / sizeof(TARGET_ADDRS_RANDOM_SPREAD[0]);
    volatile uint8_t tmp;

    for (size_t i = 0; i < table_size; i++) {
        const uint8_t *ptr = (const uint8_t *)TARGET_ADDRS_RANDOM_SPREAD[i];
        tmp = *ptr;  /* touch the address to bring it into caches */
    }
    (void)tmp; /* silence unused variable warning */

    /* Full system barrier to complete all loads */
    __asm__ volatile("dsb sy" ::: "memory");

}

/* Baremetal-like local PRNG used for AES-column generation. */
static uint32_t g_aes_col_rng = 132791832u;

static inline uint8_t baremetal_rand_byte(void)
{
    g_aes_col_rng = 1664525u * g_aes_col_rng + 1013904223u;
    return (uint8_t)((g_aes_col_rng >> 16) & 0xFFu);
}

