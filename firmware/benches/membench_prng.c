#include "include/membench.h"

/* Single definition of the LCG state */
uint32_t lcg_seed = 123456789u;

/* Simple LCG (glibc-like modulus behavior masked to 31 bits) */
uint32_t lcg_rand(void)
{
    lcg_seed = (1103515245u * lcg_seed + 12345u) & 0x7FFFFFFFu;
    return lcg_seed;
}

const uint8_t random_det_value(void)
{
    /* Take the lowest 8 bits from the LCG output */
    return (uint8_t)(lcg_rand() & 0xFFu);
}

