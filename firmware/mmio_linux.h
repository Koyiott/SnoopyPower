#pragma once
#include <stddef.h>     // size_t
#include <stdint.h>     // uint*_t

typedef struct {
    volatile uint8_t *vbase;  // virtual base pointer to mapped region
    size_t size;              // mapped size
    uintptr_t pbase;          // physical base (for unmap bookkeeping)
} mmio_t;

int  mmio_map(mmio_t *m, uintptr_t pbase, size_t size);  // 0 on success
void mmio_unmap(mmio_t *m);

/* Optional helpers for quick poke/peek inside the mapped window */
static inline uint32_t mmio_read32(const mmio_t *m, size_t off) {
    return *(volatile uint32_t *)(m->vbase + off);
}
static inline void mmio_write32(const mmio_t *m, size_t off, uint32_t v) {
    *(volatile uint32_t *)(m->vbase + off) = v;
}

