#define _GNU_SOURCE
#include "mmio_linux.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

int mmio_map(mmio_t *m, uintptr_t pbase, size_t size) {
    memset(m, 0, sizeof *m);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) return -1;

    size_t pagesz     = (size_t)sysconf(_SC_PAGESIZE);
    uintptr_t pagebase = pbase & ~(pagesz - 1);
    size_t   pageoff   = (size_t)(pbase - pagebase);
    size_t   mapsize   = pageoff + size;

    void *addr = mmap(NULL, mapsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, pagebase);
    close(fd);
    if (addr == MAP_FAILED) return -1;

    m->vbase = (volatile uint8_t *)addr + pageoff;
    m->size  = size;
    m->pbase = pbase;
    return 0;
}

void mmio_unmap(mmio_t *m) {
    if (!m || !m->vbase) return;
    size_t pagesz     = (size_t)sysconf(_SC_PAGESIZE);
    uintptr_t pagebase = m->pbase & ~(pagesz - 1);
    size_t   pageoff   = (size_t)(m->pbase - pagebase);
    munmap((void *)(m->vbase - pageoff), pageoff + m->size);
    m->vbase = NULL; m->size = 0; m->pbase = 0;
}

