/* shared_region.c — storage for the bound mapping */
#include <stddef.h>
#include <stdint.h>
#include "shared_region.h"

volatile uint8_t *g_shared_base = NULL;
size_t            g_shared_size = 0;
