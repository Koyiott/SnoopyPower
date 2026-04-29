#pragma once
#include <stdint.h>
typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t  s32;
typedef int64_t  s64;

typedef uintptr_t UINTPTR;

// NULL
#ifndef NULL
#define NULL ((void*)0)
#endif


#ifndef XIL_COMPONENT_IS_READY
#define XIL_COMPONENT_IS_READY 1U
#endif

#ifndef XIL_COMPONENT_IS_STARTED
#define XIL_COMPONENT_IS_STARTED 0x22222222U
#endif
