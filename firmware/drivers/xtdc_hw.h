#pragma once
#include "xil_types.h"
#include <stdint.h>
#include "xil_io.h"

#define XTDC_WORD_SIZE 4
#define XTDC_DATA_OFFSET    0x00
#define XTDC_STATE_OFFSET   0x04
#define XTDC_SEL_OFFSET     0x08
#define XTDC_COARSE_OFFSET  0x0C
#define XTDC_FINE_OFFSET    0x10

#define XTDC_DEFAULT_CALIBRATE_IT 8192
#define XTDC_CALIBRATE_TARGET     16
#define XTDC_COARSE_MAX           0x3
#define XTDC_FINE_MAX             0xF

#define XTDC_Delay_64(fine, coarse) ((((uint64_t)(coarse) << 32) | ((uint64_t)(fine) & 0xffffffff)))
#define XTDC_Fine_Mask(id)   ~((uint32_t)(0x0000000F << (4 * (id))))
#define XTDC_Coarse_Mask(id) ~((uint32_t)(0x00000003 << (2 * (id))))
#define XTDC_Weight_Mask(id) ~((uint32_t)(0x000000FF << (8 * (id))))
#define XTDC_Weight(weights, id) ((uint32_t)(((weights) & ~XTDC_Weight_Mask(id)) >> (8 * (id))))

#define XTDC_ReadReg(addr, off)         Xil_In32((addr) + (off))
#define XTDC_WriteReg(addr, off, data)  Xil_Out32((addr) + (off), (data))
#define DIST(a,b) ((a)>(b)?(a)-(b):(b)-(a))

