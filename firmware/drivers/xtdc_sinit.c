#include "xtdc.h"
#include "xparameters.h"

/* Defaults if Vivado didn't generate them */
#ifndef XPAR_TDC_BANK_0_DEVICE_ID
#  define XPAR_TDC_BANK_0_DEVICE_ID 0
#endif

#ifndef XPAR_TDC_BANK_0_S_AXI_BASEADDR
#  error "XPAR_TDC_BANK_0_S_AXI_BASEADDR must be defined in xparameters.h"
#endif

#ifndef XPAR_TDC_BANK_0_DEPTH
#  define XPAR_TDC_BANK_0_DEPTH 16u
#endif

#ifndef XPAR_TDC_BANK_0_CHANNEL_COUNT
#  define XPAR_TDC_BANK_0_CHANNEL_COUNT 8u
#endif

XTDC_Config XTDC_ConfigTable[] = {
    {
        XPAR_TDC_BANK_0_DEVICE_ID,
        (u32)XPAR_TDC_BANK_0_S_AXI_BASEADDR,
        (u32)XPAR_TDC_BANK_0_DEPTH,
        (u32)XPAR_TDC_BANK_0_CHANNEL_COUNT
    }
};

XTDC_Config* XTDC_LookupConfig(u32 DeviceId)
{
    for (unsigned i = 0; i < sizeof(XTDC_ConfigTable)/sizeof(XTDC_ConfigTable[0]); i++) {
        if (XTDC_ConfigTable[i].DeviceId == DeviceId)
            return &XTDC_ConfigTable[i];
    }
    return NULL;
}
