#include "xparameters.h"
#include "xfifo.h"

/*
* The configuration table for devices
*/

XFIFO_Config XFIFO_ConfigTable[XPAR_XFIFO_NUM_INSTANCES] =
{
	{
		XPAR_FIFO_AND_CTRL_0_DEVICE_ID,
		XPAR_FIFO_AND_CTRL_0_S_AXI_BASEADDR,
		XPAR_FIFO_AND_CTRL_0_DEPTH_G,
		XPAR_FIFO_AND_CTRL_0_WIDTH_G
	}
};

XFIFO_Config *XFIFO_LookupConfig(u32 DeviceId)
{
	extern XFIFO_Config XFIFO_ConfigTable[];
	XFIFO_Config *CfgPtr = NULL;
	u32 Index;

	for(Index = 0; Index < XPAR_XFIFO_NUM_INSTANCES; Index++)
	{
		if(XFIFO_ConfigTable[Index].DeviceId == DeviceId)
		{
			CfgPtr = &XFIFO_ConfigTable[Index];
			break;
		}
	}

	return CfgPtr;
}
