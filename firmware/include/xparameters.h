#ifndef XPARAMETERS_H
#define XPARAMETERS_H

/* Platform hint (optional) */
#define PLATFORM_ZYNQ 1

/* PS UART (optional; used by some printf helpers) */
#define STDIN_BASEADDRESS  0xE0001000U
#define STDOUT_BASEADDRESS 0xE0001000U

/* ---------- FIFO_AND_CTRL_0 (your FIFO IP) ---------- */
#define XPAR_XFIFO_NUM_INSTANCES            1
#define XPAR_FIFO_AND_CTRL_0_DEVICE_ID      0
#define XPAR_FIFO_AND_CTRL_0_S_AXI_BASEADDR 0x43C00000U
#define XPAR_FIFO_AND_CTRL_0_S_AXI_HIGHADDR 0x43C0FFFFU
/* Fill with your real generics: */
#define XPAR_FIFO_AND_CTRL_0_DEPTH_G        8192
#define XPAR_FIFO_AND_CTRL_0_WIDTH_G        32

/* Canonical (some drivers expect XPAR_XFIFO_0_*) */
#define XPAR_XFIFO_0_DEVICE_ID              XPAR_FIFO_AND_CTRL_0_DEVICE_ID
#define XPAR_XFIFO_0_S_AXI_BASEADDR         XPAR_FIFO_AND_CTRL_0_S_AXI_BASEADDR
#define XPAR_XFIFO_0_S_AXI_HIGHADDR         XPAR_FIFO_AND_CTRL_0_S_AXI_HIGHADDR
#define XPAR_XFIFO_0_depth_g                XPAR_FIFO_AND_CTRL_0_DEPTH_G
#define XPAR_XFIFO_0_width_g                XPAR_FIFO_AND_CTRL_0_WIDTH_G

/* ---------- TDC_BANK_0 (your TDC IP) ---------- */
#define XPAR_XTDC_NUM_INSTANCES             1
#define XPAR_TDC_BANK_0_DEVICE_ID           0
#define XPAR_TDC_BANK_0_S_AXI_BASEADDR      0x43C10000U
#define XPAR_TDC_BANK_0_S_AXI_HIGHADDR      0x43C1FFFFU
#define XPAR_TDC_BANK_0_DEPTH_G             8
#define XPAR_TDC_BANK_0_COUNT_G             8

/* Canonical */
#define XPAR_XTDC_0_DEVICE_ID               XPAR_TDC_BANK_0_DEVICE_ID
#define XPAR_XTDC_0_S_AXI_BASEADDR          XPAR_TDC_BANK_0_S_AXI_BASEADDR
#define XPAR_XTDC_0_S_AXI_HIGHADDR          XPAR_TDC_BANK_0_S_AXI_HIGHADDR
#define XPAR_XTDC_0_depth_g                 XPAR_TDC_BANK_0_DEPTH_G
#define XPAR_XTDC_0_count_g                 XPAR_TDC_BANK_0_COUNT_G

#endif /* XPARAMETERS_H */

