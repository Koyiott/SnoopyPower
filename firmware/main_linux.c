/*
 * SnoopyPower — userspace measurement application for the on-chip
 * TDC power sensor running on Linux/PYNQ on a Zynq-7000 SoC.
 *
 * Provides cache-state characterization (L1 hit / L2 hit / DRAM miss)
 * via /dev/mem AXI MMIO. No kernel module required.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <getopt.h>
#include <stddef.h>
#include <inttypes.h>
#include <errno.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifndef __has_include
#  define __has_include(x) 0
#endif

#include "mmio_linux.h"
#include "xil_types.h"
#include "xstatus.h"
#include "xparameters.h"
#include "xil_io.h"
#include "xil_assert.h"
#include "xil_cache.h"

#include "xtdc.h"
#include "xfifo.h"

#ifdef SNOOPYPOWER_MEMORY
#include "membench.h"
#include "shared_region.h"
#endif

/* ---------------------- UI / UX Helpers ---------------------- */
#define COLOR_RESET   "\033[0m"
#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_BOLD    "\033[1m"

/* Simple progress bar: [=====>      ] 45% (ETA: 12s) */
static void ui_progress_bar(size_t current, size_t total, const char *label, time_t start_time) {
    const int bar_width = 40;

    if (total == 0) total = 1;
    double ratio = (double)current / (double)total;
    if (ratio > 1.0) ratio = 1.0;

    int filled_width = (int)(ratio * bar_width);

    time_t now = time(NULL);
    double elapsed = difftime(now, start_time);
    int eta = 0;
    if (ratio > 0.01) {
        eta = (int)(elapsed / ratio) - (int)elapsed;
    }

    printf("\r" COLOR_CYAN "%-10s" COLOR_RESET " [", label);
    for (int i = 0; i < bar_width; ++i) {
        if (i < filled_width) printf("=");
        else if (i == filled_width) printf(">");
        else printf(" ");
    }
    printf("] " COLOR_BOLD "%3.0f%%" COLOR_RESET, ratio * 100.0);

    if (current < total && ratio > 0.0) {
        printf(" (ETA: %ds)  ", eta);
    } else if (current >= total) {
        printf(" (Done)      ");
    } else {
        printf("             ");
    }

    fflush(stdout);
}

/* ---------------------- Global driver instances ---------------------- */
XFIFO fifo_inst;
XTDC  tdc_inst;

static mmio_t mm_fifo = {0};
static mmio_t mm_tdc  = {0};

/* ---- Trace output directory ---- */
#define TRACE_DIR   "traces"
#define TRACE_FILE  TRACE_DIR "/all_traces.csv"

static void ensure_trace_dir(void) {
    struct stat st;
    if (stat(TRACE_DIR, &st) == -1) {
        if (mkdir(TRACE_DIR, 0755) == -1 && errno != EEXIST) {
            perror("mkdir " TRACE_DIR);
        }
    }
}

/* ----------------------------- Banner -------------------------------- */
static void print_banner(void) {
    printf("\n");
    printf(COLOR_CYAN "════════════════════════════════════════════════════════" COLOR_RESET "\n");
    printf(COLOR_BOLD "                    S N O O P Y   P O W E R" COLOR_RESET "\n");
    printf("              On-chip TDC power side-channel sensor\n");
    printf("                 Zynq-7000 / Cortex-A9 / Linux\n");
    printf(COLOR_CYAN "════════════════════════════════════════════════════════" COLOR_RESET "\n\n");
}

/* --------------------------- FIFO helpers (SW mode) ------------------ */
static void fifo_status(void) {
    uint32_t rd = XFIFO_ReadReg(fifo_inst.Config.BaseAddr, XFIFO_STATUS_RD_OFFSET);
    uint32_t wr = XFIFO_ReadReg(fifo_inst.Config.BaseAddr, XFIFO_STATUS_WR_OFFSET);
    uint32_t ct = XFIFO_GetCount(fifo_inst.Config.BaseAddr);
    printf("STATUS_RD=0x%08X STATUS_WR=0x%08X COUNT=%u  (E=%d F=%d R=%d)\n",
           rd, wr, ct,
           !!(rd & XFIFO_STATUS_EMPTY_MASK),
           !!(rd & XFIFO_STATUS_FULL_MASK),
           !!(rd & XFIFO_STATUS_REACHED_MASK));
}
void fifo_flush(void) {
    XFIFO_Reset(&fifo_inst);
}

void fifo_acquire(int end) {
    fifo_inst.Mode = XFIFO_MODE_SW;
    XFIFO_Reset(&fifo_inst);
    XFIFO_Write(&fifo_inst, end, NULL);
}

void fifo_read(int verbose, int start, int end) {
    uint32_t *weights = malloc(32 * (end - start));
    if (!weights) {
        perror("malloc");
        return;
    }
    int len = XFIFO_Read(&fifo_inst, weights, start, end, 1);

    if (len == 0) {
        free(weights);
        return;
    }

    FILE *f = fopen(TRACE_FILE, "a");
    if (!f) {
        perror("fopen " TRACE_FILE);
        free(weights);
        return;
    }

    for (int i = 0; i < len; i++) {
        fprintf(f, "%d", weights[i]);
        if (i < len - 1)
            fprintf(f, ",");
    }
    fprintf(f, "\n");
    fclose(f);

    if (verbose) {
        static long trace_count = 0;
        trace_count++;
        if (trace_count % 1000 == 0) {
            printf("Traces written: %ld\n", trace_count);
            fflush(stdout);
        }
    }

    free(weights);
}

/* ----------------------------- TDC utils ----------------------------- */
static inline int popcount32(uint32_t x) { return __builtin_popcount(x); }
static inline int bit_polarity(uint32_t v) { return ((v & 0x80000000u) == 0); }

static void tdc_print_info(void) {
    printf("TDC Addr:  0x%08" PRIX32 "\n", tdc_inst.Config.BaseAddr);
    printf("TDC Count: %u\n",   (unsigned)tdc_inst.Config.Count);
    printf("TDC Depth: %u\n",   (unsigned)tdc_inst.Config.Depth);
    printf("TDC Fine:  %u\n",   (unsigned)tdc_inst.Fine);
    printf("TDC Coarse:%u\n",   (unsigned)tdc_inst.Coarse);
    printf("TDC IsReady:%u\n",  (unsigned)tdc_inst.IsReady);
}

static void tdc_dump_state(int ch, int reads) {
    if (ch < 0) ch = 0;
    if (reads <= 0) reads = 64;
    XTDC_WriteReg(tdc_inst.Config.BaseAddr, XTDC_SEL_OFFSET, (uint32_t)ch);
    printf("TDC %d state :", ch);
    for (int i = 0; i < reads; i++) {
        if (i % 8 == 0) printf("\n");
        uint32_t v = XTDC_ReadReg(tdc_inst.Config.BaseAddr, XTDC_STATE_OFFSET);
        printf("%08" PRIX32 " ", v);
    }
    printf("\n");
}

static void tdc_get_delay(int ch) {
    uint64_t d = XTDC_ReadDelay(&tdc_inst, ch);
    if (ch == -1) {
        printf("delay: 0x%08" PRIX32 "%08" PRIX32 "\n",
               (uint32_t)(d >> 32), (uint32_t)d);
    } else {
        uint32_t fine   = (uint32_t)(d & 0xFFFFFFFFu);
        uint32_t coarse = (uint32_t)(d >> 32);
        printf("CH%d delay: fine=0x%" PRIX32 " coarse=0x%" PRIX32 "\n",
               ch, fine, coarse);
    }
}

static void tdc_set_delay_all(uint32_t fine, uint32_t coarse) {
    XTDC_WriteDelay(&tdc_inst, -1, fine, coarse);
    tdc_get_delay(-1);
}

static void tdc_set_delay_one(int ch, uint32_t fine, uint32_t coarse) {
    XTDC_WriteDelay(&tdc_inst, ch, fine, coarse);
    tdc_get_delay(ch);
}

static void tdc_avg(int ch, int iters) {
    if (iters <= 0) iters = 1024;
    XTDC_WriteReg(tdc_inst.Config.BaseAddr, XTDC_SEL_OFFSET, (uint32_t)ch);
    uint64_t value = 0;
    int pol = 0;
    for (int i = 0; i < iters; i++) {
        uint32_t raw = XTDC_ReadReg(tdc_inst.Config.BaseAddr, XTDC_STATE_OFFSET);
        value += (uint64_t)popcount32(raw);
        pol   += bit_polarity(raw);
    }
    double avg = (double)value / (double)iters;
    printf("CH%d AVG_WEIGHT=%.2f POLARITY_MAJ=%s\n",
           ch, avg, (pol > (iters/2)) ? "HIGH" : "LOW");
}

/* Fallback OS-mapped virtual address (legacy / --addr override) */
static uint8_t g_dummy_buffer[4096] __attribute__((aligned(4096)));
static const uint8_t *pick_os_virtual_addr(void) {
    return (const uint8_t *)&g_dummy_buffer[0];
}

/* ----------------------------- Self-test ----------------------------- */
static int do_selftest(void) {
    int fails = 0;

    uint32_t f_before = XTDC_ReadReg(tdc_inst.Config.BaseAddr, XTDC_FINE_OFFSET);
    XTDC_WriteReg(tdc_inst.Config.BaseAddr, XTDC_FINE_OFFSET, 0x5A5A5A5A);
    uint32_t f_after  = XTDC_ReadReg(tdc_inst.Config.BaseAddr, XTDC_FINE_OFFSET);
    if (f_after != 0x5A5A5A5A) {
        printf(COLOR_RED "[FAIL]" COLOR_RESET " TDC FINE write/readback: got 0x%08" PRIX32 "\n", f_after);
        fails++;
    } else {
        printf(COLOR_GREEN "[PASS]" COLOR_RESET " TDC FINE write/readback\n");
    }
    XTDC_WriteReg(tdc_inst.Config.BaseAddr, XTDC_FINE_OFFSET, f_before);

    fifo_inst.Mode = XFIFO_MODE_SW;
    XFIFO_Reset(&fifo_inst);
    uint32_t st1 = XFIFO_ReadReg(fifo_inst.Config.BaseAddr, XFIFO_STATUS_RD_OFFSET);
    uint32_t c1  = XFIFO_GetCount(fifo_inst.Config.BaseAddr);

    int empty1 = (st1 & XFIFO_STATUS_EMPTY_MASK) != 0;
    if (!empty1 || c1 != 0) {
        printf(COLOR_RED "[FAIL]" COLOR_RESET " FIFO reset: after st=0x%08" PRIX32 " cnt=%" PRIu32 "\n", st1, c1);
        fails++;
    } else {
        printf(COLOR_GREEN "[PASS]" COLOR_RESET " FIFO reset: EMPTY=1 and COUNT=0 after reset\n");
    }

    uint32_t c2_before = XFIFO_GetCount(fifo_inst.Config.BaseAddr);
    (void)XFIFO_Write(&fifo_inst, c2_before + 1, NULL);
    for (volatile int i=0;i<100000;i++);
    uint32_t c2_after  = XFIFO_GetCount(fifo_inst.Config.BaseAddr);
    if (c2_after > c2_before) {
        printf(COLOR_GREEN "[PASS]" COLOR_RESET " Acquire increased COUNT: %" PRIu32 " -> %" PRIu32 "\n", c2_before, c2_after);
    } else {
        printf(COLOR_YELLOW "[WARN]" COLOR_RESET " Acquire did not increase COUNT (producer may be idle)\n");
    }

    return fails ? -1 : 0;
}

/* -------------------------- HW init / deinit ------------------------- */
static int hw_init(void) {
    int init = -1;

    ensure_trace_dir();

    if (Xil_CacheInit() != 0) {
        fprintf(stderr, COLOR_RED "Warning: Xil_CacheInit failed — "
                        "cache operations may not work." COLOR_RESET "\n");
    }

    size_t fifo_size = (size_t)(XPAR_FIFO_AND_CTRL_0_S_AXI_HIGHADDR
                              - XPAR_FIFO_AND_CTRL_0_S_AXI_BASEADDR + 1);
    if (mmio_map(&mm_fifo, XPAR_FIFO_AND_CTRL_0_S_AXI_BASEADDR, fifo_size)) {
        perror("mmio fifo"); return -1;
    }

    size_t tdc_size = (size_t)(XPAR_TDC_BANK_0_S_AXI_HIGHADDR
                             - XPAR_TDC_BANK_0_S_AXI_BASEADDR + 1);
    if (mmio_map(&mm_tdc, XPAR_TDC_BANK_0_S_AXI_BASEADDR, tdc_size)) {
        perror("mmio tdc"); return -1;
    }

    printf("\n******************** IP Cores ***********************\n\n");

    XFIFO_Config *fifo_cfg = XFIFO_LookupConfig(0);
    if (fifo_cfg != NULL) {
        fifo_cfg->BaseAddr = (u32)mm_fifo.vbase;

        printf("FIFO PHYS: 0x%08lX .. 0x%08lX\n",
               (unsigned long)XPAR_FIFO_AND_CTRL_0_S_AXI_BASEADDR,
               (unsigned long)XPAR_FIFO_AND_CTRL_0_S_AXI_HIGHADDR);
        printf("FIFO VIRT: %p\n", (void*)mm_fifo.vbase);

        printf("FIFO Depth: %lu - Width: %lu bit\n",
               (unsigned long)fifo_cfg->Depth,
               (unsigned long)fifo_cfg->Width);

        init = XFIFO_CfgInitialize(&fifo_inst, fifo_cfg, (uintptr_t)mm_fifo.vbase);
        if (init != XST_SUCCESS) {
            printf(COLOR_RED "SnoopyPower> FIFO Init Error" COLOR_RESET "\n");
            return -1;
        } else {
            printf(COLOR_GREEN "SnoopyPower> FIFO Init OK" COLOR_RESET "\n");
        }
    } else {
        printf("Error: Unable to find configuration for FIFO Device ID 0.\n");
        return -1;
    }

    XTDC_Config *tdc_cfg = XTDC_LookupConfig(0);
    if (tdc_cfg != NULL) {
        tdc_cfg->BaseAddr = (u32)mm_tdc.vbase;

        printf("TDC  PHYS: 0x%08lX .. 0x%08lX\n",
               (unsigned long)XPAR_TDC_BANK_0_S_AXI_BASEADDR,
               (unsigned long)XPAR_TDC_BANK_0_S_AXI_HIGHADDR);
        printf("TDC  VIRT: %p\n", (void*)mm_tdc.vbase);

        printf("TDC  Quantization: %u levels - Channels: %u\n",
               (unsigned)tdc_cfg->Depth, (unsigned)tdc_cfg->Count);

        init = XTDC_CfgInitialize(&tdc_inst, tdc_cfg);
        if (init != XST_SUCCESS) {
            printf(COLOR_RED "SnoopyPower> TDC Init Error" COLOR_RESET "\n");
            return -1;
        } else {
            printf(COLOR_GREEN "SnoopyPower> TDC Init OK" COLOR_RESET "\n");
        }

        /* ---- TDC calibration or preset ----
         *
         * If SNOOPYPOWER_TDC_DELAY="FINE,COARSE" is set in the environment,
         * skip the slow calibration sweep and write those tap values
         * directly. This makes startup instant and keeps the TDC baseline
         * identical to training — important for QDA/LDA models trained on
         * raw TDC weights (no HPF).
         *
         * Example:
         *   export SNOOPYPOWER_TDC_DELAY="0x00006006,0x00000303"
         *   sudo -E ./myapp sca --pattern 1 --iters 1000
         */
        const char *tdc_preset = getenv("SNOOPYPOWER_TDC_DELAY");
        if (tdc_preset) {
            uint32_t pf = 0, pc = 0;
            if (sscanf(tdc_preset, "0x%x,0x%x", &pf, &pc) == 2 ||
                sscanf(tdc_preset, "%u,%u", &pf, &pc) == 2) {
                XTDC_WriteDelay(&tdc_inst, -1, pf, pc);
                uint64_t d = XTDC_ReadDelay(&tdc_inst, -1);
                printf(COLOR_GREEN "TDC preset loaded from SNOOPYPOWER_TDC_DELAY"
                       COLOR_RESET "\n");
                printf("  fine=0x%08" PRIX32 "  coarse=0x%08" PRIX32 "\n",
                       (uint32_t)(d & 0xFFFFFFFFu), (uint32_t)(d >> 32));
            } else {
                fprintf(stderr, COLOR_YELLOW
                        "Warning: SNOOPYPOWER_TDC_DELAY='%s' — bad format "
                        "(expected 'FINE,COARSE'); falling back to calibration"
                        COLOR_RESET "\n", tdc_preset);
                goto do_calibration;
            }
        } else {
do_calibration:;
            printf("Starting TDC calibration (set SNOOPYPOWER_TDC_DELAY to skip)...\n");
            uint64_t d = XTDC_Calibrate(&tdc_inst, 0, 0);
            uint32_t cal_fine   = (uint32_t)(d & 0xFFFFFFFFu);
            uint32_t cal_coarse = (uint32_t)(d >> 32);
            printf("Calibration result: fine=0x%08" PRIX32 "  coarse=0x%08" PRIX32 "\n",
                   cal_fine, cal_coarse);
            printf(COLOR_CYAN
                   "  To reuse this calibration:\n"
                   "    export SNOOPYPOWER_TDC_DELAY=\"0x%08" PRIX32 ",0x%08" PRIX32 "\"\n"
                   "    sudo -E ./myapp sca --pattern 1 --iters 1000"
                   COLOR_RESET "\n", cal_fine, cal_coarse);
        }
    } else {
        printf("Error: Unable to find configuration for TDC Device ID 0.\n");
        return -1;
    }

#ifdef SNOOPYPOWER_MEMORY
    if (membench_rand_init() != 0) {
        fprintf(stderr, COLOR_YELLOW "Warning: random arena init failed — "
                        "patterns 1/2/3 will not work." COLOR_RESET "\n");
    }
    warmup_probe_icache();
#endif

    printf("\nSummary:\n");
    printf("  FIFO virt=%p depth=%u width=%u\n",
           (void*)fifo_inst.Config.BaseAddr, fifo_inst.Config.Depth, fifo_inst.Config.Width);
    printf("  TDC  virt=%p depth=%u count=%u\n",
           (void*)tdc_inst.Config.BaseAddr,  tdc_inst.Config.Depth,  tdc_inst.Config.Count);

    fifo_inst.Mode = XFIFO_MODE_SW;
    XFIFO_Reset(&fifo_inst);
    fifo_status();

    printf("\n*****************************************************\n\n");
    return 0;
}

static void hw_deinit(void) {
#ifdef SNOOPYPOWER_MEMORY
    membench_rand_deinit();
#endif
    Xil_CacheDeinit();
    mmio_unmap(&mm_tdc);
    mmio_unmap(&mm_fifo);
}

/* ------------------------------ CLI ---------------------------------- */
static void usage(FILE *f) {
    fprintf(f,
"Usage:\n"
"  sudo ./myapp <subcommand> [options]\n"
"\n"
"Subcommands:\n"
"  fifo        FIFO operations (SW mode only)\n"
"  tdc        TDC operations (calibration, delay, state)\n"
"  sca        Side-channel cache-state characterization\n"
"  selftest   Run built-in hardware self-test\n"
"\n"
"fifo options (SW mode):\n"
"  --flush                Reset FIFO\n"
"  --acquire              Toggle write enable (SW mode)\n"
"  --start N              Start index (default 0)\n"
"  --end N                End index (default depth-1)\n"
"  --verbose              Print hex/ascii dump\n"
"  --status               Print STATUS/COUNT registers\n"
"\n"
"tdc options:\n"
"  --info                 Print TDC info\n"
"  --avg CH --iters K     Average weight/polarity for channel CH (default K=1024)\n"
"  --state CH --reads R   Dump R readings of STATE for channel CH (default 64)\n"
"  --set-all F C          Set ALL channels delay (fine, coarse)\n"
"  --set CH F C           Set one channel delay\n"
"  --get CH               Get channel delay; CH=-1 prints raw regs\n"
"  --calibrate K          Calibrate with K iterations\n"
"  --verbose              Verbose prints during calibration\n"
"\n"
"sca options:\n"
"  --pattern N            Pattern ID:\n"
"                          1  = L1 cache HIT     (random addr, random value)\n"
"                          2  = L2 HIT / L1 miss (random addr, random value)\n"
"                          3  = DRAM miss        (random addr, random value)\n"
"                          14 = L1 HIT  (unprivileged probe profile)\n"
"                          15 = L2 HIT  (unprivileged probe profile)\n"
"                          16 = DRAM miss (unprivileged probe profile)\n"
"                          10 = legacy noL1 (deterministic / --addr override)\n"
"  --iters N              Number of iterations (default 1)\n"
"  --hit-idx N            Cache-hit index (for pattern 10, default 0)\n"
"  --addr 0xADDR          Override address (for pattern 10)\n"
"  --mode STR             Mode string, e.g. 'memint' (default)\n"
"  --start N              FIFO read start (default 0)\n"
"  --end N                FIFO read end (default depth-1)\n"
"  --raw                  Emit raw markers around acquire/read (disable progress bar)\n"
"  --verbose              Verbose output\n"
"\n"
"  Patterns 1/2/3/14/15/16 draw a fresh random address and random value\n"
"  from /dev/urandom on every iteration (strong dataset generation).\n"
"\n"
"Examples:\n"
"  sudo ./myapp sca --pattern 1 --iters 5000 --mode memint\n"
"  sudo ./myapp sca --pattern 2 --iters 5000 --mode memint\n"
"  sudo ./myapp sca --pattern 3 --iters 5000 --mode memint\n"
"  sudo ./myapp sca --pattern 14 --iters 5000 --mode memint\n"
"  sudo ./myapp tdc --calibrate 4096 --verbose\n"
"  sudo ./myapp selftest\n"
    );
}

/* --------------------------- Subcommand: FIFO (SW) ------------------- */
static int cmd_fifo(int argc, char **argv) {
    int flush=0, acquire=0, verbose=0, show_status=0;
    int start=-1, end=-1;

    static struct option o[] = {
        {"flush",   0,0,'f'},
        {"acquire", 0,0,'a'},
        {"start",   1,0,'s'},
        {"end",     1,0,'e'},
        {"verbose", 0,0,'v'},
        {"status",  0,0,'p'},
        {0,0,0,0}
    };
    optind = 1;
    for (int c; (c=getopt_long(argc, argv, "fas:e:vp", o, NULL)) != -1;) {
        if (c=='f') flush=1;
        else if (c=='a') acquire=1;
        else if (c=='s') start=atoi(optarg);
        else if (c=='e') end=atoi(optarg);
        else if (c=='v') verbose=1;
        else if (c=='p') show_status=1;
        else { usage(stderr); return 1; }
    }

    if (start < 0) start = 0;
    if (end   < 0) end   = (int)fifo_inst.Config.Depth - 1;

    if (show_status) fifo_status();
    if (flush)   fifo_flush();
    if (acquire) fifo_acquire(end);
    fifo_read(verbose, start, end);

    return 0;
}

/* --------------------------- Subcommand: TDC ------------------------- */
static int cmd_tdc(int argc, char **argv) {
    int info=0, verbose=0;
    int do_avg=0, avg_ch=0, iters=1024;
    int do_state=0, state_ch=0, reads=64;
    int do_set_all=0, do_set_one=0, set_ch=0;
    uint32_t fine=0, coarse=0;
    int do_get=0, get_ch=0;
    int do_cal=0, cal_iters=8192;

    static struct option o[] = {
        {"info",      0,0,'i'},
        {"avg",       1,0,'A'},
        {"iters",     1,0,'k'},
        {"state",     1,0,'S'},
        {"reads",     1,0,'r'},
        {"set-all",   1,0,'G'},
        {"set",       1,0,'g'},
        {"get",       1,0,'t'},
        {"calibrate", 1,0,'c'},
        {"verbose",   0,0,'v'},
        {0,0,0,0}
    };

    optind = 1;
    for (int c; (c=getopt_long(argc, argv, "iA:k:S:r:G:g:t:c:v", o, NULL)) != -1;) {
        switch (c) {
            case 'i': info = 1; break;
            case 'A': do_avg = 1; avg_ch = atoi(optarg); break;
            case 'k': iters = atoi(optarg); break;
            case 'S': do_state = 1; state_ch = atoi(optarg); break;
            case 'r': reads = atoi(optarg); break;
            case 'G': do_set_all = 1;
                      if (optind >= argc) { fprintf(stderr,"--set-all needs F C\n"); return 1; }
                      fine    = (uint32_t)strtoul(optarg, NULL, 0);
                      coarse = (uint32_t)strtoul(argv[optind++], NULL, 0);
                      break;
            case 'g': do_set_one = 1;
                      set_ch = atoi(optarg);
                      if (optind+1 >= argc) { fprintf(stderr,"--set CH F C\n"); return 1; }
                      fine    = (uint32_t)strtoul(argv[optind++], NULL, 0);
                      coarse = (uint32_t)strtoul(argv[optind++], NULL, 0);
                      break;
            case 't': do_get = 1; get_ch = atoi(optarg); break;
            case 'c': do_cal = 1; cal_iters = atoi(optarg); break;
            case 'v': verbose = 1; break;
            default: usage(stderr); return 1;
        }
    }
    (void)verbose;

    if (info)        tdc_print_info();
    if (do_avg)      tdc_avg(avg_ch, iters);
    if (do_state)    tdc_dump_state(state_ch, reads);
    if (do_set_all)  tdc_set_delay_all(fine, coarse);
    if (do_set_one)  tdc_set_delay_one(set_ch, fine, coarse);
    if (do_get)      tdc_get_delay(get_ch);
    if (do_cal) {
        uint64_t d = XTDC_Calibrate(&tdc_inst, 0, 0);
        printf("Calibration: fine=0x%08" PRIX32 " coarse=0x%08" PRIX32 "\n",
               (uint32_t)(d & 0xFFFFFFFFu), (uint32_t)(d >> 32));
        (void)cal_iters;
    }

    return 0;
}

/* --------------------------- Subcommand: SELFTEST -------------------- */
static int cmd_selftest(void) {
    int rc = do_selftest();
    printf("SELFTEST %s\n", rc==0 ? "PASSED" : "FAILED");
    return rc ? 4 : 0;
}

/* --------------------------- Subcommand: SCA ------------------------- */
static int cmd_sca(int argc, char **argv) {
    printf("\n" COLOR_BOLD "******************** SCA Mode ***********************" COLOR_RESET "\n");
    printf("Cache-state characterization mode.\n");

    int iterations = 1;
    int pattern_id = -1;
    size_t cache_hit_index = 0;
    const char *mode = "memint";
    int start = 0;
    int end = (int)(fifo_inst.Config.Depth - 1);
    int verbose = 0;
    int raw = 0;
    const uint8_t *direct_ptr = NULL;

    static struct option o[] = {
        {"iters",    1,0,'t'},
        {"pattern",  1,0,'p'},
        {"hit-idx",  1,0,'z'},
        {"mode",     1,0,'m'},
        {"start",    1,0,'s'},
        {"end",      1,0,'e'},
        {"addr",     1,0,'a'},
        {"verbose",  0,0,'v'},
        {"raw",      0,0,'r'},
        {0,0,0,0}
    };
    optind = 1;
    for (int c; (c=getopt_long(argc, argv, "t:p:z:m:s:e:a:vr", o, NULL)) != -1;) {
        switch (c) {
            case 't': iterations = atoi(optarg); break;
            case 'p': pattern_id = atoi(optarg); break;
            case 'z': cache_hit_index = (size_t)strtoull(optarg, NULL, 0); break;
            case 'm': mode = optarg; break;
            case 's': start = atoi(optarg); break;
            case 'e': end    = atoi(optarg); break;
            case 'a': direct_ptr = (const uint8_t *)strtoull(optarg, NULL, 0); break;
            case 'v': verbose = 1; break;
            case 'r': raw = 1; break;
            default:
                fprintf(stderr, "Unknown SCA option\n");
                return 1;
        }
    }

    if (pattern_id < 0) {
        fprintf(stderr,
            COLOR_RED "Error: --pattern N is required.\n" COLOR_RESET
            "  1  = L1 hit  (random addr+val from /dev/urandom)\n"
            "  2  = L2 hit  (random addr+val from /dev/urandom)\n"
            "  3  = DRAM miss (random addr+val from /dev/urandom)\n"
            "  14 = L1 hit (unpriv probe profile)\n"
            "  15 = L2 hit (unpriv probe profile)\n"
            "  16 = DRAM miss (unpriv probe profile)\n"
            "  10 = legacy noL1 (deterministic / --addr override)\n");
        return 1;
    }

#ifdef SNOOPYPOWER_TDC
    printf("sensors: %u;;\n", (unsigned)tdc_inst.Config.Count);
#endif

    printf("  %-15s: " COLOR_YELLOW "%d" COLOR_RESET "\n", "Pattern ID", pattern_id);
    printf("  %-15s: " COLOR_YELLOW "%s" COLOR_RESET "\n", "Mode", mode);
    printf("  %-15s: " COLOR_YELLOW "%d" COLOR_RESET "\n", "Iterations", iterations);
    printf("  %-15s: " COLOR_YELLOW "%d" COLOR_RESET "\n", "End of FIFO", end);

    if (pattern_id == 10 || direct_ptr) {
        if (!direct_ptr) {
            direct_ptr = pick_os_virtual_addr();
            printf("  %-15s: " COLOR_YELLOW "Fallback OS Addr" COLOR_RESET "\n", "Target Addr");
        }
        printf("  %-15s: " COLOR_YELLOW "%p" COLOR_RESET "\n", "Fixed Addr", (void *)direct_ptr);
    } else {
        printf("  %-15s: " COLOR_YELLOW "Random Arena (/dev/urandom)" COLOR_RESET "\n", "Target Addr");
    }
    printf("\n");
    fflush(stdout);

    time_t start_time = time(NULL);

    /* Truncate the trace file so we only capture THIS run (fifo_read appends). */
    {
        FILE *trunc = fopen(TRACE_FILE, "w");
        if (trunc) fclose(trunc);
    }

    for (int it = 0; it < iterations; ++it) {

        if (!raw && !verbose) {
            ui_progress_bar(it, iterations, "Running", start_time);
        }

        if (raw) {
            printf("\xfd\xfd\xfd\xfd;;\n");
        }

        fifo_acquire(end);

#ifdef SNOOPYPOWER_MEMORY
        if (!strcmp(mode, "memint")) {
            memory_bench_run(pattern_id, cache_hit_index, end, direct_ptr);
        } else
#endif
        {
            if (verbose) printf("Unknown mode: %s\n", mode);
        }

        fifo_read(verbose, start, end);

        if (raw) {
            printf("\xfe\xfe\xfe\xfe;;\n");
            fflush(stdout);
        }
    }

    if (!raw && !verbose) {
        ui_progress_bar(iterations, iterations, "Completed", start_time);
        printf("\n\n" COLOR_GREEN "Success! %d iterations captured to %s" COLOR_RESET "\n", iterations, TRACE_FILE);
    } else {
        fifo_flush();
        printf("\xff\xff\xff\xff;;\n");
    }

    return 0;
}

/* ------------------------------ main --------------------------------- */
int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    print_banner();

    if (geteuid() != 0) {
        fprintf(stderr, COLOR_RED
                "This binary must run as root (MMIO /dev/mem access required for FIFO/TDC).\n"
                COLOR_RESET);
        return 1;
    }

    if (hw_init() != 0) {
        fprintf(stderr, COLOR_RED "Hardware init failed. Are you root? Bitstream loaded?" COLOR_RESET "\n");
        return 1;
    }

    if (argc < 2) { usage(stderr); hw_deinit(); return 1; }

    const char *sub = argv[1];
    int rc = 0;
    if      (!strcmp(sub, "fifo"))     rc = cmd_fifo(argc-1, argv+1);
    else if (!strcmp(sub, "tdc"))      rc = cmd_tdc (argc-1, argv+1);
    else if (!strcmp(sub, "selftest")) rc = cmd_selftest();
    else if (!strcmp(sub, "sca"))      rc = cmd_sca(argc-1, argv+1);
    else if (!strcmp(sub, "-h") || !strcmp(sub, "--help") || !strcmp(sub, "help")) {
        usage(stdout); rc = 0;
    } else {
        fprintf(stderr, "Unknown subcommand: %s\n\n", sub);
        usage(stderr); rc = 1;
    }

    hw_deinit();
    return rc;
}
