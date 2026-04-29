CC      = gcc
CFLAGS  = -O0 -g -Wall -Iinclude -Idrivers -Ibenches -Iutils -I. -D_GNU_SOURCE \
          -fno-schedule-insns -fno-schedule-insns2 -fno-inline \
          -fno-omit-frame-pointer \
          -DSNOOPYPOWER_TDC -DSNOOPYPOWER_MEMORY

LDFLAGS =
LDLIBS  = -lm -lrt -lpthread

DRIVER_OBJS = drivers/xtdc.o drivers/xtdc_sinit.o \
              drivers/xfifo.o drivers/xfifo_sinit.o

BENCH_OBJS  = benches/membench_core.o benches/membench_patterns.o \
              benches/membench_addr.o benches/membench_prng.o \
              benches/membench_rand.o

UTILS_OBJS  = utils/pmu.o shared_region.o cache/xil_cache_linux.o

OBJS = mmio_linux.o main_linux.o \
       $(DRIVER_OBJS) $(BENCH_OBJS) $(UTILS_OBJS)

all: myapp

myapp: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) myapp

.PHONY: all clean
