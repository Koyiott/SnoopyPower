CC      = gcc
CFLAGS  = -O2 -Wall -D_GNU_SOURCE

all: merge_dataset

merge_dataset: merge_dataset.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f merge_dataset

.PHONY: all clean
