# 微型推理引擎 Makefile
# 本地编译:       make
# RISC-V 交叉编译: make CC=riscv64-unknown-elf-gcc
#                  make CC=riscv32-unknown-elf-gcc

CC      ?= gcc
CFLAGS  ?= -std=c99 -Wall -Wextra -O2
LDFLAGS ?=

# 裸机/硬浮点 ABI 示例:
#   make CC=riscv64-unknown-elf-gcc CFLAGS="-std=c99 -O2 -march=rv32imac -mabi=ilp32"

SRCS   = tensor.c ops.c nn.c main.c
OBJS   = $(SRCS:.c=.o)
HDRS   = tensor.h ops.h nn.h
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

test: $(TARGET)
	./$(TARGET) && echo "ALL TESTS PASSED" || echo "TESTS FAILED"

clean:
	rm -f $(OBJS) $(TARGET) $(TARGET).exe

.PHONY: all run test clean
