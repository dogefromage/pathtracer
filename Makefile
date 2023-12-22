
### CHATGPT GENERATED

# Makefile for simple C project

# Compiler and compiler flags
CC = gcc
CFLAGS = -Wall -Wextra -I./include -std=gnu99

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# HEADER_FILES := $(wildcard $(SRC_DIR)/*.h)
SRC_FILES := $(wildcard $(SRC_DIR)/*.c)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRC_FILES))

EXEC_NAME = raytracer

# Executable name
TARGET = $(BIN_DIR)/$(EXEC_NAME)

# Default target
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(OBJ_FILES)
	@mkdir -p $(BIN_DIR)
	$(CC) $^ -o $@ -lm

# Rule to compile C source files to object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Phony target to avoid conflicts with files of the same name
.PHONY: all clean
