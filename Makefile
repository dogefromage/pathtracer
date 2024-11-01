
CUDA_INCLUDE = /usr/local/cuda-12.5/include
CUDA_LIB_DIR = -L/usr/local/cuda-12.5/lib64
CUDA_LINK_LIBS= -lcudart -lcurand

# CC = gcc
# CC_FLAGS = -Wall -Wextra -g -G -I./include -std=gnu99

NVCC = nvcc
NVCC_FLAGS = -I./include -I$(CUDA_INCLUDE) -dc -cudart shared # -D USE_CPU_RENDER 

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

C_SRC_FILES := $(wildcard $(SRC_DIR)/*.c)
CU_SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(C_SRC_FILES)) $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CU_SRC_FILES))

EXEC_NAME = raytracer
# Executable name
TARGET = $(BIN_DIR)/$(EXEC_NAME)

# Default target
all: $(TARGET)

# $(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
# 	@mkdir -p $(BUILD_DIR)
# 	$(CC) $(CC_FLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(TARGET): $(OBJ_FILES)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $^ -o $@ $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

#removed -lm because of cuda math library ??

# Clean rule
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Phony target to avoid conflicts with files of the same name
.PHONY: all clean
