
CUDA_INCLUDE = 
CUDA_LIB_DIR = 
CUDA_LINK_LIBS= -lcudart -lcurand

NVCC = nvcc
NVCC_FLAGS = -I./include $(CUDA_INCLUDE) -dc -cudart shared -arch=sm_89 --compiler-options "-Wall -Wno-format-truncation"
# -g -G

SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

C_SRC_FILES  := $(shell find $(SRC_DIR) -name '*.c')
CU_SRC_FILES := $(shell find $(SRC_DIR) -name '*.cu')

$(info C_SRC_FILES  = $(C_SRC_FILES))
$(info CU_SRC_FILES = $(CU_SRC_FILES))

OBJ_FILES := \
	$(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(C_SRC_FILES)) \
	$(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CU_SRC_FILES))

$(info OBJ_FILES  = $(OBJ_FILES))


TARGET = $(BIN_DIR)/pathtracer

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(TARGET): $(OBJ_FILES)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $^ -o $@ -arch=sm_89 $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

all: $(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: all clean
