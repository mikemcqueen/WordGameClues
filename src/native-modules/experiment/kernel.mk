-include $(DIR)/$(FILE).d

#'-Xcompiler', '-pedantic', '-Werror', # options i couldn't get working
# -g # debug
# -G # device debug
# -DCUDA_FORCE_CDP1_IF_SUPPORTED
# -lcudadevrt -lcudart

ARCH := -arch sm_89

NVCC_COMPILE_FLAGS := -Xcompiler -fPIC \
	-Xcompiler -Wall \
	-Xcompiler -Wextra \
	-Xcudafe --diag_suppress=declared_but_not_referenced \
	--expt-relaxed-constexpr \
	-O3 \
	-Xptxas=-v \
	-lineinfo \
	-std=c++20 \
	-dopt=on \
	-maxrregcount=40


NVCC_LINK_FLAGS := -Xcompiler -fPIC

.PHONY: compile dlink_obj dlink_lib dlink_all
compile: $(DIR)/$(FILE).o
dlink: $(DIR)/$(FILE)_dlink.o
dlink_lib: $(DIR)/$(FILE)_dlink.a
dlink_all: $(DIR)/kernels_dlink.o

$(DIR)/$(FILE)_dlink.a: $(DIR)/$(FILE).a
	echo "Device linking library $@"
	nvcc $(ARCH) $(NVCC_LINK_FLAGS) -dlink $^ -o $@


#$(DIR)/$(FILE)_dlink.o: $(DIR)/$(FILE).o $(DIR)/filter.o $(DIR)/or-filter.o

$(DIR)/kernels_dlink.o: $(DIR)/merge.o $(DIR)/filter.o $(DIR)/or-filter.o
	echo "Device linking $@"
	nvcc $(ARCH) $(NVCC_LINK_FLAGS) -dlink $^ -o $@

$(DIR)/$(FILE)_dlink.o: $(DIR)/$(FILE).o $(DEP)
	echo "Device linking $@"
	nvcc $(ARCH) $(NVCC_LINK_FLAGS) -dlink $^ -o $@

$(DIR)/merge.o: merge.cu
	@mkdir -p $(DIR)
	echo "Compiling $@"
	nvcc $(ARCH) -MP -MMD -MF $(DIR)/$(FILE).d $(NVCC_COMPILE_FLAGS) -dc $< -o $@

$(DIR)/filter.o: filter.cu
	@mkdir -p $(DIR)
	echo "Compiling $@"
	nvcc $(ARCH) -MP -MMD -MF $(DIR)/$(FILE).d $(NVCC_COMPILE_FLAGS) -dc $< -o $@

$(DIR)/or-filter.o: or-filter.cu
	@mkdir -p $(DIR)
	echo "Compiling $@"
	nvcc $(ARCH) -MP -MMD -MF $(DIR)/$(FILE).d $(NVCC_COMPILE_FLAGS) -dc $< -o $@
