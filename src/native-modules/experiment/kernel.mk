-include $(DIR)/$(FILE).d

#'-Xcompiler', '-pedantic', #'-Werror', '-W' # shit i couldn't get working
#'-g', # debug
#'-DCUDA_FORCE_CDP1_IF_SUPPORTED',
# -lcudadevrt -lcudart

ARCH := -arch sm_89

NVCC_COMPILE_FLAGS := -Xcompiler -fPIC \
	-Xcompiler -Wall \
	-Xcompiler -Wextra \
        -Xcudafe --diag_suppress=declared_but_not_referenced \
	--expt-relaxed-constexpr \
	-maxrregcount=40 \
	-O3 \
        -dopt=on \
        -Xptxas=-v \
        -lineinfo \
        -std=c++20

NVCC_LINK_FLAGS := -Xcompiler -fPIC

build: $(DIR)/$(FILE)_dlink.o

$(DIR)/$(FILE)_dlink.o: $(DIR)/$(FILE).o
	echo "Device linking $@"
	nvcc $(ARCH) $(NVCC_LINK_FLAGS) -dlink $(DIR)/$(FILE).o -o $@

$(DIR)/$(FILE).o: $(FILE).cu
	@mkdir -p $(DIR)
	echo "Compiling $@"
	nvcc $(ARCH) -MP -MMD -MF $(DIR)/$(FILE).d $(NVCC_COMPILE_FLAGS) -dc $(FILE).cu -o $@
