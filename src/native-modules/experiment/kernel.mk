-include $(DIR)/$(FILE).d

# '-Xcompiler', '-pedantic', '-Werror' : options i couldn't get working
# -g # debug
# -G # device debug

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

.PHONY: compile

compile: $(DIR)/$(FILE).o

# would like something like this but make doesn't match due to no directory
# %_dlink.o: $(patsubst %,$(*D)/%,$(OBJ_FILES)) 

$(DIR)/%_dlink.o: $(patsubst %,$(DIR)/%,$(OBJ_FILES))
	echo "Device linking $@"
	nvcc $(ARCH) $(NVCC_LINK_FLAGS) -dlink $^ -o $@

$(DIR)/%.o: %.cu
	@mkdir -p $(*D)
	echo "Compiling $<"
	nvcc $(ARCH) -MP -MMD -MF $*.d $(NVCC_COMPILE_FLAGS) -dc $< -o $@
