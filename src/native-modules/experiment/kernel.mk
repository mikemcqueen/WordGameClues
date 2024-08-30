# compile pre-reqs
-include $(OBJ_DIR)/*.d 
#-include $(patsubst %.o,$(DIR)/%.d,$(OBJ_FILES)) # dlink pre-reqs

# '-Xcompiler', '-pedantic', '-Werror' : options i couldn't get working
# -g # debug
# -G # device debug

ARCH := -arch sm_89

NVCC_COMPILE_FLAGS := -Xcompiler -fPIC \
	-Xcompiler -Wall \
	-Xcompiler -Wextra \
	-Xcompiler -Wconversion \
	-Xcudafe --diag_suppress=declared_but_not_referenced \
	--expt-relaxed-constexpr \
	--extended-lambda \
	-Xptxas=-v \
	-lineinfo \
	-std=c++20 \
	-O3 \
	-dopt=on \
	-maxrregcount=40

NVCC_LINK_FLAGS := -Xcompiler -fPIC

.PHONY: compile

compile: $(OBJ_DIR)/$(FILE).o

# would like something like this but make doesn't match due to no directory
# %_dlink.o: $(patsubst %,$(*D)/%,$(OBJ_FILES)) 

$(OBJ_DIR)/%_dlink.a: $(LIB_DIR)/%_lib.a 
	echo "Device linking $@..."
	nvcc $(ARCH) $(NVCC_LINK_FLAGS) -dlink $^ -o $@
	cp $@ $(LIB_DIR)

$(LIB_DIR)/%_lib.a: $(OBJ_DIR)/*.o

$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(*D)
	echo "Compiling $<..."
	nvcc $(ARCH) -MP -MMD -MF $(OBJ_DIR)/$*.d $(NVCC_COMPILE_FLAGS) -dc $< -o $@
