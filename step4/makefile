ifeq ($(BLISLAB_USE_INTEL),true)
include $(BLISLAB_DIR)/make.inc.files/make.intel.inc
else
include $(BLISLAB_DIR)/make.inc.files/make.gnu.inc
endif

$(info * Using CFLAGS=${CFLAGS})
$(info * Using LDFLAGS=${LDFLAGS})

FRAME_CC_SRC=  \
							 dgemm/my_dgemm.c \
							 dgemm/bl_dgemm_ref.c \
							 dgemm/bl_dgemm_util.c \

FRAME_CPP_SRC= \

KERNEL_SRC=    \
							 kernels/bl_dgemm_int_8x4.c \
			                 kernels/bl_dgemm_asm_12x4.c \
			                 kernels/bl_dgemm_asm_8x6.c \
							 kernels/bl_dgemm_ukr.c \
							 kernels/bl_dgemm_asm_8x4.c \

OTHER_DEP = \
			                 include/bl_dgemm.h \
			                 include/bl_dgemm_kernel.h \

BLISLAB_OBJ=$(FRAME_CC_SRC:.c=.o) $(FRAME_CPP_SRC:.cpp=.o) $(KERNEL_SRC:.c=.o) $(FRAME_CC_SRC_S:.c=.os) $(KERNEL_SRC_S:.c=.os)

all: $(LIBBLISLAB) $(SHAREDLIBBLISLAB) TESTBLISLAB

TESTBLISLAB: $(LIBBLISLAB)
	cd $(BLISLAB_DIR)/test && $(MAKE) && cd $(BLISLAB_DIR) $(LDFLAGS)

$(LIBBLISLAB): $(BLISLAB_OBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(BLISLAB_OBJ)
	$(RANLIB) $@

$(SHAREDLIBBLISLAB): $(BLISLAB_OBJ)
	$(CC) $(CFLAGS) -shared -o $@ $(BLISLAB_OBJ) $(LDLIBS)

# ---------------------------------------------------------------------------
# Object files compiling rules
# ---------------------------------------------------------------------------
%.o: %.c $(OTHER_DEP)
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@ $(LDFLAGS)
# ---------------------------------------------------------------------------

clean:
	-rm $(BLISLAB_OBJ) $(LIBBLISLAB) $(SHAREDLIBBLISLAB) dgemm/*~ kernels/*~ kernels/*.o test/*~ include/*~ *~ make.inc.files/*~
	$(MAKE) clean -C test

