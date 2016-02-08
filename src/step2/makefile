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
							 kernels/bli_dgemm_ukr.c \

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
	rm $(BLISLAB_OBJ) $(LIBBLISLAB) $(SHAREDLIBBLISLAB) dgemm/*~ test/*~ include/*~ *~ make.inc.files/*~ kernels/*~
	cd $(BLISLAB_DIR)/test && $(MAKE) clean && cd $(BLISLAB_DIR)
