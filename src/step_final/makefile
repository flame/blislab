ifeq ($(BLISGEMM_USE_INTEL),true)
include $(BLISGEMM_DIR)/make.intel.inc
else
include $(BLISGEMM_DIR)/make.gnu.inc
endif


FRAME_CC_SRC=  \
							 frame/blis_dgemm.c \
							 frame/blis_dgemm_ref.c \
							 frame/blis_dgemm_util.c \

FRAME_CPP_SRC= \
#							 frame/blisgemm_ref_stl.cpp \

KERNEL_SRC=    \
							 kernels/$(BLISGEMM_ARCH)/sq2nrm_asm_d8x4.c \
							 kernels/$(BLISGEMM_ARCH)/bli_dgemm_ukr.c \
							 kernels/$(BLISGEMM_ARCH)/bli_dgemm_int_d8x4.c \
							 kernels/$(BLISGEMM_ARCH)/bli_dgemm_asm_d8x4.c \
							 #kernels/$(BLISGEMM_ARCH)/rnn_rank_k_asm_d8x4.c \

OTHER_DEP = \
			                 kernels/$(BLISGEMM_ARCH)/blis_dgemm_kernel.h
			                 #frame/dblis_dgemm_ref_impl.h

BLISGEMM_OBJ=$(FRAME_CC_SRC:.c=.o) $(FRAME_CPP_SRC:.cpp=.o) $(KERNEL_SRC:.c=.o) $(FRAME_CC_SRC_S:.c=.os) $(KERNEL_SRC_S:.c=.os)

all: $(LIBBLISGEMM) $(SHAREDLIBBLISGEMM) TESTBLISGEMM

TESTBLISGEMM: $(LIBBLISGEMM)
	cd $(BLISGEMM_DIR)/test && $(MAKE) && cd $(BLISGEMM_DIR) $(LDFLAGS)

$(LIBBLISGEMM): $(BLISGEMM_OBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(BLISGEMM_OBJ)
	$(RANLIB) $@

$(SHAREDLIBBLISGEMM): $(BLISGEMM_OBJ)
	$(CC) $(CFLAGS) -shared -o $@ $(BLISGEMM_OBJ) $(LDLIBS)

# ---------------------------------------------------------------------------
# Object files compiling rules
# ---------------------------------------------------------------------------
%.o: %.c $(OTHER_DEP)
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@ $(LDFLAGS)
# ---------------------------------------------------------------------------

clean:
	rm $(BLISGEMM_OBJ) $(LIBBLISGEMM) $(SHAREDLIBBLISGEMM) frame/*~ test/*~ include/*~ *~
	cd $(BLISGEMM_DIR)/test && $(MAKE) clean && cd $(BLISGEMM_DIR)
