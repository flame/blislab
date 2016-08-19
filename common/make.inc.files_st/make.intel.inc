CC = icc
CXX = icpc

ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib

ifeq ($(BLISLAB_USE_BLAS),true)
CFLAGS = -$(COMPILER_OPT_LEVEL) -mavx -fPIC -DUSE_BLAS
LDLIBS = $(LIBBLISLAB) -lm -mkl=parallel -Werror -Wall -pedantic
#LDLIBS = $(LIBBLISLAB) -lm -mkl=sequential -Werror -Wall -pedantic
else
CFLAGS = -$(COMPILER_OPT_LEVEL) -mavx -fPIC
LDLIBS = $(LIBBLISLAB) -lm -Werror -Wall -pedantic
endif

LIBBLISLAB = $(BLISLAB_DIR)/lib/libblislab.a
SHAREDLIBBLISLAB = $(BLISLAB_DIR)/lib/libblislab.so

LDFLAGS = -I$(BLISLAB_DIR)/include -I$(BLISLAB_DIR)/kernels

OS_NAME = $(shell uname -s)
ifeq ($(OS_NAME),Linux)
LDLIBS += -lrt
endif

$(info * Using CFLAGS=${CFLAGS})
$(info * Using LDFLAGS=${LDFLAGS})
$(info * Using LDLIBS=${LDLIBS})


