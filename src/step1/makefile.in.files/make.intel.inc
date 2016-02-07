CC = icc
CXX = icpc

ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib

ifeq ($(BLISLAB_USE_BLAS),true)
CFLAGS = -O2 -fopenmp -mavx -fPIC -DUSE_BLAS
LDLIBS = $(LIBBLISLAB) -lpthread -lm -openmp -mkl=parallel -Werror -Wall -pedantic
#LDLIBS = $(LIBBLISLAB) -lpthread -lm -openmp -mkl=sequential -Werror -Wall -pedantic
else
CFLAGS = -O2 -fopenmp -mavx -fPIC
LDLIBS = $(LIBBLISLAB) -lpthread -lm -openmp -Werror -Wall -pedantic
endif

LIBBLISLAB = $(BLISLAB_DIR)/lib/libblislab.a
SHAREDLIBBLISLAB = $(BLISLAB_DIR)/lib/libblislab.so

LDFLAGS = -I$(BLISLAB_DIR)/include -I$(BLISLAB_DIR)/kernels/$(BLISLAB_ARCH) -I$(BLISLAB_MKL_DIR)/include
