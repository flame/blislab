CC = gcc
CXX = g++

ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib

ifeq ($(BLISGEMM_USE_BLAS),true)
CFLAGS = -O3 -fopenmp -mavx -fPIC -DUSE_BLAS
LDLIBS = $(LIBBLISGEMM) -lpthread -lm -fopenmp -lblas
else
CFLAGS = -O3 -fopenmp -mavx -fPIC
LDLIBS = $(LIBBLISGEMM) -lpthread -lm -fopenmp
endif

LIBBLISGEMM = $(BLISGEMM_DIR)/lib/libblisgemm.a
SHAREDLIBBLISGEMM = $(BLISGEMM_DIR)/lib/libblisgemm.so

LDFLAGS = -I$(BLISGEMM_DIR)/include -I$(BLISGEMM_DIR)/kernels/$(BLISGEMM_ARCH)
