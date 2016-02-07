CC = gcc
CXX = g++

ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib

ifeq ($(BLISLAB_USE_BLAS),true)
CFLAGS = -O2 -fopenmp -mavx -fPIC -DUSE_BLAS
LDLIBS = $(LIBBLISLAB) -lpthread -lm -fopenmp -lblas
else
CFLAGS = -O2 -fopenmp -mavx -fPIC
LDLIBS = $(LIBBLISLAB) -lpthread -lm -fopenmp
endif

LIBBLISLAB = $(BLISLAB_DIR)/lib/libblisgemm.a
SHAREDLIBBLISLAB = $(BLISLAB_DIR)/lib/libblisgemm.so

LDFLAGS = -I$(BLISLAB_DIR)/include -I$(BLISLAB_DIR)/kernels/$(BLISLAB_ARCH)
