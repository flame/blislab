CC = gcc
CXX = g++

ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib

ifeq ($(BLISLAB_USE_BLAS),true)
CFLAGS = -$(COMPILER_OPTIMIZATION_LEVEL) -fopenmp -mavx -fPIC -DUSE_BLAS
LDLIBS = $(LIBBLISLAB) -lpthread -lm -fopenmp -lblas
else
CFLAGS = -$(COMPILER_OPTIMIZATION_LEVEL) -fopenmp -mavx -fPIC
LDLIBS = $(LIBBLISLAB) -lpthread -lm -fopenmp
endif

LIBBLISLAB = $(BLISLAB_DIR)/lib/libblislab.a
SHAREDLIBBLISLAB = $(BLISLAB_DIR)/lib/libblislab.so

LDFLAGS = -I$(BLISLAB_DIR)/include -I$(BLISLAB_DIR)/kernels

