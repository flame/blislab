CC = icc
CXX = icpc

ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib

ifeq ($(BLISGEMM_USE_BLAS),true)
CFLAGS = -O3 -fopenmp -mavx -fPIC -DUSE_BLAS -g
LDLIBS = $(LIBBLISGEMM) -lpthread -lm -openmp -mkl=parallel -Werror -Wall -pedantic
#LDLIBS = $(LIBBLISGEMM) -lpthread -lm -openmp -mkl=sequential -Werror -Wall -pedantic
else
CFLAGS = -O3 -fopenmp -mavx -fPIC
LDLIBS = $(LIBBLISGEMM) -lpthread -lm -openmp -Werror -Wall -pedantic
endif

LIBBLISGEMM = $(BLISGEMM_DIR)/lib/libblisgemm.a
SHAREDLIBBLISGEMM = $(BLISGEMM_DIR)/lib/libblisgemm.so

LDFLAGS = -I$(BLISGEMM_DIR)/include -I$(BLISGEMM_DIR)/kernels/$(BLISGEMM_ARCH) -I$(BLISGEMM_MKL_DIR)/include
