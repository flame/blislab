export BLISLAB_DIR=..

ifeq ($(BLISLAB_USE_INTEL),true)
include $(BLISLAB_DIR)/makefile.in.files/make.intel.inc
else
include $(BLISLAB_DIR)/makefile.in.files/make.gnu.inc
endif


BLISLAB_TEST_CC_SRC= \
									 test_bl_dgemm.c \

BLISLAB_TEST_CPP_SRC=\


OTHER_DEP = \
			                $(LIBBLISLAB) \

#$(info ${OTHER_DEP})

BLISLAB_TEST_EXE= $(BLISLAB_TEST_CC_SRC:.c=.x) $(BLISLAB_TEST_CPP_SRC:.cpp=.x)

all: $(BLISLAB_TEST_EXE)

clean:
	rm -f $(BLISLAB_TEST_EXE)

# ---------------------------------------------------------------------------
# Executable files compiling rules
# ---------------------------------------------------------------------------
%.x: %.c ${OTHER_DEP}
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

%.x: %.cpp ${OTHER_DEP}
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)
# ---------------------------------------------------------------------------
