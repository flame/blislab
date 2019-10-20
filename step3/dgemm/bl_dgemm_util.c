#include "bl_dgemm.h"

/*
 *
 *
 */
double* bl_malloc_aligned(int m, int n, int size) {
  double* ptr;
  int err;

  err =
      posix_memalign((void**)&ptr, (size_t)GEMM_SIMD_ALIGN_SIZE, size * m * n);

  if (err) {
    printf("bl_malloc_aligned(): posix_memalign() failures");
    exit(1);
  }

  return ptr;
}

/*
 *
 *
 */
void bl_dgemm_printmatrix(double* A, int lda, int m, int n) {
  int i, j;
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      printf("%lf\t", A[j * lda + i]);
    }
    printf("\n");
  }
}

/*
 * The timer functions are copied directly from BLIS 0.2.0
 *
 */
static double gtod_ref_time_sec = 0.0;

double bl_clock(void) {
  return bl_clock_helper();
}

#if BL_OS_WINDOWS
// --- Begin Windows build definitions -----------------------------------------

double bl_clock_helper() {
  LARGE_INTEGER clock_freq = {0};
  LARGE_INTEGER clock_val;
  BOOL r_val;

  r_val = QueryPerformanceFrequency(&clock_freq);

  if (r_val == 0) {
    fprintf(
        stderr,
        "\nblislab: %s (line %lu):\nblislab: %s \n",
        __FILE__,
        __LINE__,
        "QueryPerformanceFrequency() failed");
    fflush(stderr);
    abort();
  }

  r_val = QueryPerformanceCounter(&clock_val);

  if (r_val == 0) {
    fprintf(
        stderr,
        "\nblislab: %s (line %lu):\nblislab: %s \n",
        __FILE__,
        __LINE__,
        "QueryPerformanceFrequency() failed");
    fflush(stderr);
    abort();
  }

  return ((double)clock_val.QuadPart / (double)clock_freq.QuadPart);
}

// --- End Windows build definitions -------------------------------------------
#elif BL_OS_OSX
// --- Begin OSX build definitions -------------------------------------------

double bl_clock_helper() {
  mach_timebase_info_data_t timebase;
  mach_timebase_info(&timebase);

  uint64_t nsec = mach_absolute_time();

  double the_time = (double)nsec * 1.0e-9 * timebase.numer / timebase.denom;

  if (gtod_ref_time_sec == 0.0)
    gtod_ref_time_sec = the_time;

  return the_time - gtod_ref_time_sec;
}

// --- End OSX build definitions ---------------------------------------------
#else
// --- Begin Linux build definitions -------------------------------------------

double bl_clock_helper() {
  double the_time, norm_sec;
  struct timespec ts;

  clock_gettime(CLOCK_MONOTONIC, &ts);

  if (gtod_ref_time_sec == 0.0)
    gtod_ref_time_sec = (double)ts.tv_sec;

  norm_sec = (double)ts.tv_sec - gtod_ref_time_sec;

  the_time = norm_sec + ts.tv_nsec * 1.0e-9;

  return the_time;
}

// --- End Linux build definitions ---------------------------------------------
#endif
