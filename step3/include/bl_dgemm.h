#ifndef BLISLAB_DGEMM_H
#define BLISLAB_DGEMM_H

// Allow C++ users to include this header file in their source code. However,
// we make the extern "C" conditional on whether we're using a C++ compiler,
// since regular C compilers don't understand the extern "C" construct.
#ifdef __cplusplus
extern "C" {
#endif

#include <immintrin.h>
#include <math.h>

#include <stdio.h>
#include <stdlib.h>

// Determine the target operating system
#if defined(_WIN32) || defined(__CYGWIN__)
#define BL_OS_WINDOWS 1
#elif defined(__APPLE__) || defined(__MACH__)
#define BL_OS_OSX 1
#elif defined(__ANDROID__)
#define BL_OS_ANDROID 1
#elif defined(__linux__)
#define BL_OS_LINUX 1
#elif defined(__bgq__)
#define BL_OS_BGQ 1
#elif defined(__bg__)
#define BL_OS_BGP 1
#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
    defined(__bsdi__) || defined(__DragonFly__)
#define BL_OS_BSD 1
#else
#error "Cannot determine operating system"
#endif

// gettimeofday() needs this.
#if BL_OS_WINDOWS
#include <time.h>
#elif BL_OS_OSX
#include <mach/mach_time.h>
#else
#include <sys/time.h>
#include <time.h>
#endif

#include "bl_config.h"

#define min(i, j) ((i) < (j) ? (i) : (j))

#define A(i, j) A[(j)*lda + (i)]
#define B(i, j) B[(j)*ldb + (i)]
#define C(i, j) C[(j)*ldc + (i)]
#define C_ref(i, j) C_ref[(j)*ldc_ref + (i)]

void bl_dgemm(
    int m,
    int n,
    int k,
    double* A,
    int lda,
    double* B,
    int ldb,
    double* C,
    int ldc);

double* bl_malloc_aligned(int m, int n, int size);

void bl_printmatrix(double* A, int lda, int m, int n);

double bl_clock(void);
double bl_clock_helper();

void bl_dgemm_ref(
    int m,
    int n,
    int k,
    double* XA,
    int lda,
    double* XB,
    int ldb,
    double* XC,
    int ldc);

void bl_get_range(int n, int bf, int* start, int* end);

// End extern "C" construct block.
#ifdef __cplusplus
}
#endif

#endif
