#ifndef BLISLAB_DGEMM_KERNEL_H
#define BLISLAB_DGEMM_KERNEL_H

#include "bl_config.h"

#include <immintrin.h> // AVX
#include <stdio.h>

// Allow C++ users to include this header file in their source code. However,
// we make the extern "C" conditional on whether we're using a C++ compiler,
// since regular C compilers don't understand the extern "C" construct.
#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned long long dim_t;

typedef union {
  __m256d v;
  __m256i u;
  double d[4];
} v4df_t;

typedef union {
  __m128i v;
  int d[4];
} v4li_t;

struct aux_s {
  double* b_next;
  float* b_next_s;
  char* flag;
  int pc;
  int m;
  int n;
};
typedef struct aux_s aux_t;

void bl_dgemm_ukr(
    int k,
    double* a,
    double* b,
    double* c,
    unsigned long long ldc,
    aux_t* data);

void bl_dgemm_int_8x4(
    int k,
    double* a,
    double* b,
    double* c,
    unsigned long long ldc,
    aux_t* data);

void bl_dgemm_asm_8x4(
    int k,
    double* a,
    double* b,
    double* c,
    unsigned long long ldc,
    aux_t* data);

void bl_dgemm_asm_12x4(
    int k,
    double* a,
    double* b,
    double* c,
    unsigned long long ldc,
    aux_t* data);

void bl_dgemm_asm_8x6(
    int k,
    double* a,
    double* b,
    double* c,
    unsigned long long ldc,
    aux_t* data);

void bl_dgemm_asm_6x8(
    int k,
    double* a,
    double* b,
    double* c,
    unsigned long long ldc,
    aux_t* data);

void bl_dgemm_asm_4x12(
    int k,
    double* a,
    double* b,
    double* c,
    unsigned long long ldc,
    aux_t* data);

static void (*bl_micro_kernel)(
    int k,
    double* a,
    double* b,
    double* c,
    unsigned long long ldc,
    aux_t* aux) = {
    BL_MICRO_KERNEL
    // bl_dgemm_ukr
    // bl_dgemm_int_8x4
    // bl_dgemm_asm_8x4
    // bl_dgemm_asm_8x6
    // bl_dgemm_asm_12x4
};

// End extern "C" construct block.
#ifdef __cplusplus
}
#endif

#endif
