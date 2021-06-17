#include <stdio.h>
#include <immintrin.h>
#include <bl_config.h>
#include "bl_dgemm_kernel.h"

//micro-panel a is stored in column major, lda=DGEMM_MR.
#define a(i,j) a[ (j)*DGEMM_MR + (i) ]
//micro-panel b is stored in row major, ldb=DGEMM_NR.
#define b(i,j) b[ (i)*DGEMM_NR + (j) ]
//result      c is stored in column major.
#define c(i,j) c[ (j)*ldc + (i) ]


void bl_dgemm_ukr( int    k,
                   double *a,
                   double *b,
                   double *c,
                   unsigned long long ldc,
                   aux_t* data )
{
    register long cstep2 = ldc * 2;
    register long cstep3 = ldc * 3;

    __m256d c_vec0 = _mm256_load_pd(c);
    __m256d c_vec1 = _mm256_load_pd(c + ldc);
    __m256d c_vec2 = _mm256_load_pd(c + cstep2);
    __m256d c_vec3 = _mm256_load_pd(c + cstep3);

    for (int i = 0; i < k; i++) {
        __m256d a_vec = _mm256_load_pd(a);
        __m256d b_vec;

        b_vec = _mm256_set1_pd(b[0]);
        c_vec0 = _mm256_fmadd_pd(b_vec, a_vec, c_vec0);

        b_vec = _mm256_set1_pd(b[1]);
        c_vec1 = _mm256_fmadd_pd(b_vec, a_vec, c_vec1);

        b_vec = _mm256_set1_pd(b[2]);
        c_vec2 = _mm256_fmadd_pd(b_vec, a_vec, c_vec2);

        b_vec = _mm256_set1_pd(b[3]);
        c_vec3 = _mm256_fmadd_pd(b_vec, a_vec, c_vec3);

        a += DGEMM_NR;
        b += DGEMM_MR;
    }

    // Save the results in C
    _mm256_store_pd(c, c_vec0);
    _mm256_store_pd(c + ldc, c_vec1);
    _mm256_store_pd(c + cstep2, c_vec2);
    _mm256_store_pd(c + cstep3, c_vec3);

}

