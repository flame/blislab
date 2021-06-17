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

    __m256d c_vec00 = _mm256_load_pd(c);
    __m256d c_vec01 = _mm256_load_pd(c + ldc);
    __m256d c_vec02 = _mm256_load_pd(c + (ldc * 2));
    __m256d c_vec03 = _mm256_load_pd(c + (ldc * 3));

    __m256d c_vec10 = _mm256_load_pd(c + 4);
    __m256d c_vec11 = _mm256_load_pd(c + 4 + ldc);
    __m256d c_vec12 = _mm256_load_pd(c + 4 + (ldc * 2));
    __m256d c_vec13 = _mm256_load_pd(c + 4 + (ldc * 3));

    for (int i = 0; i < k; i++) {
        __m256d a_vec = _mm256_load_pd(a);
        __m256d b_vec_0 = _mm256_set1_pd(b[0]);
        __m256d b_vec_1 = _mm256_set1_pd(b[1]);
        __m256d b_vec_2 = _mm256_set1_pd(b[2]);
        __m256d b_vec_3 = _mm256_set1_pd(b[3]);

        c_vec00 = _mm256_fmadd_pd(b_vec_0, a_vec, c_vec00);

        c_vec01 = _mm256_fmadd_pd(b_vec_1, a_vec, c_vec01);

        c_vec02 = _mm256_fmadd_pd(b_vec_2, a_vec, c_vec02);

        c_vec03 = _mm256_fmadd_pd(b_vec_3, a_vec, c_vec03);

        a_vec = _mm256_load_pd(a + 4);

        c_vec10 = _mm256_fmadd_pd(b_vec_0, a_vec, c_vec10);

        c_vec11 = _mm256_fmadd_pd(b_vec_1, a_vec, c_vec11);

        c_vec12 = _mm256_fmadd_pd(b_vec_2, a_vec, c_vec12);

        c_vec13 = _mm256_fmadd_pd(b_vec_3, a_vec, c_vec13);

        a += DGEMM_MR;
        b += DGEMM_NR;
    }

    // Save the results in C
    _mm256_store_pd(c, c_vec00);
    _mm256_store_pd(c + ldc, c_vec01);
    _mm256_store_pd(c + (ldc * 2), c_vec02);
    _mm256_store_pd(c + (ldc * 3), c_vec03);

    _mm256_store_pd(c + 4, c_vec10);
    _mm256_store_pd(c + 4 + ldc, c_vec11);
    _mm256_store_pd(c + 4 + (ldc * 2), c_vec12);
    _mm256_store_pd(c + 4 + (ldc * 3), c_vec13);
}

