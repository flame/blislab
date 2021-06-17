#include <bl_config.h>
#include "bl_dgemm_kernel.h"

//micro-panel a is stored in column major, lda=DGEMM_MR.
#define a(i, j) a[ (j)*DGEMM_MR + (i) ]
//micro-panel b is stored in row major, ldb=DGEMM_NR.
#define b(i, j) b[ (i)*DGEMM_NR + (j) ]
//result      c is stored in column major.
#define c(i, j) c[ (j)*ldc + (i) ]


void bl_dgemm_ukr(int k,
                  double *a,
                  double *b,
                  double *c,
                  unsigned long long ldc,
                  aux_t *data) {
    int l, i, j;

    // 9 registers to store a 3x3 area of the C matrix
    register double c00 = 0.0, c01 = 0.0, c02 = 0.0;
    register double c10 = 0.0, c11 = 0.0, c12 = 0.0;
    register double c20 = 0.0, c21 = 0.0, c22 = 0.0;

    double *Bp = b;
    double *Ap = a;

    // Computing the dot product of 3 rows in A with 3 columns in B
    for (int p = 0; p < k; p++) {
        // For each index in from 0 to k, we will compute all of the values for the 3x3 dot products
        // Save the values for the products in index p (there are 3 in A and 3 in B)
        register double a0 = *Ap;
        register double b0 = *Bp;
        register double b1 = *(Bp + 1);
        register double b2 = *(Bp + 2);

        // A(0,p) * B(p,0)
        c00 += a0 * b0;
        // A(0,p) * B(p,1)
        c01 += a0 * b1;
        // A(0,p) * B(p,2)
        c02 += a0 * b2;

        register double a1 = *(Ap + 1);
        c10 += a1 * b0;
        c11 += a1 * b1;
        c12 += a1 * b2;

        register double a2 = *(Ap + 2);
        c20 += a2 * b0;
        c21 += a2 * b1;
        c22 += a2 * b2;

        // Advance to the next row in B and the next column in A
        Bp += DGEMM_NR;
        Ap += DGEMM_MR;
    }

    // Save the results in C
    register double *cp = c;
    *cp += c00;
    *(cp + 1) += c10;
    *(cp + 2) += c20;

    register double *cp1 = c + ldc;
    *cp1 += c01;
    *(cp1 + 1) += c11;
    *(cp1 + 2) += c21;

    register double *cp2 = c + (2 * ldc);
    *cp2 += c02;
    *(cp2 + 1) += c12;
    *(cp2 + 2) += c22;
}

