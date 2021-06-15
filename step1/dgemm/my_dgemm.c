/*
 * --------------------------------------------------------------------------
 * BLISLAB 
 * --------------------------------------------------------------------------
 * Copyright (C) 2016, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * bl_dgemm.c
 *
 *
 * Purpose:
 * this is the main file of blislab dgemm.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 
 * */


#include "bl_dgemm.h"


void AddDot_MRxNR(int k, double *A, int lda, double *B, int ldb, double *C, int ldc) {
    // 9 registers to store a 3x3 area of the C matrix
    register double c00 = 0.0, c01 = 0.0, c02 = 0.0;
    register double c10 = 0.0, c11 = 0.0, c12 = 0.0;
    register double c20 = 0.0, c21 = 0.0, c22 = 0.0;

    // Using this value inside the loop
    int ldb2 = ldb * 2;

    double *Bp = B;
    double *Ap = A;

    // Computing the dot product of 3 rows in A with 3 columns in B
    for (int p = 0; p < k; p++) {
        // For each index in from 0 to k, we will compute all of the values for the 3x3 dot products
        // Save the values for the products in index p (there are 3 in A and 3 in B)
        register double a0 = *Ap;
        register double b0 = *Bp;
        register double b1 = *(Bp + ldb);
        register double b2 = *(Bp + ldb2);

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
        Bp++;
        Ap += lda;
    }

    // Save the results in C
    register double *cp = C;
    *cp += c00;
    *(cp + 1) += c10;
    *(cp + 2) += c20;

    register double *cp1 = C + ldc;
    *cp1 += c01;
    *(cp1 + 1) += c11;
    *(cp1 + 2) += c21;

    register double *cp2 = C + (2 * ldc);
    *cp2 += c02;
    *(cp2 + 1) += c12;
    *(cp2 + 2) += c22;
}

void bl_dgemm(
        int m,
        int n,
        int k,
        double *A,
        int lda,
        double *B,
        int ldb,
        double *C,        // must be aligned
        int ldc        // ldc must also be aligned
) {
    int i, j, p;
    int ir, jr;

    // Early return if possible
    if (m == 0 || n == 0 || k == 0) {
        printf("bl_dgemm(): early return\n");
        return;
    }

    for (j = 0; j < n; j += DGEMM_NR) {          // Start 2-nd loop
        for (i = 0; i < m; i += DGEMM_MR) {      // Start 1-st loop
            AddDot_MRxNR(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);

        }                                          // End   1-st loop
    }                                              // End   2-nd loop

}


