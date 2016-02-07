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
 

#include <stdio.h>
#include <omp.h>
#include <bl_dgemm.h>
#define min( i, j ) ( (i)<(j) ? (i): (j) )

#include <bl_config.h>

// C must be aligned
void bl_dgemm(
    int    m,
    int    n,
    int    k,
    double *A,
    double *B,
    double *C,        // must be aligned
    int    ldc        // ldc must also be aligned
)
{
  int    i, j, p, bl_ic_nt;
  int    ic, ib, jc, jb, pc, pb;
  int    ir, jr;

  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) {
    printf( "bl_dgemm(): early return\n" );
    return;
  }

  #pragma omp parallel for private( i, p )
  for ( j = 0; j < n; j ++ ) {                  // 2-th loop

    for ( p = 0; p < k; p ++ ) {                // 1-th loop

      for ( i = 0; i < m; i ++ ) {              // 0-th loop

          C[ j * ldc + i ] += A[ p * m + i ] * B[ j * k + p ];

      }                                         // End 0.th loop
    }                                           // End 1.th loop
  }                                             // End 2.th loop

}


