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

#include "bl_dgemm_kernel.h"
#include "bl_dgemm.h"

static inline void packA_mcxkc_d(
        int    m,
        int    k,
        double *XA,
        int    ldXA,
        int    offseta,
        double *packA
        )
{
    int    i, p;
    double *a_pntr[ DGEMM_MR ];

    for ( i = 0; i < m; i ++ ) {
        a_pntr[ i ] = XA + ( offseta + i );
    }

    for ( i = m; i < DGEMM_MR; i ++ ) {
        a_pntr[ i ] = XA + ( offseta + 0 );
    }

    for ( p = 0; p < k; p ++ ) {
        for ( i = 0; i < DGEMM_MR; i ++ ) {
            *packA = *a_pntr[ i ];
            packA ++;
            a_pntr[ i ] = a_pntr[ i ] + ldXA;
        }
    }
}


/*
 * --------------------------------------------------------------------------
 */

static inline void packB_kcxnc_d(
        int    n,
        int    k,
        double *XB,
        int    ldXB, // ldXB is the original k
        int    offsetb,
        double *packB
        )
{
    int    j, p; 
    double *b_pntr[ DGEMM_NR ];

    for ( j = 0; j < n; j ++ ) {
        b_pntr[ j ] = XB + ldXB * ( offsetb + j );
    }

    for ( j = n; j < DGEMM_NR; j ++ ) {
        b_pntr[ j ] = XB + ldXB * ( offsetb + 0 );
    }

    for ( p = 0; p < k; p ++ ) {
        for ( j = 0; j < DGEMM_NR; j ++ ) {
            *packB ++ = *b_pntr[ j ] ++;
        }
    }
}

/*
 * --------------------------------------------------------------------------
 */
void bl_macro_kernel(
        int    m,
        int    n,
        int    k,
        double *packA,
        double *packB,
        double *C,
        int    ldc
        )
{
    int bl_ic_nt;
    int    i, ii, j;
    aux_t  aux;
    char *str;

    aux.b_next = packB;

    // We can also parallelize with OMP here.
    // sequential is the default situation
    bl_ic_nt = 4;

    #pragma omp parallel for num_threads( bl_ic_nt ) private( j, i, aux )
    for ( j = 0; j < n; j += DGEMM_NR ) {                        // 2-th loop around micro-kernel
        aux.n  = min( n - j, DGEMM_NR );
        for ( i = 0; i < m; i += DGEMM_MR ) {                    // 1-th loop around micro-kernel
            aux.m = min( m - i, DGEMM_MR );
            if ( i + DGEMM_MR >= m ) {
                aux.b_next += DGEMM_NR * k;
            }

            ( *bl_micro_kernel ) (
                    k,
                    &packA[ i * k ],
                    &packB[ j * k ],
                    &C[ j * ldc + i ],
                    (unsigned long long) ldc,
                    &aux
                    );
        }                                                        // 1-th loop around micro-kernel
    }                                                            // 2-th loop around micro-kernel
}

// C must be aligned
void bl_dgemm(
        int    m,
        int    n,
        int    k,
        double *XA,
        int    lda,
        double *XB,
        int    ldb,
        double *C,        // must be aligned
        int    ldc        // ldc must also be aligned
        )
{
    int    i, j, p, bl_ic_nt;
    int    ic, ib, jc, jb, pc, pb;
    int    ir, jr;
    double *packA, *packB;
    char   *str;

    // Early return if possible
    if ( m == 0 || n == 0 || k == 0 ) {
        printf( "bl_dgemm(): early return\n" );
        return;
    }

    // sequential is the default situation
    bl_ic_nt = 4;

    // Allocate packing buffers
    packA  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_MC + 1 ) * bl_ic_nt, sizeof(double) );
    packB  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_NC + 1 )           , sizeof(double) );

    for ( jc = 0; jc < n; jc += DGEMM_NC ) {                                       // 5-th loop around micro-kernel
        jb = min( n - jc, DGEMM_NC );
        for ( pc = 0; pc < k; pc += DGEMM_KC ) {                                   // 4-th loop around micro-kernel
            pb = min( k - pc, DGEMM_KC );

            #pragma omp parallel for num_threads( bl_ic_nt ) private( jr )
            for ( j = 0; j < jb; j += DGEMM_NR ) {
                packB_kcxnc_d(
                        min( jb - j, DGEMM_NR ),
                        pb,
                        &XB[ pc ],
                        ldb,
                        jc + j,
                        &packB[ j * pb ]
                        );
            }

            //#pragma omp parallel for num_threads( bl_ic_nt ) private( ic, ib, i, ir )
            #pragma omp parallel num_threads( bl_ic_nt ) private( ic, ib, i, ir )
            {
                int     tid      = omp_get_thread_num();
                int     my_start;
                int     my_end;

                bl_get_range( m, DGEMM_MR, &my_start, &my_end );

                for ( ic = my_start; ic < my_end; ic += DGEMM_MC ) {              // 3-rd loop around micro-kernel

                    ib = min( my_end - ic, DGEMM_MC );

                    for ( i = 0; i < ib; i += DGEMM_MR ) {
                        packA_mcxkc_d(
                                min( ib - i, DGEMM_MR ),
                                pb,
                                &XA[ pc * lda ],
                                lda,
                                ic + i,
                                &packA[ tid * DGEMM_MC * pb + i * pb ]
                                );
                    }

                    bl_macro_kernel(
                            ib,
                            jb,
                            pb,
                            packA  + tid * DGEMM_MC * pb,
                            packB,
                            &C[ jc * ldc + ic ], 
                            ldc
                            );

                }                                                                // End 3.rd loop around micro-kernel

            }
        }                                                                        // End 4.th loop around micro-kernel
    }                                                                            // End 5.th loop around micro-kernel

    free( packA );
    free( packB );
}

