/*
 * --------------------------------------------------------------------------
 * BLISGEMM 
 * --------------------------------------------------------------------------
 * Copyright (C) 2015, The University of Texas at Austin
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 *
 * blisgemm.c
 *
 *
 * Purpose:
 * this is the main file of blis gemm.
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
#include <blis_dgemm.h>
#define min( i, j ) ( (i)<(j) ? (i): (j) )

#include <blis_config.h>
#include <blis_dgemm_kernel.h>

inline void packA_kcxmc_d(
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
        a_pntr[ i ] = XA + ldXA * ( offseta + i );
    }

    for ( i = m; i < DGEMM_MR; i ++ ) {
        a_pntr[ i ] = XA + ldXA * ( offseta + 0 );
    }

    for ( p = 0; p < k; p ++ ) {
        for ( i = 0; i < DGEMM_MR; i ++ ) {
            *packA ++ = *a_pntr[ i ] ++;
        }
    }
}
/*
 * --------------------------------------------------------------------------
 */

inline void packB_kcxnc_d(
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
void blis_macro_kernel(
        int    m,
        int    n,
        int    k,
        double *packA,
        double *packB,
        double *C,
        int    ldc,
        int    pc,
        int    lastiter
        )
{
    int blis_ic_nt;
    int    i, ii, j;
    aux_t  aux;
    char *str;

    aux.pc     = pc;
    aux.b_next = packB;


    //printf( "here, pc = %d, last = %d, ldc = %d, m = %d, n = %d, k %d\n", 
    //    pc, lastiter, ldc, m, n , k );


    //// sequential is the default situation
    //blis_ic_nt = 1;
    //// check the environment variable
    //str = getenv( "BLISGEMM_IC_NT" );
    //if ( str != NULL ) {
    //    blis_ic_nt = (int)strtol( str, NULL, 10 );
    //}


    // We can also parallelize with OMP here.
    //#pragma omp parallel for num_threads( blis_ic_nt ) private( j, i, aux )
    for ( j = 0; j < n; j += DGEMM_NR ) {                      // 2-th loop around micro-kernel
        aux.n  = min( n - j, DGEMM_NR );
        for ( i = 0; i < m; i += DGEMM_MR ) {                    // 1-th loop around micro-kernel
            aux.m = min( m - i, DGEMM_MR );
            if ( i + DGEMM_MR >= m ) {
                aux.b_next += DGEMM_NR * k;
            }

            ( *blis_micro_kernel ) (
                    k,
                    &packA[ i * k ],
                    &packB[ j * k ],
                    &C[ j * ldc + i ],
                    (unsigned long long) ldc,
                    //(unsigned long long) lastiter,
                    &aux
                    );
        }                                                        // 1-th loop around micro-kernel
    }                                                          // 2-th loop around micro-kernel
}

// C must be aligned
void blis_dgemm(
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
    int    i, j, p, blis_ic_nt;
    int    ic, ib, jc, jb, pc, pb;
    int    ir, jr;
    double *packA, *packB;
    char   *str;

    // Early return if possible
    if ( m == 0 || n == 0 || k == 0 ) {
        printf( "blis_dgemm(): early return\n" );
        return;
    }

    // sequential is the default situation
    blis_ic_nt = 1;
    // check the environment variable
    str = getenv( "BLISGEMM_IC_NT" );
    if ( str != NULL ) {
        blis_ic_nt = (int)strtol( str, NULL, 10 );
    }

    // Allocate packing buffers
    packA  = blis_malloc_aligned( DGEMM_KC, ( DGEMM_MC + 1 ) * blis_ic_nt, sizeof(double) );
    packB  = blis_malloc_aligned( DGEMM_KC, ( DGEMM_NC + 1 )            , sizeof(double) );

    for ( jc = 0; jc < n; jc += DGEMM_NC ) {                  // 5-th loop around micro-kernel
        jb = min( n - jc, DGEMM_NC );
        for ( pc = 0; pc < k; pc += DGEMM_KC ) {                // 4-th loop around micro-kernel
            pb = min( k - pc, DGEMM_KC );

            #pragma omp parallel for num_threads( blis_ic_nt ) private( jr )
            for ( j = 0; j < jb; j += DGEMM_NR ) {

                packB_kcxnc_d(
                        min( jb - j, DGEMM_NR ),
                        pb,
                        &XB[ pc ],
                        k, // should be ldXB instead
                        jc + j,
                        &packB[ j * pb ]
                        );
            }

            #pragma omp parallel for num_threads( blis_ic_nt ) private( ic, ib, i, ir )
            for ( ic = 0; ic < m; ic += DGEMM_MC ) {              // 3-th loop around micro-kernel
                int     tid = omp_get_thread_num();

                ib = min( m - ic, DGEMM_MC );
                for ( i = 0; i < ib; i += DGEMM_MR ) {

                    packA_kcxmc_d(
                            min( ib - i, DGEMM_MR ),
                            pb,
                            &XA[ pc ],
                            k,
                            ic + i,
                            &packA[ tid * DGEMM_MC * pb + i * pb ]
                            );
                }

                blis_macro_kernel(
                        ib,
                        jb,
                        pb,
                        packA  + tid * DGEMM_MC * pb,
                        packB,
                        &C[ jc * ldc + ic ], 
                        ldc,
                        pc,
                        ( pc + DGEMM_KC >= k )
                        );

            }                                                    // End 3.th loop around micro-kernel
        }                                                      // End 4.th loop around micro-kernel
    }                                                        // End 5.th loop around micro-kernel

    free( packA );
    free( packB );
}


