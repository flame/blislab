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
 * test_bl_dgemm.h
 *
 *
 * Purpose:
 * test driver for BLISLAB dgemm routine and reference dgemm routine.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 
 * */


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>

#include <bl_dgemm.h>
#include <bl_dgemm_ref.h>
#include <bl_config.h>

#define USE_SET_DIFF 1
#define TOLERANCE 1E-10

void computeError(
    int    ldc,
    int    ldc_ref,
    int    m,
    int    n,
    double *C,
    double *C_ref
    )
{
    int    i, j;
    for ( i = 0; i < m; i ++ ) {
        for ( j = 0; j < n; j ++ ) {
            if ( fabs( C[ j * ldc + i ] - C_ref[ j * ldc_ref + i ] ) > TOLERANCE ) {
                printf( "C[ %d ][ %d ] != C_gold, %E, %E\n", i, j, C[ j * ldc + i ], C_ref[ j * ldc_ref + i ] );
                break;
            }
        }
    }

}


void test_bl_dgemm(
    int m,
    int n,
    int k
    ) 
{
    int    i, j, p, nx;
    double *XA, *XB, *XC, *XC_ref, *XD;
    double tmp, error, flops;
    double ref_beg, ref_time, bl_dgemm_beg, bl_dgemm_time;
    int    nrepeats;
    int    ldc;
    double ref_rectime, bl_dgemm_rectime;

    XA    = (double*)malloc( sizeof(double) * m * k );
    XB    = (double*)malloc( sizeof(double) * k * n );


    ldc = ( ( m - 1 ) / DGEMM_MR + 1 ) * DGEMM_MR;
    XC     = bl_malloc_aligned( ldc, n + 4, sizeof(double) );
    XC_ref = (double*)malloc( sizeof(double) * m * n );

    nrepeats = 3;

    // Randonly generate points in [ 0, 1 ].
    for ( p = 0; p < k; p ++ ) {
        for ( i = 0; i < m; i ++ ) {
            XA[ p * m + i ] = (double)( rand() % 1000000 ) / 1000000.0;	
            //XA[ p * m + i ] = (double)( p * m + i );	
        }
    }

    for ( j = 0; j < n; j ++ ) {
        for ( p = 0; p < k; p ++ ) {
            XB[ j * k + p ] = (double)( rand() % 1000000 ) / 1000000.0;	
            //XB[ j * k + p ] = (double)( 1.0 );	
        }
    }

    for ( j = 0; j < n; j ++ ) {
        for ( i = 0; i < m; i ++ ) {
            XC[ i + j * ldc ] = (double)( 0.0 );	
            XC_ref[ i + j * m ] = (double)( 0.0 );	
        }
    }

    // Use the same coordinate table
    //XB  = XA;

    for ( i = 0; i < nrepeats; i ++ ) {
        bl_dgemm_beg = omp_get_wtime();
        {
            bl_dgemm(
                    m,
                    n,
                    k,
                    XA,
                    XB,
                    XC,
                    ldc
                    );
        }
        bl_dgemm_time = omp_get_wtime() - bl_dgemm_beg;

        if ( i == 0 ) {
            bl_dgemm_rectime = bl_dgemm_time;
        } else {
            //bl_dgemm_rectime = bl_dgemm_time < bl_dgemm_rectime ? bl_dgemm_time : bl_dgemm_rectime;
            if ( bl_dgemm_time < bl_dgemm_rectime ) {
                bl_dgemm_rectime = bl_dgemm_time;
            }
        }
    }
    //printf("bl_dgemm_rectime: %lf\n", bl_dgemm_rectime);

    for ( i = 0; i < nrepeats; i ++ ) {
        ref_beg = omp_get_wtime();
        {
            bl_dgemm_ref(
                    m,
                    n,
                    k,
                    XA,
                    XB,
                    XC_ref,
                    m
                    );
        }
        ref_time = omp_get_wtime() - ref_beg;

        if ( i == 0 ) {
            ref_rectime = ref_time;
        } else {
            //ref_rectime = ref_time < ref_rectime ? ref_time : ref_rectime;
            if ( ref_time < ref_rectime ) {
                ref_rectime = ref_time;
            }
        }
    }

    //printf("ref_rectime: %lf\n", ref_rectime);

    computeError(
            ldc,
            m,
            m,
            n,
            XC,
            XC_ref
            );

    // Compute overall floating point operations.
    flops = ( m * n / ( 1000.0 * 1000.0 * 1000.0 ) ) * ( 2 * k );

    printf( "%5d\t %5d\t %5d\t %5.2lf\t %5.2lf\n", 
            m, n, k, flops / bl_dgemm_rectime, flops / ref_rectime );

    free( XA     );
    free( XB     );
    free( XC     );
    free( XC_ref );
}


int main( int argc, char *argv[] )
{
    int    m, n, k; 

    if ( argc != 4 ) {
        printf( "Error: require 3 arguments, but only %d provided.\n", argc - 1 );
        exit( 0 );
    }

    sscanf( argv[ 1 ], "%d", &m );
    sscanf( argv[ 2 ], "%d", &n );
    sscanf( argv[ 3 ], "%d", &k );

    test_bl_dgemm( m, n, k );

    return 0;
}
