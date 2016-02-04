#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>

#include <blis_dgemm.h>
#include <blis_dgemm_ref.h>

#include <blis_config.h>

#define USE_SET_DIFF 1
#define TOLERANCE 1E-12

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


void test_blis_dgemm(
        int m,
        int n,
        int k
        ) 
{
    int    i, j, p, nx;
    double *XA, *XB, *XC, *XC_ref, *XD;
    double tmp, error, flops;
    double ref_beg, ref_time, blis_dgemm_beg, blis_dgemm_time;
    int    nrepeats;
    int    lda, ldb, ldc, ldc_ref;
    double ref_rectime, blis_dgemm_rectime;

    XA    = (double*)malloc( sizeof(double) * k * m );
    XB    = (double*)malloc( sizeof(double) * k * n );


    lda = k;
    ldb = k;
    ldc = ( ( m - 1 ) / DGEMM_MR + 1 ) * DGEMM_MR;
    ldc_ref = m;
    XC     = blis_malloc_aligned( ldc, n + 4, sizeof(double) );
    XC_ref = (double*)malloc( sizeof(double) * m * n );

    nrepeats = 3;

    // Randonly generate points in [ 0, 1 ].
    for ( i = 0; i < m; i ++ ) {
        for ( p = 0; p < k; p ++ ) {
            //XA[ i * k + p ] = (double)( rand() % 1000000 ) / 1000000.0;	
            XA[ i * lda + p ] = (double)( i * k + p );	
        }
    }
    for ( i = 0; i < n; i ++ ) {
        for ( p = 0; p < k; p ++ ) {
            //XB[ i * k + p ] = (double)( rand() % 1000000 ) / 1000000.0;	
            XB[ i * ldb + p ] = (double)( 1.0 );	
        }
    }

    for ( i = 0; i < ldc_ref; i ++ ) {
        for ( p = 0; p < n; p ++ ) {
            XC_ref[ i + p * ldc_ref ] = (double)( 0.0 );	
        }
    }


    for ( i = 0; i < ldc; i ++ ) {
        for ( p = 0; p < n; p ++ ) {
            XC[ i + p * ldc ] = (double)( 0.0 );	
        }
    }


    // Use the same coordinate table
    //XB  = XA;

    for ( i = 0; i < nrepeats; i ++ ) {
        blis_dgemm_beg = omp_get_wtime();
        {
            blis_dgemm(
                    m,
                    n,
                    k,
                    XA,
                    lda,
                    XB,
                    ldb,
                    XC,
                    ldc
                    );
        }
        blis_dgemm_time = omp_get_wtime() - blis_dgemm_beg;

        if ( i == 0 ) {
            blis_dgemm_rectime = blis_dgemm_time;
        } else {
            //blis_dgemm_rectime = blis_dgemm_time < blis_dgemm_rectime ? blis_dgemm_time : blis_dgemm_rectime;
            if ( blis_dgemm_time < blis_dgemm_rectime ) {
                blis_dgemm_rectime = blis_dgemm_time;
            }
        }
    }
    //printf("blis_dgemm_rectime: %lf\n", blis_dgemm_rectime);

    for ( i = 0; i < nrepeats; i ++ ) {
        ref_beg = omp_get_wtime();
        {
            blis_dgemm_ref(
                    m,
                    n,
                    k,
                    XA,
                    lda,
                    XB,
                    ldb,
                    XC_ref,
                    ldc_ref
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
            m, n, k, flops / blis_dgemm_rectime, flops / ref_rectime );

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

    test_blis_dgemm( m, n, k );

    return 0;
}
