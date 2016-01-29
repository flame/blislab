#include <stdio.h>
//#include <immintrin.h> // AVX

#include <blis_dgemm.h>
//#include <avx_types.h>

#define inc_t unsigned long long 
#define dim_t int

                        //double* restrict alpha,
                        //double* restrict beta,
                        //inc_t rs_c,
                        //inc_t cs_c,
void bli_dgemm_ukr_ref( dim_t k,
                        double *a,
                        double *b,
                        double *c,
                        inc_t ldc,
                        aux_t* data )
{
    const inc_t cs_c = ldc;
    const inc_t rs_c = 1;
    double alpha_val = 1.0, beta_val = 1.0;
    double *alpha, *beta;

    alpha = &alpha_val;
    beta  = &beta_val;

    const dim_t m = 8;
    const dim_t n = 4;
    const inc_t cs_a = 8;
    const inc_t rs_b = 4;
    const inc_t rs_ab = 1;
    const inc_t cs_ab = 8;
    dim_t l, j, i;
    double ab[ 8 * 4 ];
    double ai;
    double bj;

    //printf( "a: " );
    //for ( l = 0; l < k; ++l ) {
    //    for ( i = 0; i < m; ++i ) {
    //        printf( "%lf ", a[ i*1 + l*cs_a ] );
    //    }
    //    printf( "\n" );
    //}
    //printf( "\n" );

    for ( i = 0; i < m * n; ++i )
    {
        ((*(ab + i))) = (0.0);
    }

    for ( l = 0; l < k; ++l )
    {                 
        //double* restrict abij = ab;
        double* abij = ab;
        for ( j = 0; j < n; ++j )
        { 
            bj = *(b + j);
            for ( i = 0; i < m; ++i )
            { 
                ai = *(a + i);
                {
                    (( *abij )) += (( ai )) * (( bj ));
                };
                abij += rs_ab;
            }
        }
        a += cs_a;
        b += rs_b;
    }
    for ( i = 0; i < m * n; ++i )
    {
        ( *(ab + i) ) = ( *alpha ) * ( *(ab + i) );
    }

    if (  (*beta) == 0.0  )
    {
        dim_t i, j;
        for ( j = 0; j < n; ++j )
            for ( i = 0;i < m; ++i )
            {
                (( *(c + i*rs_c + j*cs_c) )) = (( *(ab + i*rs_ab + j*cs_ab) ));
            };
    } else {
        dim_t i, j;
        if (  ((*beta)) == (0.0)  )
        {
            dim_t i, j;
            for ( j = 0;j < n; ++j )
                for ( i = 0;i < m; ++i )
                {
                    (( *(c + i*rs_c + j*cs_c) )) = (( *(ab + i*rs_ab + j*cs_ab) ));
                };
        } else {
            for ( j = 0; j < n; ++j )
                for ( i = 0; i < m; ++i )
                {
                    (( *(c + i*rs_c + j*cs_c) )) = (( *(ab + i*rs_ab + j*cs_ab) )) + (( *(beta) )) * (( *(c + i*rs_c + j*cs_c) ));
                };
        } 
    }

}

