#include <stdio.h>
//#include <immintrin.h> // AVX

#include <blis_dgemm.h>
//#include <avx_types.h>

#define inc_t unsigned long long 


void bli_dgemm_ukr_ref( dim_t k,
                        double *a,
                        double *b,
                        double *c,
                        inc_t ldc,
                        aux_t* data )
{

    const dim_t m = 8;
    const dim_t n = 4;

    const inc_t cs_a = 8;
    const inc_t rs_b = 4;

    dim_t l, j, i;

    for ( l = 0; l < k; ++l )
    {                 
        for ( j = 0; j < n; ++j )
        { 
            for ( i = 0; i < m; ++i )
            { 
                c[i+j*ldc] = c[i+j*ldc] + a[i]*b[j];
            }
        }
        a += cs_a;
        b += rs_b;
    }

}

