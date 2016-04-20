#include <bl_config.h>
#include "bl_sgemm_kernel.h"

//micro-panel a is stored in column major, lda=SGEMM_MR.
#define a(i,j) a[ (j)*SGEMM_MR + (i) ]
//micro-panel b is stored in row major, ldb=SGEMM_NR.
#define b(i,j) b[ (i)*SGEMM_NR + (j) ]
//result      c is stored in column major.
#define c(i,j) c[ (j)*ldc + (i) ]

void bl_sgemm_ukr_ref( int    k,
                        float *a,
                        float *b,
                        float *c,
                        dim_t ldc,
                        aux_t* data )
{

    const dim_t m = SGEMM_MR;
    const dim_t n = SGEMM_NR;

    dim_t l, j, i;

    for ( l = 0; l < k; ++l )
    {                 
        for ( j = 0; j < n; ++j )
        { 
            for ( i = 0; i < m; ++i )
            { 
                c(i,j) += a(i,l)*b(l,j);
            }
        }
    }

}

