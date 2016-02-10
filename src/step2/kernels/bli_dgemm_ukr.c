#include <bl_config.h>
#include "bl_dgemm.h"

//micro-panel a is stored in column major, lda=DGEMM_MR.
#define a(i,j) a[ (j)*DGEMM_MR + (i) ]
//micro-panel b is stored in row major, ldb=DGEMM_NR.
#define b(i,j) b[ (i)*DGEMM_NR + (j) ]
//result      c is stored in column major.
#define c(i,j) c[ (j)*ldc + (i) ]

void bli_dgemm_ukr_ref( int    k,
                        double *a,
                        double *b,
                        double *c,
                        dim_t ldc,
                        aux_t* data )
{

    const dim_t m = DGEMM_MR;
    const dim_t n = DGEMM_NR;

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

