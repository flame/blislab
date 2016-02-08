#include <bl_config.h>
#include <bl_dgemm.h>

//micro-panel a is stored in column major, lda=DGEMM_MR=8
#define A(i,j) a[ (j)*DGEMM_MR + (i) ]
//micro-panel b is stored in row major, ldb=DGEMM_NR=4
#define B(i,j) b[ (i)*DGEMM_NR + (j) ]

#define C(i,j) c[ (j)*ldc + (i) ]

void bli_dgemm_ukr_ref( dim_t k,
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
                C(i,j) += A(i,l)*B(l,j);
            }
        }
    }

}

