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
 * blisgemm_util.c
 *
 *
 *
 * Purpose:
 * Aligned memory allocation.
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

#include <blis_dgemm.h>
#include <blis_config.h>

/*
 *
 *
 */ 
double *blis_malloc_aligned(
        int    m,
        int    n,
        int    size
        )
{
    double *ptr;
    int    err;

    err = posix_memalign( (void**)&ptr, (size_t)GEMM_SIMD_ALIGN_SIZE, size * m * n );

    if ( err ) {
        printf( "blis_malloc_aligned(): posix_memalign() failures" );
        exit( 1 );    
    }

    return ptr;
}



/*
 *
 *
 */
void blisgemm_printmatrix(
        double *A,
        int    lda,
        int    m,
        int    n
        )
{
    int    i, j;
    for ( i = 0; i < m; i ++ ) {
        for ( j = 0; j < n; j ++ ) {
            printf("%lf\t", A[j * lda + i]);
        }
        printf("\n");
    }
}


