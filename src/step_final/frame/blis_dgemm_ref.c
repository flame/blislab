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
 * blisgemm_ref.c
 *
 *
 * Purpose:
 * implement reference mkl using GEMM (optional) in C.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 
 * */

#include <omp.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>

#include <blis_dgemm.h>
#include <blis_dgemm_ref.h>

#ifdef USE_BLAS
/* 
 * dgemm and sgemm prototypes
 *
 */ 
void dgemm(char*, char*, int*, int*, int*, double*, double*, 
    int*, double*, int*, double*, double*, int*);
void sgemm(char*, char*, int*, int*, int*, float*, float*, 
    int*, float*, int*, float*, float*, int*);
void daxpy(int*, double*, double*, int*, double*, int*); 
#endif

void blis_dgemm_ref(
    int    m,
    int    n,
    int    k,
    double *XA,
    double *XB,
    double *XC,
    int    ldc
    )
{
  // Local variables.
  int    i, j, p;
  double beg, time_collect, time_dgemm, time_square;
  double *As, *Bs, *Cs;
  double alpha = 1.0, beta = 1.0;

  // Sanity check for early return.
  if ( m == 0 || n == 0 || k == 0 ) return;

  // Allocate buffers.
//  As = (double*)malloc( sizeof(double) * m * k );
//  Bs = (double*)malloc( sizeof(double) * n * k );
  As = XA;
  Bs = XB;
  Cs = XC;

  // Compute the inner-product term.
  beg = omp_get_wtime();

#ifdef USE_BLAS

//  int len = k*m;
//  double alpha2 = 1.0;
//  int incx = 1;
//  int incy = 1;
//  daxpy( &len, &alpha2, As, &incx, As, &incy );
  dgemm( "T", "N", &m, &n, &k, &alpha,
        As, &k, Bs, &k, &beta, Cs, &m );
#else
  #pragma omp parallel for private( i, p )
  for ( j = 0; j < n; j ++ ) {
    for ( i = 0; i < m; i ++ ) {
      Cs[ j * m + i ] = 0.0;
      for ( p = 0; p < k; p ++ ) {
        Cs[ j * m + i ] += As[ i * k + p ] * Bs[ j * k + p ];
      }
    }
  }
#endif

  time_dgemm = omp_get_wtime() - beg;
  printf("time_dgemm: %lf\n", time_dgemm);

  printf("%lf GFLOPS\n", 2.0*m*n*k/time_dgemm/1000.0/1000.0/1000.0);

  if ( n <= 32 && m <= 32 ) {
    printf("refC:\n");
    blisgemm_printmatrix( Cs, m, m, n );
  }

  // Free  buffers
//  free( As );
//  free( Bs );

}
