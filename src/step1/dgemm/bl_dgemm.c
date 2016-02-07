/*
 * --------------------------------------------------------------------------
 * BLISLAB 
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
 * bl_dgemm.c
 *
 *
 * Purpose:
 * this is the main file of blislab dgemm.
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
#include <bl_dgemm.h>
#define min( i, j ) ( (i)<(j) ? (i): (j) )

#include <bl_config.h>


// C must be aligned
void bl_dgemm(
    int    m,
    int    n,
    int    k,
    double *A,
    double *B,
    double *C,        // must be aligned
    int    ldc        // ldc must also be aligned
)
{
  int    i, j, p, bl_ic_nt;
  int    ic, ib, jc, jb, pc, pb;
  int    ir, jr;

  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) {
    printf( "bl_dgemm(): early return\n" );
    return;
  }

  #pragma omp parallel for private( i, p )
  for ( j = 0; j < n; j ++ ) {                  // 2-th loop

    for ( p = 0; p < k; p ++ ) {                // 1-th loop

      for ( i = 0; i < m; i ++ ) {              // 0-th loop

          C[ j * ldc + i ] += A[ p * m + i ] * B[ j * k + p ];

      }                                         // End 0.th loop
    }                                           // End 1.th loop
  }                                             // End 2.th loop

}


