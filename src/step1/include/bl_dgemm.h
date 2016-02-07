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
 * bl_dgemm.h
 *
 *
 *
 * Purpose:
 * this header file contains all function prototypes.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 
 * */


#include <math.h>
#include <immintrin.h>


typedef unsigned long long dim_t;

struct aux_s {
  double *b_next;
  float  *b_next_s;
  int    ldr;
  char   *flag;
  int    pc;
  int    m;
  int    n;
};
typedef struct aux_s aux_t;

void blis_dgemm(
    int    m,
    int    n,
    int    k,
    double *XA,
    double *XB,
    double *XC,
    int    ldc
    );

double *blis_malloc_aligned(
    int    m,
    int    n,
    int    size
    );

void blis_printmatrix(
    double *A,
    int    lda,
    int    m,
    int    n
    );



