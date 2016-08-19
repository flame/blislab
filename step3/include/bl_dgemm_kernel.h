/*
 * --------------------------------------------------------------------------
 * BLISLAB 
 * --------------------------------------------------------------------------
 * Copyright (C) 2016, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * bl_dgemm_kernel.h
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


#ifndef BLISLAB_DGEMM_KERNEL_H
#define BLISLAB_DGEMM_KERNEL_H

#include "bl_config.h"

#include <stdio.h>
#include <immintrin.h> // AVX


// Allow C++ users to include this header file in their source code. However,
// we make the extern "C" conditional on whether we're using a C++ compiler,
// since regular C compilers don't understand the extern "C" construct.
#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned long long dim_t;

typedef union {
    __m256d v;
    __m256i u;
    double d[ 4 ];
} v4df_t;


typedef union {
    __m128i v;
    int d[ 4 ];
} v4li_t;

struct aux_s {
    double *b_next;
    float  *b_next_s;
    char   *flag;
    int    pc;
    int    m;
    int    n;
};
typedef struct aux_s aux_t;

void bl_dgemm_ukr( int k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t* data );

void bl_dgemm_int_8x4( int k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t* data );

void bl_dgemm_asm_8x4( int k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t* data );

void bl_dgemm_asm_12x4( int k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t* data );

void bl_dgemm_asm_8x6( int k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t* data );

void bl_dgemm_asm_6x8( int k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t* data );

void bl_dgemm_asm_4x12( int k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t* data );

static void (*bl_micro_kernel) (
        int    k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t  *aux
        ) = {
        BL_MICRO_KERNEL
        //bl_dgemm_ukr
        //bl_dgemm_int_8x4
        //bl_dgemm_asm_8x4
        //bl_dgemm_asm_8x6
        //bl_dgemm_asm_12x4
};



// End extern "C" construct block.
#ifdef __cplusplus
}
#endif

#endif

