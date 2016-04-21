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
 * bl_dgemm_util.c
 *
 *
 * Purpose:
 * Utility routines (Mem allocation, Print, etc.) that will come in handy later.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 
 * */

#include "bl_dgemm.h"

/*
 *
 *
 */ 
double *bl_malloc_aligned(
        int    m,
        int    n,
        int    size
        )
{
    double *ptr;
    int    err;

    err = posix_memalign( (void**)&ptr, (size_t)GEMM_SIMD_ALIGN_SIZE, size * m * n );

    if ( err ) {
        printf( "bl_malloc_aligned(): posix_memalign() failures" );
        exit( 1 );    
    }

    return ptr;
}



/*
 *
 *
 */
void bl_dgemm_printmatrix(
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

/*
 * The timer functions are copied directly from BLIS 0.2.0
 *
 */
static double gtod_ref_time_sec = 0.0;

double bl_clock( void )
{
	return bl_clock_helper();
}

#if BLIS_OS_WINDOWS
// --- Begin Windows build definitions -----------------------------------------

double bl_clock_helper()
{
    LARGE_INTEGER clock_freq = {0};
    LARGE_INTEGER clock_val;
    BOOL          r_val;

    r_val = QueryPerformanceFrequency( &clock_freq );

    if ( r_val == 0 )
    {
        fprintf( stderr, "\nblislab: %s (line %lu):\nblislab: %s \n", __FILE__, __LINE__, "QueryPerformanceFrequency() failed" );
        fflush( stderr );
        abort();
    }

    r_val = QueryPerformanceCounter( &clock_val );

    if ( r_val == 0 )
    {
        fprintf( stderr, "\nblislab: %s (line %lu):\nblislab: %s \n", __FILE__, __LINE__, "QueryPerformanceFrequency() failed" );
        fflush( stderr );
        abort();
    }

    return ( ( double) clock_val.QuadPart / ( double) clock_freq.QuadPart );
}

// --- End Windows build definitions -------------------------------------------
//#elif BLIS_OS_OSX
#elif defined(__APPLE__) || defined(__MACH__)
// --- Begin OSX build definitions -------------------------------------------

double bl_clock_helper()
{
    mach_timebase_info_data_t timebase;
    mach_timebase_info( &timebase );

    uint64_t nsec = mach_absolute_time();

    double the_time = (double) nsec * 1.0e-9 * timebase.numer / timebase.denom;

    if ( gtod_ref_time_sec == 0.0 )
        gtod_ref_time_sec = the_time;

    return the_time - gtod_ref_time_sec;
}

// --- End OSX build definitions ---------------------------------------------
#else
// --- Begin Linux build definitions -------------------------------------------

double bl_clock_helper()
{
    double the_time, norm_sec;
    struct timespec ts;

    clock_gettime( CLOCK_MONOTONIC, &ts );

    if ( gtod_ref_time_sec == 0.0 )
        gtod_ref_time_sec = ( double ) ts.tv_sec;

    norm_sec = ( double ) ts.tv_sec - gtod_ref_time_sec;

    the_time = norm_sec + ts.tv_nsec * 1.0e-9;

    return the_time;
}

// --- End Linux build definitions ---------------------------------------------
#endif



