#include <immintrin.h> // AVX
#include <math.h>

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h> 

#include <omp.h>

#include <assert.h>

typedef union {
    __m256d v;
    double d[ 4 ];
} v4df_t;


#define TOLERANCE 1E-10
void computeError(
        int    n,
        double *y,
        double *y_ref
        )
{
    int    i;
    for ( i = 0; i < n; i ++ ) {
        if ( fabs( y[ i ] - y_ref[ i ] > TOLERANCE ) ) {
            printf( "y[ %d ] != y_ref, %E, %E\n", i, y[ i ], y_ref[ i ] );
            break;
        }
    }
}

void bl_daxpy_int_4x1(
        double *alpha,
        double *x,
        double *y
        )
{
    v4df_t alphav, y03v, x03v;
    
    alphav.v = _mm256_setzero_pd(); // set alphav to 0
    y03v.v   = _mm256_setzero_pd(); // set y03v   to 0
    x03v.v   = _mm256_setzero_pd(); // set x03v   to 0

    alphav.v = _mm256_broadcast_sd( (double*)alpha       ); // broadcast alpha to alphav.v
    x03v.v   = _mm256_load_pd(      (double*)x           ); // load x to x03v.v
    y03v.v   = _mm256_load_pd(      (double*)y           ); // load y to y03v.v

    y03v.v   = _mm256_fmadd_pd( alphav.v, x03v.v, y03v.v ); // y03v.v := alphav.v * x03v.v + y03v.v (fma)

    _mm256_store_pd( y, y03v.v );  // store back y03v.v to y03v
}

void bl_daxpy_asm_4x1(
        double *alpha,
        double *x,
        double *y
        )
{
    __asm__ volatile
    (
	"                                            \n\t"
	"movq                %1, %%rax               \n\t" // load address of x.              ( v )
	"movq                %2, %%rbx               \n\t" // load address of y.              ( v )
	"movq                %0, %%rcx               \n\t" // load address of alpha.          ( s )
	"                                            \n\t"
    "vxorpd    %%ymm0,  %%ymm0,  %%ymm0          \n\t" // set ymm0 to 0                   ( v )
    "vxorpd    %%ymm1,  %%ymm1,  %%ymm1          \n\t" // set ymm1 to 0                   ( v )
    "vxorpd    %%ymm2,  %%ymm2,  %%ymm2          \n\t" // set ymm2 to 0                   ( v )
	"                                            \n\t"
    "vmovapd   0 * 32(%%rax), %%ymm0             \n\t" // load x
    "vmovapd   0 * 32(%%rbx), %%ymm1             \n\t" // load y
	"                                            \n\t"
	"vbroadcastsd       0 *  8(%%rcx), %%ymm2    \n\t" // load alpha, broacast to ymm2
	"vfmadd231pd       %%ymm2, %%ymm0, %%ymm1    \n\t" // y := alpha * x + y (fma)
	"vmovaps           %%ymm1, 0 * 32(%%rbx)     \n\t" // store back y
	"                                            \n\t"
	".DDONE:                                     \n\t"
	"                                            \n\t"
	: // output operands (none)
	: // input operands
	  "m" (alpha),        // 0
	  "m" (x),            // 1
	  "m" (y)             // 2
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	);
}

void bl_daxpy(
        double *alpha,
        double *x,
        double *y,
        int n
        )
{
    int i;
    for ( i = 0; i < n; i += 4 ) {
        bl_daxpy_asm_4x1( alpha, x, y );
        //bl_daxpy_int_4x1( alpha, x, y );
        x += 4;
        y += 4;
    }
}

void bl_daxpy_ref(
        double *alpha,
        double *x,
        double *y,
        int n
        )
{
    int i;
    for ( i = 0; i < n; i ++ ) {
        y[ i ] = *alpha * x[ i ] + y[ i ];
    }
}



void test_bl_daxpy(
        int n
        )
{
    double *x, *y, *y_ref, *alpha;
    double flops;
    int err;
    int i, nrepeats;
    double ref_beg, ref_time, bl_daxpy_beg, bl_daxpy_time;
    double ref_rectime, bl_daxpy_rectime;

    nrepeats = 3;
    assert( n % 4 == 0 );

    //Memory Alighment Malloc space for x, y.
    err = posix_memalign( (void **)&x, (size_t)32, sizeof(double) * n );
    if ( err ) {
        printf( "bl_malloc_aligned(): posix_memalign() failures" );
        exit( 1 );    
    }
    err = posix_memalign( (void **)&y, (size_t)32, sizeof(double) * n );
    if ( err ) {
        printf( "bl_malloc_aligned(): posix_memalign() failures" );
        exit( 1 );    
    }
    err = posix_memalign( (void **)&y_ref, (size_t)32, sizeof(double) * n );
    if ( err ) {
        printf( "bl_malloc_aligned(): posix_memalign() failures" );
        exit( 1 );    
    }
    alpha = (double*)malloc( sizeof(double) * 1 );


    srand (time(NULL));

    for( i = 0; i < n; i ++ ) {
        x[ i ] = (double)( drand48() );
        //x[ i ] = (double)( 1.0 );
    }

    for( i = 0; i < n; i ++ ) {
        y[ i ] = (double)( drand48() );
        //y[ i ] = (double)( 2.0 );
        y_ref[ i ] = y[ i ];
    }

    *alpha = (double)( drand48() );
    //*alpha = (double)( 0.5 );


    for ( i = 0; i < nrepeats; i ++ ) {
        bl_daxpy_beg = omp_get_wtime();
        bl_daxpy( alpha, x, y, n );
        bl_daxpy_time = omp_get_wtime() - bl_daxpy_beg;

        if ( i == 0 ) {
            bl_daxpy_rectime = bl_daxpy_time;
        } else {
            bl_daxpy_rectime = bl_daxpy_time < bl_daxpy_rectime ? bl_daxpy_time : bl_daxpy_rectime;
        }
    }

    for ( i = 0; i < nrepeats; i ++ ) {
        ref_beg = omp_get_wtime();
        bl_daxpy_ref( alpha, x, y_ref, n );
        ref_time = omp_get_wtime() - ref_beg;

        if ( i == 0 ) {
            ref_rectime = ref_time;
        } else {
            ref_rectime = ref_time < ref_rectime ? ref_time : ref_rectime;
        }

    }

    computeError( n, y, y_ref );

    //bl_daxpy( alpha, x, y, n );
    //bl_daxpy_ref( alpha, x, y );

    //printf("y:\t");
    //for ( i = 0; i < n; i ++ ) {
    //    printf( "%lf ", y[ i ] );
    //}
    //printf("\n");

    //printf("y_ref:\t");
    //for ( i = 0; i < n; i ++ ) {
    //    printf( "%lf ", y_ref[ i ] );
    //}
    //printf("\n");
    
    // Compute overall floating point operations.
    flops = 2.0 * n / ( 1000.0 * 1000.0 * 1000.0 );

    printf( "%5d\t %5.2lf\t %5.2lf\n", n, flops/ bl_daxpy_rectime, flops / ref_rectime );

    free( x );
    free( y );
    free( y_ref );
    free( alpha );

}

int main( int argc, char *argv[] ) {
    int n;

    n = 40000;

    test_bl_daxpy( n );

    return 0;
}

