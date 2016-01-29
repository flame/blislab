#include <stdio.h>
#include <immintrin.h> // AVX

#include <blis_dgemm.h>
#include <avx_types.h>


void sq2nrm_asm_d8x4(
    int    k,
    double *a,
//    double *aa,
    double *b,
//    double *bb,
    double *c,
    unsigned long long ldc,
//    unsigned long long last,
    aux_t  *aux
    )
{
  unsigned long long k_iter = k / 4;
  unsigned long long k_left = k % 4;
  unsigned long long pc     = aux->pc;
  
  //printf( "%ld\n", last );

	__asm__ volatile
	(
	"                                            \n\t"
	"                                            \n\t"
	"movq                %2, %%rax               \n\t" // load address of a.              ( v )
	"movq                %3, %%rbx               \n\t" // load address of b.              ( v )
	"movq                %5, %%r15               \n\t" // load address of b_next.         ( v )
	"addq          $-4 * 64, %%r15               \n\t" //                                 ( ? )
	"                                            \n\t"
	"vmovapd   0 * 32(%%rax), %%ymm0             \n\t" // initialize loop by pre-loading
	"vmovapd   0 * 32(%%rbx), %%ymm2             \n\t" // elements of a and b.
  "vpermilpd  $0x5, %%ymm2, %%ymm3             \n\t"
  "                                            \n\t"
  "                                            \n\t"
  "movq                %4, %%rcx               \n\t" // load address of c
  "movq               %7, %%rdi               \n\t" // load ldc
  "leaq        (,%%rdi,8), %%rdi               \n\t" // ldc * sizeof(double)
  "leaq   (%%rcx,%%rdi,2), %%r10               \n\t" // load address of c + 2 * ldc;
  "                                            \n\t"
  "                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
  "prefetcht0   3 * 8(%%rcx)                   \n\t" // prefetch c + 0 * ldc
  "prefetcht0   3 * 8(%%rcx,%%rdi)             \n\t" // prefetch c + 1 * ldc
  "prefetcht0   3 * 8(%%r10)                   \n\t" // prefetch c + 2 * ldc
  "prefetcht0   3 * 8(%%r10,%%rdi)             \n\t" // prefetch c + 3 * ldc
  "                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vxorpd    %%ymm8,  %%ymm8,  %%ymm8          \n\t" // set ymm8 to 0                   ( v )
	"vxorpd    %%ymm9,  %%ymm9,  %%ymm9          \n\t"
	"vxorpd    %%ymm10, %%ymm10, %%ymm10         \n\t"
	"vxorpd    %%ymm11, %%ymm11, %%ymm11         \n\t"
	"vxorpd    %%ymm12, %%ymm12, %%ymm12         \n\t"
	"vxorpd    %%ymm13, %%ymm13, %%ymm13         \n\t"
	"vxorpd    %%ymm14, %%ymm14, %%ymm14         \n\t"
	"vxorpd    %%ymm15, %%ymm15, %%ymm15         \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq      %0, %%rsi                         \n\t" // i = k_iter;                     ( v )
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.        ( v )
	"je     .DCONSIDKLEFT                        \n\t" // if i == 0, jump to code that    ( v )
	"                                            \n\t" // contains the k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".DLOOPKITER:                                \n\t" // MAIN LOOP
	"                                            \n\t"
	"addq         $4 * 4 * 8,  %%r15             \n\t" // b_next += 4*4 (unroll x nr)     ( v )
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"vmovapd   1 * 32(%%rax),  %%ymm1            \n\t" // preload a47 for iter 0
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t" // ymm6 ( c_tmp0 ) = ymm0 ( a03 ) * ymm2( b0 )
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t" // ymm4 ( b0x3_0 )
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t" // ymm7 ( c_tmp1 ) = ymm0 ( a03 ) * ymm3( b0x5 )
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t" // ymm5 ( b0x3_1 )
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \n\t" // ymm15 ( c_03_0 ) += ymm6( c_tmp0 )
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \n\t" // ymm13 ( c_03_1 ) += ymm7( c_tmp1 )
	"                                            \n\t"
	"prefetcht0  16 * 32(%%rax)                  \n\t" // prefetch a03 for iter 1
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovapd   1 * 32(%%rbx),  %%ymm2            \n\t" // preload b for iter 1
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \n\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd   2 * 32(%%rax),  %%ymm0            \n\t" // preload a03 for iter 1
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \n\t"
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \n\t"
	"prefetcht0   0 * 32(%%r15)                  \n\t" // prefetch b_next[0*4]
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \n\t"
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 1
	"vmovapd   3 * 32(%%rax),  %%ymm1            \n\t" // preload a47 for iter 1
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \n\t"
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	"prefetcht0  18 * 32(%%rax)                  \n\t" // prefetch a for iter 9  ( ? )
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovapd   2 * 32(%%rbx),  %%ymm2            \n\t" // preload b for iter 2 
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \n\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd   4 * 32(%%rax),  %%ymm0            \n\t" // preload a03 for iter 2
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \n\t"
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \n\t"
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 2
	"vmovapd   5 * 32(%%rax),  %%ymm1            \n\t" // preload a47 for iter 2
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \n\t"
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	"prefetcht0  20 * 32(%%rax)                  \n\t" // prefetch a for iter 10 ( ? )
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovapd   3 * 32(%%rbx),  %%ymm2            \n\t" // preload b for iter 3
	"addq         $4 * 4 * 8,  %%rbx             \n\t" // b += 4*4 (unroll x nr)
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \n\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd   6 * 32(%%rax),  %%ymm0            \n\t" // preload a03 for iter 3
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \n\t"
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \n\t"
	"prefetcht0   2 * 32(%%r15)                  \n\t" // prefetch b_next[2*4]
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \n\t"
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 3
	"vmovapd   7 * 32(%%rax),  %%ymm1            \n\t" // preload a47 for iter 3
	"addq         $4 * 8 * 8,  %%rax             \n\t" // a += 4*8 (unroll x mr)
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \n\t"
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	"prefetcht0  14 * 32(%%rax)                  \n\t" // prefetch a for iter 11 ( ? )
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovapd   0 * 32(%%rbx),  %%ymm2            \n\t" // preload b for iter 4
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \n\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd   0 * 32(%%rax),  %%ymm0            \n\t" // preload a03 for iter 4
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \n\t"
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \n\t"
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .DLOOPKITER                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DCONSIDKLEFT:                              \n\t"
	"                                            \n\t"
	"movq      %1, %%rsi                         \n\t" // i = k_left;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .DPOSTACCUM                          \n\t" // if i == 0, we're done; jump to end.
	"                                            \n\t" // else, we prepare to enter k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".DLOOPKLEFT:                                \n\t" // EDGE LOOP
	"                                            \n\t"
	"vmovapd   1 * 32(%%rax),  %%ymm1            \n\t" // preload a47 
	"addq         $8 * 1 * 8,  %%rax             \n\t" // a += 8 (1 x mr)
	"vmulpd           %%ymm0,  %%ymm2, %%ymm6    \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2, %%ymm4    \n\t"
	"vmulpd           %%ymm0,  %%ymm3, %%ymm7    \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3, %%ymm5    \n\t"
	"vaddpd           %%ymm15, %%ymm6, %%ymm15   \n\t"
	"vaddpd           %%ymm13, %%ymm7, %%ymm13   \n\t"
	"                                            \n\t"
	"prefetcht0  14 * 32(%%rax)                  \n\t" // prefetch a03 for iter 7 later ( ? )
	"vmulpd           %%ymm1,  %%ymm2, %%ymm6    \n\t"
	"vmovapd   1 * 32(%%rbx),  %%ymm2            \n\t"
	"addq         $4 * 1 * 8,  %%rbx             \n\t" // b += 4 (1 x nr)
	"vmulpd           %%ymm1,  %%ymm3, %%ymm7    \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6, %%ymm14   \n\t"
	"vaddpd           %%ymm12, %%ymm7, %%ymm12   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4, %%ymm6    \n\t"
	"vmulpd           %%ymm0,  %%ymm5, %%ymm7    \n\t"
	"vmovapd   0 * 32(%%rax),  %%ymm0            \n\t"
	"vaddpd           %%ymm11, %%ymm6, %%ymm11   \n\t"
	"vaddpd           %%ymm9,  %%ymm7, %%ymm9    \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4, %%ymm6    \n\t"
	"vmulpd           %%ymm1,  %%ymm5, %%ymm7    \n\t"
	"vaddpd           %%ymm10, %%ymm6, %%ymm10   \n\t"
	"vaddpd           %%ymm8,  %%ymm7, %%ymm8    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .DLOOPKLEFT                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DPOSTACCUM:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
	"                                            \n\t" // ( ab00  ( ab01  ( ab02  ( ab03
	"                                            \n\t" //   ab11    ab10    ab13    ab12  
	"                                            \n\t" //   ab22    ab23    ab20    ab21
	"                                            \n\t" //   ab33 )  ab32 )  ab31 )  ab30 )
	"                                            \n\t"
	"                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
	"                                            \n\t" // ( ab40  ( ab41  ( ab42  ( ab43
	"                                            \n\t" //   ab51    ab50    ab53    ab52  
	"                                            \n\t" //   ab62    ab63    ab60    ab61
	"                                            \n\t" //   ab73 )  ab72 )  ab71 )  ab70 )
	"                                            \n\t"
	"vmovapd          %%ymm15, %%ymm7            \n\t"
	"vshufpd    $0xa, %%ymm15, %%ymm13, %%ymm15  \n\t"
	"vshufpd    $0xa, %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	"vmovapd          %%ymm11, %%ymm7            \n\t"
	"vshufpd    $0xa, %%ymm11, %%ymm9,  %%ymm11  \n\t"
	"vshufpd    $0xa, %%ymm9,  %%ymm7,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmovapd          %%ymm14, %%ymm7            \n\t"
	"vshufpd    $0xa, %%ymm14, %%ymm12, %%ymm14  \n\t"
	"vshufpd    $0xa, %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmovapd          %%ymm10, %%ymm7            \n\t"
	"vshufpd    $0xa, %%ymm10, %%ymm8,  %%ymm10  \n\t"
	"vshufpd    $0xa, %%ymm8,  %%ymm7,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
	"                                            \n\t" // ( ab01  ( ab00  ( ab03  ( ab02
	"                                            \n\t" //   ab11    ab10    ab13    ab12  
	"                                            \n\t" //   ab23    ab22    ab21    ab20
	"                                            \n\t" //   ab33 )  ab32 )  ab31 )  ab30 )
	"                                            \n\t"
	"                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
	"                                            \n\t" // ( ab41  ( ab40  ( ab43  ( ab42
	"                                            \n\t" //   ab51    ab50    ab53    ab52  
	"                                            \n\t" //   ab63    ab62    ab61    ab60
	"                                            \n\t" //   ab73 )  ab72 )  ab71 )  ab70 )
	"                                            \n\t"
	"vmovapd           %%ymm15, %%ymm7           \n\t"
	"vperm2f128 $0x30, %%ymm15, %%ymm11, %%ymm15 \n\t"
	"vperm2f128 $0x12, %%ymm7,  %%ymm11, %%ymm11 \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm13, %%ymm7           \n\t"
	"vperm2f128 $0x30, %%ymm13, %%ymm9,  %%ymm13 \n\t"
	"vperm2f128 $0x12, %%ymm7,  %%ymm9,  %%ymm9  \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm14, %%ymm7           \n\t"
	"vperm2f128 $0x30, %%ymm14, %%ymm10, %%ymm14 \n\t"
	"vperm2f128 $0x12, %%ymm7,  %%ymm10, %%ymm10 \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm12, %%ymm7           \n\t"
	"vperm2f128 $0x30, %%ymm12, %%ymm8,  %%ymm12 \n\t"
	"vperm2f128 $0x12, %%ymm7,  %%ymm8,  %%ymm8  \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm9:   ymm11:  ymm13:  ymm15:
	"                                            \n\t" // ( ab00  ( ab01  ( ab02  ( ab03
	"                                            \n\t" //   ab10    ab11    ab12    ab13  
	"                                            \n\t" //   ab20    ab21    ab22    ab23
	"                                            \n\t" //   ab30 )  ab31 )  ab32 )  ab33 )
	"                                            \n\t"
	"                                            \n\t" // ymm8:   ymm10:  ymm12:  ymm14:
	"                                            \n\t" // ( ab40  ( ab41  ( ab42  ( ab43
	"                                            \n\t" //   ab50    ab51    ab52    ab53  
	"                                            \n\t" //   ab60    ab61    ab62    ab63
	"                                            \n\t" //   ab70 )  ab71 )  ab72 )  ab73 )
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
//	"movq      %6, %%rdi                         \n\t" // load pc
//	"testq  %%rdi, %%rdi                         \n\t" // check pc via logical AND. 
//	"je     .SQDISTANCE                          \n\t" // if pc == 0, jump to code
	"                                            \n\t"
	"                                            \n\t"
  "movq                   %4, %%rcx            \n\t" // load address of c
	"movq                  %7, %%rdi            \n\t" // load  ldc
	"leaq           (,%%rdi,8), %%rdi            \n\t" // rsi = ldc * sizeof(double)
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd    0 * 32(%%rcx),  %%ymm0           \n\t" // ymm0 = C_c( 0:3, 0 )
	"vaddpd            %%ymm9,  %%ymm0,  %%ymm9  \n\t" // ymm0 += ymm9
	"vmovapd    1 * 32(%%rcx),  %%ymm1           \n\t" // ymm0 = C_c( 4:7, 0 )
	"vaddpd            %%ymm8,  %%ymm1,  %%ymm8  \n\t" // ymm0 += ymm8
	"                                            \n\t"
	"addq              %%rdi,   %%rcx            \n\t"
	"                                            \n\t"
	"vmovapd    0 * 32(%%rcx),  %%ymm2           \n\t" // ymm0 = C_c( 0:3, 1 )
	"vaddpd            %%ymm11, %%ymm2,  %%ymm11 \n\t" // ymm0 += ymm11
	"vmovapd    1 * 32(%%rcx),  %%ymm3           \n\t" // ymm0 = C_c( 4:7, 1 )
	"vaddpd            %%ymm10, %%ymm3,  %%ymm10 \n\t" // ymm0 += ymm10
	"                                            \n\t"
	"addq              %%rdi,   %%rcx            \n\t"
	"                                            \n\t"
	"vmovapd    0 * 32(%%rcx),  %%ymm4           \n\t" // ymm0 = C_c( 0:3, 2 )
	"vaddpd            %%ymm13, %%ymm4,  %%ymm13 \n\t" // ymm0 += ymm13
	"vmovapd    1 * 32(%%rcx),  %%ymm5           \n\t" // ymm0 = C_c( 4:7, 2 )
	"vaddpd            %%ymm12, %%ymm5,  %%ymm12 \n\t" // ymm0 += ymm12
	"                                            \n\t"
	"addq              %%rdi,   %%rcx            \n\t"
	"                                            \n\t"
	"vmovapd    0 * 32(%%rcx),  %%ymm6           \n\t" // ymm0 = C_c( 0:3, 3 )
	"vaddpd            %%ymm15, %%ymm6,  %%ymm15 \n\t" // ymm0 += ymm15
	"vmovapd    1 * 32(%%rcx),  %%ymm7           \n\t" // ymm0 = C_c( 4:7, 3 )
	"vaddpd            %%ymm14, %%ymm7,  %%ymm14 \n\t" // ymm0 += ymm14
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SQDISTANCE:                                \n\t"
	"                                            \n\t"
//	"movq     %8, %%rdi                         \n\t" // load lastiter flag
//	"testq  %%rdi, %%rdi                         \n\t" // check the flag via logical AND. 
//	"je     .STOREBACK                           \n\t" // if flag == 0, jump to code
//	"                                            \n\t"
//
//	"movq                %7, %%rax               \n\t" // load address of aa.
//	"movq                %8, %%rbx               \n\t" // load address of bb.
//	"movq                %9, %%rdx               \n\t" // load address of neg2.
//	"                                            \n\t"
//	"vbroadcastsd   0(%%rdx),  %%ymm0            \n\t" // ymm0 = -2.0
//	"vmulpd           %%ymm0,  %%ymm9,  %%ymm9   \n\t"
//	"vmulpd           %%ymm0,  %%ymm11, %%ymm11  \n\t"
//	"vmulpd           %%ymm0,  %%ymm13, %%ymm13  \n\t"
//	"vmulpd           %%ymm0,  %%ymm15, %%ymm15  \n\t"
//	"vmovapd    0 * 32(%%rax),  %%ymm1           \n\t" // ymm1 = aa03
//	"vmulpd           %%ymm0,  %%ymm8,  %%ymm8   \n\t"
//	"vmulpd           %%ymm0,  %%ymm10, %%ymm10  \n\t"
//	"vmulpd           %%ymm0,  %%ymm12, %%ymm12  \n\t"
//	"vmulpd           %%ymm0,  %%ymm14, %%ymm14  \n\t"
//	"                                            \n\t"
//	"vmovapd    1 * 32(%%rax),  %%ymm2           \n\t" // ymm2 = aa47
//	"vaddpd           %%ymm1,  %%ymm9,  %%ymm9   \n\t"
//	"vaddpd           %%ymm1,  %%ymm11, %%ymm11  \n\t"
//	"vaddpd           %%ymm1,  %%ymm13, %%ymm13  \n\t"
//	"vaddpd           %%ymm1,  %%ymm15, %%ymm15  \n\t"
//	"vbroadcastsd   0(%%rbx),  %%ymm0            \n\t" // ymm0 = bb0
//	"vaddpd           %%ymm2,  %%ymm8,  %%ymm8   \n\t"
//	"vaddpd           %%ymm2,  %%ymm10, %%ymm10  \n\t"
//	"vaddpd           %%ymm2,  %%ymm12, %%ymm12  \n\t"
//	"vaddpd           %%ymm2,  %%ymm14, %%ymm14  \n\t"
//	"                                            \n\t"
//	"vbroadcastsd   8(%%rbx),  %%ymm1            \n\t" // ymm1 = bb1
//	"vaddpd           %%ymm0,  %%ymm9,  %%ymm9   \n\t"
//	"vaddpd           %%ymm0,  %%ymm8,  %%ymm8   \n\t"
//	"                                            \n\t"
//	"vbroadcastsd  16(%%rbx),  %%ymm0            \n\t" // ymm0 = bb2
//	"vaddpd           %%ymm1,  %%ymm11, %%ymm11  \n\t"
//	"vaddpd           %%ymm1,  %%ymm10, %%ymm10  \n\t"
//	"                                            \n\t"
//	"vbroadcastsd  24(%%rbx),  %%ymm1            \n\t" // ymm1 = bb3
//	"vaddpd           %%ymm0,  %%ymm13, %%ymm13  \n\t"
//	"vaddpd           %%ymm0,  %%ymm12, %%ymm12  \n\t"
//	"vaddpd           %%ymm1,  %%ymm15, %%ymm15  \n\t"
//	"vaddpd           %%ymm1,  %%ymm14, %%ymm14  \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".STOREBACK:                                 \n\t"
	"                                            \n\t"
	"movq                   %4, %%rcx            \n\t" // load address of c
	"movq                  %7, %%rdi            \n\t" // load address of ldc
	"leaq           (,%%rdi,8), %%rdi            \n\t" // rsi = ldc * sizeof(double)
	"                                            \n\t"
	"vmovapd           %%ymm9,   0(%%rcx)         \n\t" // C_c( 0, 0:3 ) = ymm9
	"vmovapd           %%ymm8,  32(%%rcx)         \n\t" // C_c( 1, 0:3 ) = ymm8
	"addq              %%rdi,   %%rcx            \n\t"
	"vmovapd           %%ymm11,  0(%%rcx)         \n\t" // C_c( 2, 0:3 ) = ymm11
	"vmovapd           %%ymm10, 32(%%rcx)         \n\t" // C_c( 3, 0:3 ) = ymm10
	"addq              %%rdi,   %%rcx            \n\t"
	"vmovapd           %%ymm13,  0(%%rcx)         \n\t" // C_c( 4, 0:3 ) = ymm13
	"vmovapd           %%ymm12, 32(%%rcx)         \n\t" // C_c( 5, 0:3 ) = ymm12
	"addq              %%rdi,   %%rcx            \n\t"
	"vmovapd           %%ymm15,  0(%%rcx)         \n\t" // C_c( 6, 0:3 ) = ymm15
	"vmovapd           %%ymm14, 32(%%rcx)         \n\t" // C_c( 7, 0:3 ) = ymm14
	"                                            \n\t"
	".DDONE:                                     \n\t"
	"                                            \n\t"
	: // output operands (none)
	: // input operands
	  "m" (k_iter),      // 0
	  "m" (k_left),      // 1
	  "m" (a),           // 2
	  "m" (b),           // 3
	  "m" (c),           // 4
	  "m" (aux->b_next), // 5
    "m" (pc),          // 6
    "m" (ldc)         // 7
//    "m" (last)         // 8
	: // register clobber list
	  "rax", "rbx", "rcx", "rsi", "rdi",
    "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	);


  //printf( "ldc = %d\n", ldc );
  //printf( "%lf, %lf, %lf, %lf\n", c[0], c[ ldc + 0], c[ ldc * 2 + 0], c[ ldc * 3 + 0] );
  //printf( "%lf, %lf, %lf, %lf\n", c[1], c[ ldc + 1], c[ ldc * 2 + 1], c[ ldc * 3 + 1] );
  //printf( "%lf, %lf, %lf, %lf\n", c[2], c[ ldc + 2], c[ ldc * 2 + 2], c[ ldc * 3 + 2] );
  //printf( "%lf, %lf, %lf, %lf\n", c[3], c[ ldc + 3], c[ ldc * 2 + 3], c[ ldc * 3 + 3] );
  //printf( "%lf, %lf, %lf, %lf\n", c[4], c[ ldc + 4], c[ ldc * 2 + 4], c[ ldc * 3 + 4] );
  //printf( "%lf, %lf, %lf, %lf\n", c[5], c[ ldc + 5], c[ ldc * 2 + 5], c[ ldc * 3 + 5] );
  //printf( "%lf, %lf, %lf, %lf\n", c[6], c[ ldc + 6], c[ ldc * 2 + 6], c[ ldc * 3 + 6] );
  //printf( "%lf, %lf, %lf, %lf\n", c[7], c[ ldc + 7], c[ ldc * 2 + 7], c[ ldc * 3 + 7] );
}
