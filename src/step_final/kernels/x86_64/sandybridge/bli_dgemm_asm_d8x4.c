/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

/* NOTE: The micro-kernels in this file were partially inspired by portions
   of code found in OpenBLAS 0.2.8 (http://www.openblas.net/). -FGVZ */

#include <stdio.h>
#include <immintrin.h> // AVX

#include <blis_dgemm.h>
#include <avx_types.h>

#define inc_t unsigned long long 
#define dim_t int

void bli_dgemm_asm_8x4(
                        dim_t              k,
                        double*  a,
                        double*  b,
                        double*  c,
                        //inc_t rs_c,
                        //inc_t cs_c,
                        inc_t ldc,
                        aux_t*         data
                      )
{

    const inc_t cs_c = ldc;
    const inc_t rs_c = 1;
    double alpha_val = 1.0, beta_val = 1.0;
    double *alpha, *beta;

    alpha = &alpha_val;
    beta  = &beta_val;

    printf( "before b_next\n" );
	////void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );
    void *b_next = data->b_next;
    printf( "after b_next\n" );

	dim_t   k_iter = k / 4;
	dim_t   k_left = k % 4;

	__asm__ volatile
	(
	"                                            \n\t"
	"                                            \n\t"
	"movq                %2, %%rax               \n\t" // load address of a.
	"movq                %3, %%rbx               \n\t" // load address of b.
	"movq                %9, %%r15               \n\t" // load address of b_next.
	//"movq               %10, %%r14               \n\t" // load address of a_next.
	"addq          $-4 * 64, %%r15               \n\t"
	"                                            \n\t"
	"vmovapd   0 * 32(%%rax), %%ymm0             \n\t" // initialize loop by pre-loading
	"vmovapd   0 * 32(%%rbx), %%ymm2             \n\t" // elements of a and b.
	"vpermilpd  $0x5, %%ymm2, %%ymm3             \n\t"
	"                                            \n\t"
	"movq                %6, %%rcx               \n\t" // load address of c
	"movq                %8, %%rdi               \n\t" // load cs_c
	"leaq        (,%%rdi,8), %%rdi               \n\t" // cs_c *= sizeof(double)
	"leaq   (%%rcx,%%rdi,2), %%r10               \n\t" // load address of c + 2*cs_c;
	"                                            \n\t"
	"prefetcht0   3 * 8(%%rcx)                   \n\t" // prefetch c + 0*cs_c
	"prefetcht0   3 * 8(%%rcx,%%rdi)             \n\t" // prefetch c + 1*cs_c
	"prefetcht0   3 * 8(%%r10)                   \n\t" // prefetch c + 2*cs_c
	"prefetcht0   3 * 8(%%r10,%%rdi)             \n\t" // prefetch c + 3*cs_c
	"                                            \n\t"
	"vxorpd    %%ymm8,  %%ymm8,  %%ymm8          \n\t"
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
	"movq      %0, %%rsi                         \n\t" // i = k_iter;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .DCONSIDKLEFT                        \n\t" // if i == 0, jump to code that
	"                                            \n\t" // contains the k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".DLOOPKITER:                                \n\t" // MAIN LOOP
	"                                            \n\t"
	"addq         $4 * 4 * 8,  %%r15             \n\t" // b_next += 4*4 (unroll x nr)
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"vmovapd   1 * 32(%%rax),  %%ymm1            \n\t"
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \n\t"
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	"prefetcht0  16 * 32(%%rax)                  \n\t"
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovapd   1 * 32(%%rbx),  %%ymm2            \n\t"
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \n\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd   2 * 32(%%rax),  %%ymm0            \n\t"
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
	"vmovapd   3 * 32(%%rax),  %%ymm1            \n\t"
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \n\t"
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	"prefetcht0  18 * 32(%%rax)                  \n\t"
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovapd   2 * 32(%%rbx),  %%ymm2            \n\t"
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \n\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd   4 * 32(%%rax),  %%ymm0            \n\t"
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
	"vmovapd   5 * 32(%%rax),  %%ymm1            \n\t"
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \n\t"
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	"prefetcht0  20 * 32(%%rax)                  \n\t"
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovapd   3 * 32(%%rbx),  %%ymm2            \n\t"
	"addq         $4 * 4 * 8,  %%rbx             \n\t" // b += 4*4 (unroll x nr)
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \n\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd   6 * 32(%%rax),  %%ymm0            \n\t"
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
	"vmovapd   7 * 32(%%rax),  %%ymm1            \n\t"
	"addq         $4 * 8 * 8,  %%rax             \n\t" // a += 4*8 (unroll x mr)
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \n\t"
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	//"prefetcht0  22 * 32(%%rax)                  \n\t"
	"prefetcht0  14 * 32(%%rax)                  \n\t"
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovapd   0 * 32(%%rbx),  %%ymm2            \n\t"
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \n\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd   0 * 32(%%rax),  %%ymm0            \n\t"
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
	//"addq   $4 * 8 * 8, %%rax                    \n\t" // a      += 4*8 (unroll x mr)
	//"addq   $4 * 4 * 8, %%rbx                    \n\t" // b      += 4*4 (unroll x nr)
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
	"vmovapd   1 * 32(%%rax),  %%ymm1            \n\t"
	"addq         $8 * 1 * 8,  %%rax             \n\t" // a += 8 (1 x mr)
	"vmulpd           %%ymm0,  %%ymm2, %%ymm6    \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2, %%ymm4    \n\t"
	"vmulpd           %%ymm0,  %%ymm3, %%ymm7    \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3, %%ymm5    \n\t"
	"vaddpd           %%ymm15, %%ymm6, %%ymm15   \n\t"
	"vaddpd           %%ymm13, %%ymm7, %%ymm13   \n\t"
	"                                            \n\t"
	"prefetcht0  14 * 32(%%rax)                  \n\t"
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
	"movq         %4, %%rax                      \n\t" // load address of alpha
	"movq         %5, %%rbx                      \n\t" // load address of beta 
	"vbroadcastsd    (%%rax), %%ymm0             \n\t" // load alpha and duplicate
	"vbroadcastsd    (%%rbx), %%ymm2             \n\t" // load beta and duplicate
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm8,  %%ymm8   \n\t" // scale by alpha
	"vmulpd           %%ymm0,  %%ymm9,  %%ymm9   \n\t"
	"vmulpd           %%ymm0,  %%ymm10, %%ymm10  \n\t"
	"vmulpd           %%ymm0,  %%ymm11, %%ymm11  \n\t"
	"vmulpd           %%ymm0,  %%ymm12, %%ymm12  \n\t"
	"vmulpd           %%ymm0,  %%ymm13, %%ymm13  \n\t"
	"vmulpd           %%ymm0,  %%ymm14, %%ymm14  \n\t"
	"vmulpd           %%ymm0,  %%ymm15, %%ymm15  \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq                %7, %%rsi               \n\t" // load rs_c
	"leaq        (,%%rsi,8), %%rsi               \n\t" // rsi = rs_c * sizeof(double)
	"                                            \n\t"
	"leaq   (%%rcx,%%rsi,4), %%rdx               \n\t" // load address of c + 4*rs_c;
	"                                            \n\t"
	"leaq        (,%%rsi,2), %%r12               \n\t" // r12 = 2*rs_c;
	"leaq   (%%r12,%%rsi,1), %%r13               \n\t" // r13 = 3*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // determine if
	"                                            \n\t" //    c    % 32 == 0, AND
	"                                            \n\t" //  8*cs_c % 32 == 0, AND
	"                                            \n\t" //    rs_c      == 1
	"                                            \n\t" // ie: aligned, ldim aligned, and
	"                                            \n\t" // column-stored
	"                                            \n\t"
	"cmpq       $8, %%rsi                        \n\t" // set ZF if (8*rs_c) == 8.
	"sete           %%bl                         \n\t" // bl = ( ZF == 1 ? 1 : 0 );
	"testq     $31, %%rcx                        \n\t" // set ZF if c & 32 is zero.
	"setz           %%bh                         \n\t" // bh = ( ZF == 0 ? 1 : 0 );
	"testq     $31, %%rdi                        \n\t" // set ZF if (8*cs_c) & 32 is zero.
	"setz           %%al                         \n\t" // al = ( ZF == 0 ? 1 : 0 );
	"                                            \n\t" // and(bl,bh) followed by
	"                                            \n\t" // and(bh,al) will reveal result
	"                                            \n\t"
	"                                            \n\t" // now avoid loading C if beta == 0
	"                                            \n\t"
	"vxorpd    %%ymm0,  %%ymm0,  %%ymm0          \n\t" // set ymm0 to zero.
	"vucomisd  %%xmm0,  %%xmm2                   \n\t" // set ZF if beta == 0.
	"je      .DBETAZERO                          \n\t" // if ZF = 1, jump to beta == 0 case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // check if aligned/column-stored
	"andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
	"andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
	"jne     .DCOLSTORED                         \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DGENSTORED:                                \n\t"
	"                                            \n\t" // update c00:c33
	"                                            \n\t"
	"vextractf128 $1, %%ymm9,  %%xmm1            \n\t"
	"vmovlpd    (%%rcx),       %%xmm0,  %%xmm0   \n\t" // load c00 and c10,
	"vmovhpd    (%%rcx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm9,  %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rcx)           \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rcx,%%rsi)     \n\t"
	"vmovlpd    (%%rcx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c20 and c30,
	"vmovhpd    (%%rcx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm1,  %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rcx,%%r12)     \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rcx,%%r13)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm11, %%xmm1            \n\t"
	"vmovlpd    (%%rcx),       %%xmm0,  %%xmm0   \n\t" // load c01 and c11,
	"vmovhpd    (%%rcx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm11, %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rcx)           \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rcx,%%rsi)     \n\t"
	"vmovlpd    (%%rcx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c21 and c31,
	"vmovhpd    (%%rcx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm1,  %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rcx,%%r12)     \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rcx,%%r13)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm13, %%xmm1            \n\t"
	"vmovlpd    (%%rcx),       %%xmm0,  %%xmm0   \n\t" // load c02 and c12,
	"vmovhpd    (%%rcx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm13, %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rcx)           \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rcx,%%rsi)     \n\t"
	"vmovlpd    (%%rcx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c22 and c32,
	"vmovhpd    (%%rcx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm1,  %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rcx,%%r12)     \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rcx,%%r13)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm15, %%xmm1            \n\t"
	"vmovlpd    (%%rcx),       %%xmm0,  %%xmm0   \n\t" // load c03 and c13,
	"vmovhpd    (%%rcx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm15, %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rcx)           \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rcx,%%rsi)     \n\t"
	"vmovlpd    (%%rcx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c23 and c33,
	"vmovhpd    (%%rcx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm1,  %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rcx,%%r12)     \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rcx,%%r13)     \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c40:c73
	"                                            \n\t"
	"vextractf128 $1, %%ymm8,  %%xmm1            \n\t"
	"vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load c40 and c50,
	"vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm8,  %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rdx)           \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rdx,%%rsi)     \n\t"
	"vmovlpd    (%%rdx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c60 and c70,
	"vmovhpd    (%%rdx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm1,  %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rdx,%%r12)     \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rdx,%%r13)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm10, %%xmm1            \n\t"
	"vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load c41 and c51,
	"vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm10, %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rdx)           \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rdx,%%rsi)     \n\t"
	"vmovlpd    (%%rdx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c61 and c71,
	"vmovhpd    (%%rdx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm1,  %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rdx,%%r12)     \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rdx,%%r13)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm12, %%xmm1            \n\t"
	"vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load c42 and c52,
	"vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm12, %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rdx)           \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rdx,%%rsi)     \n\t"
	"vmovlpd    (%%rdx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c62 and c72,
	"vmovhpd    (%%rdx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm1,  %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rdx,%%r12)     \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rdx,%%r13)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm14, %%xmm1            \n\t"
	"vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load c43 and c53,
	"vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm14, %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rdx)           \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rdx,%%rsi)     \n\t"
	"vmovlpd    (%%rdx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c63 and c73,
	"vmovhpd    (%%rdx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm1,  %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rdx,%%r12)     \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rdx,%%r13)     \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .DDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DCOLSTORED:                                \n\t"
	"                                            \n\t" // update c00:c33
	"                                            \n\t"
	"vmovapd    (%%rcx),       %%ymm0            \n\t" // load c00:c30,
	"vmulpd           %%ymm2,  %%ymm0,  %%ymm0   \n\t" // scale by beta,
	"vaddpd           %%ymm9,  %%ymm0,  %%ymm0   \n\t" // add the gemm result,
	"vmovapd          %%ymm0,  (%%rcx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd    (%%rcx),       %%ymm0            \n\t" // load c01:c31,
	"vmulpd           %%ymm2,  %%ymm0,  %%ymm0   \n\t" // scale by beta,
	"vaddpd           %%ymm11, %%ymm0,  %%ymm0   \n\t" // add the gemm result,
	"vmovapd          %%ymm0,  (%%rcx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd    (%%rcx),       %%ymm0            \n\t" // load c02:c32,
	"vmulpd           %%ymm2,  %%ymm0,  %%ymm0   \n\t" // scale by beta,
	"vaddpd           %%ymm13, %%ymm0,  %%ymm0   \n\t" // add the gemm result,
	"vmovapd          %%ymm0,  (%%rcx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd    (%%rcx),       %%ymm0            \n\t" // load c03:c33,
	"vmulpd           %%ymm2,  %%ymm0,  %%ymm0   \n\t" // scale by beta,
	"vaddpd           %%ymm15, %%ymm0,  %%ymm0   \n\t" // add the gemm result,
	"vmovapd          %%ymm0,  (%%rcx)           \n\t" // and store back to memory.
	"                                            \n\t"
	"                                            \n\t" // update c40:c73
	"                                            \n\t"
	"vmovapd    (%%rdx),       %%ymm0            \n\t" // load c40:c70,
	"vmulpd           %%ymm2,  %%ymm0,  %%ymm0   \n\t" // scale by beta,
	"vaddpd           %%ymm8,  %%ymm0,  %%ymm0   \n\t" // add the gemm result,
	"vmovapd          %%ymm0,  (%%rdx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd    (%%rdx),       %%ymm0            \n\t" // load c41:c71,
	"vmulpd           %%ymm2,  %%ymm0,  %%ymm0   \n\t" // scale by beta,
	"vaddpd           %%ymm10, %%ymm0,  %%ymm0   \n\t" // add the gemm result,
	"vmovapd          %%ymm0,  (%%rdx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd    (%%rdx),       %%ymm0            \n\t" // load c42:c72,
	"vmulpd           %%ymm2,  %%ymm0,  %%ymm0   \n\t" // scale by beta,
	"vaddpd           %%ymm12, %%ymm0,  %%ymm0   \n\t" // add the gemm result,
	"vmovapd          %%ymm0,  (%%rdx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd    (%%rdx),       %%ymm0            \n\t" // load c43:c73,
	"vmulpd           %%ymm2,  %%ymm0,  %%ymm0   \n\t" // scale by beta,
	"vaddpd           %%ymm14, %%ymm0,  %%ymm0   \n\t" // add the gemm result,
	"vmovapd          %%ymm0,  (%%rdx)           \n\t" // and store back to memory.
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .DDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DBETAZERO:                                 \n\t"
	"                                            \n\t" // check if aligned/column-stored
	"andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
	"andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
	"jne     .DCOLSTORBZ                         \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DGENSTORBZ:                                \n\t"
	"                                            \n\t" // update c00:c33
	"                                            \n\t"
	"vextractf128 $1, %%ymm9,  %%xmm1            \n\t"
	"vmovlpd          %%xmm9,  (%%rcx)           \n\t" // store to c00:c30
	"vmovhpd          %%xmm9,  (%%rcx,%%rsi)     \n\t"
	"vmovlpd          %%xmm1,  (%%rcx,%%r12)     \n\t"
	"vmovhpd          %%xmm1,  (%%rcx,%%r13)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm11, %%xmm1            \n\t"
	"vmovlpd          %%xmm11, (%%rcx)           \n\t" // store to c01:c31
	"vmovhpd          %%xmm11, (%%rcx,%%rsi)     \n\t"
	"vmovlpd          %%xmm1,  (%%rcx,%%r12)     \n\t"
	"vmovhpd          %%xmm1,  (%%rcx,%%r13)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm13, %%xmm1            \n\t"
	"vmovlpd          %%xmm13, (%%rcx)           \n\t" // store to c02:c32
	"vmovhpd          %%xmm13, (%%rcx,%%rsi)     \n\t"
	"vmovlpd          %%xmm1,  (%%rcx,%%r12)     \n\t"
	"vmovhpd          %%xmm1,  (%%rcx,%%r13)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm15, %%xmm1            \n\t"
	"vmovlpd          %%xmm15, (%%rcx)           \n\t" // store to c03:c33
	"vmovhpd          %%xmm15, (%%rcx,%%rsi)     \n\t"
	"vmovlpd          %%xmm1,  (%%rcx,%%r12)     \n\t"
	"vmovhpd          %%xmm1,  (%%rcx,%%r13)     \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c40:c73
	"                                            \n\t"
	"vextractf128 $1, %%ymm8,  %%xmm1            \n\t"
	"vmovlpd          %%xmm8,  (%%rdx)           \n\t" // store to c40:c70
	"vmovhpd          %%xmm8,  (%%rdx,%%rsi)     \n\t"
	"vmovlpd          %%xmm1,  (%%rdx,%%r12)     \n\t"
	"vmovhpd          %%xmm1,  (%%rdx,%%r13)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm10, %%xmm1            \n\t"
	"vmovlpd          %%xmm10, (%%rdx)           \n\t" // store to c41:c71
	"vmovhpd          %%xmm10, (%%rdx,%%rsi)     \n\t"
	"vmovlpd          %%xmm1,  (%%rdx,%%r12)     \n\t"
	"vmovhpd          %%xmm1,  (%%rdx,%%r13)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm12, %%xmm1            \n\t"
	"vmovlpd          %%xmm12, (%%rdx)           \n\t" // store to c42:c72
	"vmovhpd          %%xmm12, (%%rdx,%%rsi)     \n\t"
	"vmovlpd          %%xmm1,  (%%rdx,%%r12)     \n\t"
	"vmovhpd          %%xmm1,  (%%rdx,%%r13)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm14, %%xmm1            \n\t"
	"vmovlpd          %%xmm14, (%%rdx)           \n\t" // store to c43:c73
	"vmovhpd          %%xmm14, (%%rdx,%%rsi)     \n\t"
	"vmovlpd          %%xmm1,  (%%rdx,%%r12)     \n\t"
	"vmovhpd          %%xmm1,  (%%rdx,%%r13)     \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .DDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DCOLSTORBZ:                                \n\t"
	"                                            \n\t" // update c00:c33
	"                                            \n\t"
	"vmovapd          %%ymm9,  (%%rcx)           \n\t" // store c00:c30
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd          %%ymm11, (%%rcx)           \n\t" // store c01:c31
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd          %%ymm13, (%%rcx)           \n\t" // store c02:c32
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd          %%ymm15, (%%rcx)           \n\t" // store c03:c33
	"                                            \n\t"
	"                                            \n\t" // update c40:c73
	"                                            \n\t"
	"vmovapd          %%ymm8,  (%%rdx)           \n\t" // store c40:c70
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd          %%ymm10, (%%rdx)           \n\t" // store c41:c71
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd          %%ymm12, (%%rdx)           \n\t" // store c42:c72
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd          %%ymm14, (%%rdx)           \n\t" // store c43:c73
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DDONE:                                     \n\t"
	"                                            \n\t"

	: // output operands (none)
	: // input operands
	  "m" (k_iter), // 0
	  "m" (k_left), // 1
	  "m" (a),      // 2
	  "m" (b),      // 3
	  "m" (alpha),  // 4
	  "m" (beta),   // 5
	  "m" (c),      // 6
	  "m" (rs_c),   // 7
	  "m" (cs_c),   // 8
	  "m" (b_next)/*, // 9
	  "m" (a_next)*/  // 10
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



