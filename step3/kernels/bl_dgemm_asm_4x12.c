#include "bl_dgemm_kernel.h"

#define inc_t unsigned long long

#define DGEMM_INPUT_GS_BETA_NZ                            \
  "vmovlpd    (%%rcx        ),  %%xmm0,  %%xmm0  \n\t"    \
  "vmovhpd    (%%rcx,%%rsi,1),  %%xmm0,  %%xmm0  \n\t"    \
  "vmovlpd    (%%rcx,%%rsi,2),  %%xmm1,  %%xmm1  \n\t"    \
  "vmovhpd    (%%rcx,%%r13  ),  %%xmm1,  %%xmm1  \n\t"    \
  "vperm2f128 $0x20,   %%ymm1,  %%ymm0,  %%ymm0  \n\t" /* \
  "vmovlps    (%%rcx,%%rsi,4),  %%xmm2,  %%xmm2  \n\t"    \
  "vmovhps    (%%rcx,%%r15  ),  %%xmm2,  %%xmm2  \n\t"    \
  "vmovlps    (%%rcx,%%r13,2),  %%xmm1,  %%xmm1  \n\t"    \
  "vmovhps    (%%rcx,%%r10  ),  %%xmm1,  %%xmm1  \n\t"    \
  "vperm2f128 $0x20,   %%ymm1,  %%ymm2,  %%ymm2  \n\t"*/

#define DGEMM_OUTPUT_GS_BETA_NZ                         \
  "vextractf128  $1, %%ymm0,  %%xmm1           \n\t"    \
  "vmovlpd           %%xmm0,  (%%rcx        )  \n\t"    \
  "vmovhpd           %%xmm0,  (%%rcx,%%rsi  )  \n\t"    \
  "vmovlpd           %%xmm1,  (%%rcx,%%rsi,2)  \n\t"    \
  "vmovhpd           %%xmm1,  (%%rcx,%%r13  )  \n\t" /* \
  "vextractf128  $1, %%ymm2,  %%xmm1           \n\t"    \
  "vmovlpd           %%xmm2,  (%%rcx,%%rsi,4)  \n\t"    \
  "vmovhpd           %%xmm2,  (%%rcx,%%r15  )  \n\t"    \
  "vmovlpd           %%xmm1,  (%%rcx,%%r13,2)  \n\t"    \
  "vmovhpd           %%xmm1,  (%%rcx,%%r10  )  \n\t"*/

void bl_dgemm_asm_4x12(
    int k,
    double* a,
    double* b,
    double* c,
    inc_t ldc,
    aux_t* data) {
  // void*   a_next = bli_auxinfo_next_a( data );
  // void*   b_next = bli_auxinfo_next_b( data );

  // dim_t   k_iter = k / 4;
  // dim_t   k_left = k % 4;

  const inc_t cs_c = ldc;
  const inc_t rs_c = 1;
  double alpha_val = 1.0, beta_val = 1.0;
  double *alpha, *beta;

  alpha = &alpha_val;
  beta = &beta_val;

  dim_t k_iter = (unsigned long long)k / 4;
  dim_t k_left = (unsigned long long)k % 4;

  __asm__ volatile
    (
    "                                            \n\t"
    "vzeroall                                    \n\t" // zero all xmm/ymm registers.
    "                                            \n\t"
    "                                            \n\t"
    "movq                %2, %%rax               \n\t" // load address of a.
    "movq                %3, %%rbx               \n\t" // load address of b.
    //"movq                %9, %%r15               \n\t" // load address of b_next.
    "                                            \n\t"
    "addq           $32 * 4, %%rbx               \n\t"
    "                                            \n\t" // initialize loop by pre-loading
    "vmovapd           -4 * 32(%%rbx), %%ymm1    \n\t"
    "vmovapd           -3 * 32(%%rbx), %%ymm2    \n\t"
    "vmovapd           -2 * 32(%%rbx), %%ymm3    \n\t"
    "                                            \n\t"
    "movq                %6, %%rcx               \n\t" // load address of c
    "movq                %7, %%rdi               \n\t" // load rs_c
    "leaq        (,%%rdi,8), %%rdi               \n\t" // rs_c *= sizeof(double)
    "                                            \n\t"
    "leaq   (%%rdi,%%rdi,2), %%r13               \n\t" // r13 = 3*rs_c;
    "prefetcht0   7 * 8(%%rcx)                   \n\t" // prefetch c + 0*rs_c
    "prefetcht0   7 * 8(%%rcx,%%rdi)             \n\t" // prefetch c + 1*rs_c
    "prefetcht0   7 * 8(%%rcx,%%rdi,2)           \n\t" // prefetch c + 2*rs_c
    "prefetcht0   7 * 8(%%rcx,%%r13)             \n\t" // prefetch c + 3*rs_c
    "                                            \n\t"
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
    "                                            \n\t"
    "                                            \n\t" // iteration 0
    "prefetcht0   24 * 8(%%rax)                  \n\t"
    "                                            \n\t"
    "vbroadcastf128     0 *  8(%%rax), %%ymm0    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm0, %%ymm4    \n\t"
    "vfmadd231pd       %%ymm2, %%ymm0, %%ymm5    \n\t"
    "vfmadd231pd       %%ymm3, %%ymm0, %%ymm6    \n\t"
    "vpermilpd           $0x5, %%ymm0, %%ymm0    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm0, %%ymm7    \n\t"
    "vfmadd231pd       %%ymm2, %%ymm0, %%ymm8    \n\t"
    "vfmadd231pd       %%ymm3, %%ymm0, %%ymm9    \n\t"
    "vbroadcastf128     2 *  8(%%rax), %%ymm0    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm0, %%ymm10   \n\t"
    "vfmadd231pd       %%ymm2, %%ymm0, %%ymm11   \n\t"
    "vfmadd231pd       %%ymm3, %%ymm0, %%ymm12   \n\t"
    "vpermilpd           $0x5, %%ymm0, %%ymm0    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm0, %%ymm13   \n\t"
    "vmovapd           -1 * 32(%%rbx), %%ymm1    \n\t"
    "vfmadd231pd       %%ymm2, %%ymm0, %%ymm14   \n\t"
    "vmovapd            0 * 32(%%rbx), %%ymm2    \n\t"
    "vfmadd231pd       %%ymm3, %%ymm0, %%ymm15   \n\t"
    "vmovapd            1 * 32(%%rbx), %%ymm3    \n\t"
    "                                            \n\t"
    "                                            \n\t" // iteration 1
    "vbroadcastf128     4 *  8(%%rax), %%ymm0    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm0, %%ymm4    \n\t"
    "vfmadd231pd       %%ymm2, %%ymm0, %%ymm5    \n\t"
    "vfmadd231pd       %%ymm3, %%ymm0, %%ymm6    \n\t"
    "vpermilpd           $0x5, %%ymm0, %%ymm0    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm0, %%ymm7    \n\t"
    "vfmadd231pd       %%ymm2, %%ymm0, %%ymm8    \n\t"
    "vfmadd231pd       %%ymm3, %%ymm0, %%ymm9    \n\t"
    "vbroadcastf128     6 *  8(%%rax), %%ymm0    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm0, %%ymm10   \n\t"
    "vfmadd231pd       %%ymm2, %%ymm0, %%ymm11   \n\t"
    "vfmadd231pd       %%ymm3, %%ymm0, %%ymm12   \n\t"
    "vpermilpd           $0x5, %%ymm0, %%ymm0    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm0, %%ymm13   \n\t"
    "vmovapd            2 * 32(%%rbx), %%ymm1    \n\t"
    "vfmadd231pd       %%ymm2, %%ymm0, %%ymm14   \n\t"
    "vmovapd            3 * 32(%%rbx), %%ymm2    \n\t"
    "vfmadd231pd       %%ymm3, %%ymm0, %%ymm15   \n\t"
    "vmovapd            4 * 32(%%rbx), %%ymm3    \n\t"
    "                                            \n\t"
    "prefetcht0   32 * 8(%%rax)                  \n\t"
    "                                            \n\t" // iteration 2
    "vbroadcastf128     8 *  8(%%rax), %%ymm0    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm0, %%ymm4    \n\t"
    "vfmadd231pd       %%ymm2, %%ymm0, %%ymm5    \n\t"
    "vfmadd231pd       %%ymm3, %%ymm0, %%ymm6    \n\t"
    "vpermilpd           $0x5, %%ymm0, %%ymm0    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm0, %%ymm7    \n\t"
    "vfmadd231pd       %%ymm2, %%ymm0, %%ymm8    \n\t"
    "vfmadd231pd       %%ymm3, %%ymm0, %%ymm9    \n\t"
    "vbroadcastf128    10 *  8(%%rax), %%ymm0    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm0, %%ymm10   \n\t"
    "vfmadd231pd       %%ymm2, %%ymm0, %%ymm11   \n\t"
    "vfmadd231pd       %%ymm3, %%ymm0, %%ymm12   \n\t"
    "vpermilpd           $0x5, %%ymm0, %%ymm0    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm0, %%ymm13   \n\t"
    "vmovapd            5 * 32(%%rbx), %%ymm1    \n\t"
    "vfmadd231pd       %%ymm2, %%ymm0, %%ymm14   \n\t"
    "vmovapd            6 * 32(%%rbx), %%ymm2    \n\t"
    "vfmadd231pd       %%ymm3, %%ymm0, %%ymm15   \n\t"
    "vmovapd            7 * 32(%%rbx), %%ymm3    \n\t"
    "                                            \n\t"
    "                                            \n\t" // iteration 3
    "vbroadcastf128    12 *  8(%%rax), %%ymm0    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm0, %%ymm4    \n\t"
    "vfmadd231pd       %%ymm2, %%ymm0, %%ymm5    \n\t"
    "vfmadd231pd       %%ymm3, %%ymm0, %%ymm6    \n\t"
    "vpermilpd           $0x5, %%ymm0, %%ymm0    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm0, %%ymm7    \n\t"
    "vfmadd231pd       %%ymm2, %%ymm0, %%ymm8    \n\t"
    "vfmadd231pd       %%ymm3, %%ymm0, %%ymm9    \n\t"
    "vbroadcastf128    14 *  8(%%rax), %%ymm0    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm0, %%ymm10   \n\t"
    "vfmadd231pd       %%ymm2, %%ymm0, %%ymm11   \n\t"
    "vfmadd231pd       %%ymm3, %%ymm0, %%ymm12   \n\t"
    "vpermilpd           $0x5, %%ymm0, %%ymm0    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm0, %%ymm13   \n\t"
    "vmovapd            8 * 32(%%rbx), %%ymm1    \n\t"
    "vfmadd231pd       %%ymm2, %%ymm0, %%ymm14   \n\t"
    "vmovapd            9 * 32(%%rbx), %%ymm2    \n\t"
    "vfmadd231pd       %%ymm3, %%ymm0, %%ymm15   \n\t"
    "vmovapd           10 * 32(%%rbx), %%ymm3    \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "addq          $4 *  4 * 8, %%rax            \n\t" // a += 4*4  (unroll x mr)
    "addq          $4 * 12 * 8, %%rbx            \n\t" // b += 4*12 (unroll x nr)
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
    "prefetcht0   24 * 8(%%rax)                  \n\t"
    "                                            \n\t"
    "vbroadcastf128     0 *  8(%%rax), %%ymm0    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm0, %%ymm4    \n\t"
    "vfmadd231pd       %%ymm2, %%ymm0, %%ymm5    \n\t"
    "vfmadd231pd       %%ymm3, %%ymm0, %%ymm6    \n\t"
    "vpermilpd           $0x5, %%ymm0, %%ymm0    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm0, %%ymm7    \n\t"
    "vfmadd231pd       %%ymm2, %%ymm0, %%ymm8    \n\t"
    "vfmadd231pd       %%ymm3, %%ymm0, %%ymm9    \n\t"
    "vbroadcastf128     2 *  8(%%rax), %%ymm0    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm0, %%ymm10   \n\t"
    "vfmadd231pd       %%ymm2, %%ymm0, %%ymm11   \n\t"
    "vfmadd231pd       %%ymm3, %%ymm0, %%ymm12   \n\t"
    "vpermilpd           $0x5, %%ymm0, %%ymm0    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm0, %%ymm13   \n\t"
    "vmovapd           -1 * 32(%%rbx), %%ymm1    \n\t"
    "vfmadd231pd       %%ymm2, %%ymm0, %%ymm14   \n\t"
    "vmovapd            0 * 32(%%rbx), %%ymm2    \n\t"
    "vfmadd231pd       %%ymm3, %%ymm0, %%ymm15   \n\t"
    "vmovapd            1 * 32(%%rbx), %%ymm3    \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "addq          $1 *  4 * 8, %%rax            \n\t" // a += 1*4  (unroll x mr)
    "addq          $1 * 12 * 8, %%rbx            \n\t" // b += 1*12 (unroll x nr)
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
    "                                            \n\t"
    "                                            \n\t"
    "movq         %4, %%rax                      \n\t" // load address of alpha
    "movq         %5, %%rbx                      \n\t" // load address of beta
    "vbroadcastsd    (%%rax), %%ymm0             \n\t" // load alpha and duplicate
    "vbroadcastsd    (%%rbx), %%ymm3             \n\t" // load beta and duplicate
    "                                            \n\t"
    "vmulpd           %%ymm0,  %%ymm4,  %%ymm4   \n\t" // scale by alpha
    "vmulpd           %%ymm0,  %%ymm5,  %%ymm5   \n\t"
    "vmulpd           %%ymm0,  %%ymm6,  %%ymm6   \n\t"
    "vmulpd           %%ymm0,  %%ymm7,  %%ymm7   \n\t"
    "vmulpd           %%ymm0,  %%ymm8,  %%ymm8   \n\t"
    "vmulpd           %%ymm0,  %%ymm9,  %%ymm9   \n\t"
    "vmulpd           %%ymm0,  %%ymm10, %%ymm10  \n\t"
    "vmulpd           %%ymm0,  %%ymm11, %%ymm11  \n\t"
    "vmulpd           %%ymm0,  %%ymm12, %%ymm12  \n\t"
    "vmulpd           %%ymm0,  %%ymm13, %%ymm13  \n\t"
    "vmulpd           %%ymm0,  %%ymm14, %%ymm14  \n\t"
    "vmulpd           %%ymm0,  %%ymm15, %%ymm15  \n\t"
    "                                            \n\t"
    "                                            \n\t" // ymm4 : ( ab00 ab11 ab02 ab13 )
    "                                            \n\t" // ymm7 : ( ab10 ab01 ab12 ab03 )
    "                                            \n\t" // ymm10: ( ab20 ab31 ab22 ab33 )
    "                                            \n\t" // ymm13: ( ab30 ab21 ab32 ab23 )
    "                                            \n\t"
    "                                            \n\t" // ymm5 : ( ab04 ab15 ab06 ab17 )
    "                                            \n\t" // ymm8 : ( ab14 ab05 ab16 ab07 )
    "                                            \n\t" // ymm11: ( ab24 ab35 ab26 ab37 )
    "                                            \n\t" // ymm14: ( ab34 ab25 ab36 ab27 )
    "                                            \n\t"
    "                                            \n\t" // ymm6 : ( ab08 ab19 ab0A ab1B )
    "                                            \n\t" // ymm9 : ( ab18 ab09 ab1A ab0B )
    "                                            \n\t" // ymm12: ( ab28 ab39 ab2A ab3B )
    "                                            \n\t" // ymm15: ( ab38 ab29 ab3A ab2B )
    "vmovapd          %%ymm4,  %%ymm0            \n\t"
    "vshufpd    $0xa, %%ymm7,  %%ymm4,  %%ymm4   \n\t"
    "vshufpd    $0xa, %%ymm0,  %%ymm7,  %%ymm7   \n\t"
    "                                            \n\t"
    "vmovapd          %%ymm5,  %%ymm0            \n\t"
    "vshufpd    $0xa, %%ymm8,  %%ymm5,  %%ymm5   \n\t"
    "vshufpd    $0xa, %%ymm0,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "vmovapd          %%ymm6,  %%ymm0            \n\t"
    "vshufpd    $0xa, %%ymm9,  %%ymm6,  %%ymm6   \n\t"
    "vshufpd    $0xa, %%ymm0,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmovapd          %%ymm10, %%ymm0            \n\t"
    "vshufpd    $0xa, %%ymm13, %%ymm10, %%ymm10  \n\t"
    "vshufpd    $0xa, %%ymm0,  %%ymm13, %%ymm13  \n\t"
    "                                            \n\t"
    "vmovapd          %%ymm11, %%ymm0            \n\t"
    "vshufpd    $0xa, %%ymm14, %%ymm11, %%ymm11  \n\t"
    "vshufpd    $0xa, %%ymm0,  %%ymm14, %%ymm14  \n\t"
    "                                            \n\t"
    "vmovapd          %%ymm12, %%ymm0            \n\t"
    "vshufpd    $0xa, %%ymm15, %%ymm12, %%ymm12  \n\t"
    "vshufpd    $0xa, %%ymm0,  %%ymm15, %%ymm15  \n\t"
    "                                            \n\t" // ymm4 : ( ab00 ab01 ab02 ab03 )
    "                                            \n\t" // ymm7 : ( ab10 ab11 ab12 ab13 )
    "                                            \n\t" // ymm10: ( ab20 ab21 ab22 ab23 )
    "                                            \n\t" // ymm13: ( ab30 ab31 ab32 ab33 )
    "                                            \n\t"
    "                                            \n\t" // ymm5 : ( ab04 ab05 ab06 ab07 )
    "                                            \n\t" // ymm8 : ( ab14 ab15 ab16 ab17 )
    "                                            \n\t" // ymm11: ( ab24 ab25 ab26 ab27 )
    "                                            \n\t" // ymm14: ( ab34 ab35 ab36 ab37 )
    "                                            \n\t"
    "                                            \n\t" // ymm6 : ( ab08 ab09 ab0A ab0B )
    "                                            \n\t" // ymm9 : ( ab18 ab19 ab1A ab1B )
    "                                            \n\t" // ymm12: ( ab28 ab29 ab2A ab2B )
    "                                            \n\t" // ymm15: ( ab38 ab39 ab3A ab3B )
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
      //"m" (rs_c),   // 7  rdi
      //"m" (cs_c),   // 8  rsi
    "movq                %8, %%rsi               \n\t" // load cs_c
    "leaq        (,%%rsi,8), %%rsi               \n\t" // rsi = cs_c * sizeof(double)
    "                                            \n\t"
    "leaq   (%%rcx,%%rsi,4), %%rdx               \n\t" // rdx = c +  4*cs_c;
    "leaq   (%%rcx,%%rsi,8), %%r12               \n\t" // r12 = c +  8*cs_c;
    "                                            \n\t"
    "leaq   (%%rsi,%%rsi,2), %%r13               \n\t" // r13 = 3*cs_c;
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // determine if
    "                                            \n\t" //    c    % 32 == 0, AND
    "                                            \n\t" //  8*rs_c % 32 == 0, AND
    "                                            \n\t" //    cs_c      == 1
    "                                            \n\t" // ie: aligned, ldim aligned, and
    "                                            \n\t" // row-stored
    "                                            \n\t"
    "cmpq       $8, %%rsi                        \n\t" // set ZF if (8*cs_c) == 8.
    "sete           %%bl                         \n\t" // bl = ( ZF == 1 ? 1 : 0 );
    "testq     $31, %%rcx                        \n\t" // set ZF if c & 32 is zero.
    "setz           %%bh                         \n\t" // bh = ( ZF == 0 ? 1 : 0 );
    "testq     $31, %%rdi                        \n\t" // set ZF if (8*rs_c) & 32 is zero.
    "setz           %%al                         \n\t" // al = ( ZF == 0 ? 1 : 0 );
    "                                            \n\t" // and(bl,bh) followed by
    "                                            \n\t" // and(bh,al) will reveal result
    "                                            \n\t"
    "                                            \n\t" // now avoid loading C if beta == 0
    "                                            \n\t"
    "vxorpd    %%ymm0,  %%ymm0,  %%ymm0          \n\t" // set ymm0 to zero.
    "vucomisd  %%xmm0,  %%xmm3                   \n\t" // set ZF if beta == 0.
    "je      .DBETAZERO                          \n\t" // if ZF = 1, jump to beta == 0 case
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // check if aligned/row-stored
    "andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
    "andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
    "jne     .DROWSTORED                         \n\t" // jump to row storage case
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".DGENSTORED:                                \n\t"
    "                                            \n\t"
    "                                            \n\t"
    DGEMM_INPUT_GS_BETA_NZ
    "vfmadd213pd      %%ymm4,  %%ymm3,  %%ymm0   \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    DGEMM_INPUT_GS_BETA_NZ
    "vfmadd213pd      %%ymm7,  %%ymm3,  %%ymm0   \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    DGEMM_INPUT_GS_BETA_NZ
    "vfmadd213pd      %%ymm10, %%ymm3,  %%ymm0   \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    DGEMM_INPUT_GS_BETA_NZ
    "vfmadd213pd      %%ymm13, %%ymm3,  %%ymm0   \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    //"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "movq      %%rdx, %%rcx                      \n\t" // rcx = c + 4*cs_c
    "                                            \n\t"
    "                                            \n\t"
    DGEMM_INPUT_GS_BETA_NZ
    "vfmadd213pd      %%ymm5,  %%ymm3,  %%ymm0   \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    DGEMM_INPUT_GS_BETA_NZ
    "vfmadd213pd      %%ymm8,  %%ymm3,  %%ymm0   \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    DGEMM_INPUT_GS_BETA_NZ
    "vfmadd213pd      %%ymm11, %%ymm3,  %%ymm0   \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    DGEMM_INPUT_GS_BETA_NZ
    "vfmadd213pd      %%ymm14, %%ymm3,  %%ymm0   \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    //"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "movq      %%r12, %%rcx                      \n\t" // rcx = c + 8*cs_c
    "                                            \n\t"
    "                                            \n\t"
    DGEMM_INPUT_GS_BETA_NZ
    "vfmadd213pd      %%ymm6,  %%ymm3,  %%ymm0   \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    DGEMM_INPUT_GS_BETA_NZ
    "vfmadd213pd      %%ymm9,  %%ymm3,  %%ymm0   \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    DGEMM_INPUT_GS_BETA_NZ
    "vfmadd213pd      %%ymm12, %%ymm3,  %%ymm0   \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    DGEMM_INPUT_GS_BETA_NZ
    "vfmadd213pd      %%ymm15, %%ymm3,  %%ymm0   \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    //"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "jmp    .DDONE                               \n\t" // jump to end.
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".DROWSTORED:                                \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd    (%%rcx),       %%ymm0            \n\t"
    "vfmadd213pd      %%ymm4,  %%ymm3,  %%ymm0   \n\t"
    "vmovapd          %%ymm0,  (%%rcx)           \n\t"
    "addq      %%rdi, %%rcx                      \n\t"
    "vmovapd    (%%rdx),       %%ymm1            \n\t"
    "vfmadd213pd      %%ymm5,  %%ymm3,  %%ymm1   \n\t"
    "vmovapd          %%ymm1,  (%%rdx)           \n\t"
    "addq      %%rdi, %%rdx                      \n\t"
    "vmovapd    (%%r12),       %%ymm2            \n\t"
    "vfmadd213pd      %%ymm6,  %%ymm3,  %%ymm2   \n\t"
    "vmovapd          %%ymm2,  (%%r12)           \n\t"
    "addq      %%rdi, %%r12                      \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd    (%%rcx),       %%ymm0            \n\t"
    "vfmadd213pd      %%ymm7,  %%ymm3,  %%ymm0   \n\t"
    "vmovapd          %%ymm0,  (%%rcx)           \n\t"
    "addq      %%rdi, %%rcx                      \n\t"
    "vmovapd    (%%rdx),       %%ymm1            \n\t"
    "vfmadd213pd      %%ymm8,  %%ymm3,  %%ymm1   \n\t"
    "vmovapd          %%ymm1,  (%%rdx)           \n\t"
    "addq      %%rdi, %%rdx                      \n\t"
    "vmovapd    (%%r12),       %%ymm2            \n\t"
    "vfmadd213pd      %%ymm9,  %%ymm3,  %%ymm2   \n\t"
    "vmovapd          %%ymm2,  (%%r12)           \n\t"
    "addq      %%rdi, %%r12                      \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd    (%%rcx),       %%ymm0            \n\t"
    "vfmadd213pd      %%ymm10, %%ymm3,  %%ymm0   \n\t"
    "vmovapd          %%ymm0,  (%%rcx)           \n\t"
    "addq      %%rdi, %%rcx                      \n\t"
    "vmovapd    (%%rdx),       %%ymm1            \n\t"
    "vfmadd213pd      %%ymm11, %%ymm3,  %%ymm1   \n\t"
    "vmovapd          %%ymm1,  (%%rdx)           \n\t"
    "addq      %%rdi, %%rdx                      \n\t"
    "vmovapd    (%%r12),       %%ymm2            \n\t"
    "vfmadd213pd      %%ymm12, %%ymm3,  %%ymm2   \n\t"
    "vmovapd          %%ymm2,  (%%r12)           \n\t"
    "addq      %%rdi, %%r12                      \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd    (%%rcx),       %%ymm0            \n\t"
    "vfmadd213pd      %%ymm13, %%ymm3,  %%ymm0   \n\t"
    "vmovapd          %%ymm0,  (%%rcx)           \n\t"
    //"addq      %%rdi, %%rcx                      \n\t"
    "vmovapd    (%%rdx),       %%ymm1            \n\t"
    "vfmadd213pd      %%ymm14, %%ymm3,  %%ymm1   \n\t"
    "vmovapd          %%ymm1,  (%%rdx)           \n\t"
    //"addq      %%rdi, %%rdx                      \n\t"
    "vmovapd    (%%r12),       %%ymm2            \n\t"
    "vfmadd213pd      %%ymm15, %%ymm3,  %%ymm2   \n\t"
    "vmovapd          %%ymm2,  (%%r12)           \n\t"
    //"addq      %%rdi, %%r12                      \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "jmp    .DDONE                               \n\t" // jump to end.
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".DBETAZERO:                                 \n\t"
    "                                            \n\t" // check if aligned/row-stored
    "andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
    "andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
    "jne     .DROWSTORBZ                         \n\t" // jump to row storage case
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".DGENSTORBZ:                                \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd           %%ymm4,  %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd           %%ymm7,  %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd           %%ymm10, %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd           %%ymm13, %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    //"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "movq      %%rdx, %%rcx                      \n\t" // rcx = c + 4*cs_c
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd           %%ymm5,  %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd           %%ymm8,  %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd           %%ymm11, %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd           %%ymm14, %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    //"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "movq      %%r12, %%rcx                      \n\t" // rcx = c + 8*cs_c
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd           %%ymm6,  %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd           %%ymm9,  %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd           %%ymm12, %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd           %%ymm15, %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    //"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "jmp    .DDONE                               \n\t" // jump to end.
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".DROWSTORBZ:                                \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd          %%ymm4,  (%%rcx)           \n\t"
    "addq      %%rdi, %%rcx                      \n\t"
    "vmovapd          %%ymm5,  (%%rdx)           \n\t"
    "addq      %%rdi, %%rdx                      \n\t"
    "vmovapd          %%ymm6,  (%%r12)           \n\t"
    "addq      %%rdi, %%r12                      \n\t"
    "                                            \n\t"
    "vmovapd          %%ymm7,  (%%rcx)           \n\t"
    "addq      %%rdi, %%rcx                      \n\t"
    "vmovapd          %%ymm8,  (%%rdx)           \n\t"
    "addq      %%rdi, %%rdx                      \n\t"
    "vmovapd          %%ymm9,  (%%r12)           \n\t"
    "addq      %%rdi, %%r12                      \n\t"
    "                                            \n\t"
    "vmovapd          %%ymm10, (%%rcx)           \n\t"
    "addq      %%rdi, %%rcx                      \n\t"
    "vmovapd          %%ymm11, (%%rdx)           \n\t"
    "addq      %%rdi, %%rdx                      \n\t"
    "vmovapd          %%ymm12, (%%r12)           \n\t"
    "addq      %%rdi, %%r12                      \n\t"
    "                                            \n\t"
    "vmovapd          %%ymm13, (%%rcx)           \n\t"
    //"addq      %%rdi, %%rcx                      \n\t"
    "vmovapd          %%ymm14, (%%rdx)           \n\t"
    //"addq      %%rdi, %%rdx                      \n\t"
    "vmovapd          %%ymm15, (%%r12)           \n\t"
    //"addq      %%rdi, %%r12                      \n\t"
    "                                            \n\t"
    "                                            \n\t"
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
      "m" (cs_c)/*,   // 8
      "m" (b_next), // 9
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
