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

void bl_dgemm_asm_6x8(
    int k,
    double* a,
    double* b,
    double* c,
    inc_t ldc,
    aux_t* data) {
  // void*   a_next = bli_auxinfo_next_a( data );
  // void*   b_next = bli_auxinfo_next_b( data );

  // uint64_t   k_iter = k / 4;
  // uint64_t   k_left = k % 4;

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
    "vmovaps           -4 * 32(%%rbx), %%ymm0    \n\t"
    "vmovaps           -3 * 32(%%rbx), %%ymm1    \n\t"
    "                                            \n\t"
    "movq                %6, %%rcx               \n\t" // load address of c
    "movq                %7, %%rdi               \n\t" // load rs_c
    "leaq        (,%%rdi,8), %%rdi               \n\t" // rs_c *= sizeof(double)
    "                                            \n\t"
    "leaq   (%%rdi,%%rdi,2), %%r13               \n\t" // r13 = 3*rs_c;
    "leaq   (%%rcx,%%r13,1), %%rdx               \n\t" // rdx = c + 3*rs_c;
    "prefetcht0   7 * 8(%%rcx)                   \n\t" // prefetch c + 0*rs_c
    "prefetcht0   7 * 8(%%rcx,%%rdi)             \n\t" // prefetch c + 1*rs_c
    "prefetcht0   7 * 8(%%rcx,%%rdi,2)           \n\t" // prefetch c + 2*rs_c
    "prefetcht0   7 * 8(%%rdx)                   \n\t" // prefetch c + 3*rs_c
    "prefetcht0   7 * 8(%%rdx,%%rdi)             \n\t" // prefetch c + 4*rs_c
    "prefetcht0   7 * 8(%%rdx,%%rdi,2)           \n\t" // prefetch c + 5*rs_c
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
    "prefetcht0   64 * 8(%%rax)                  \n\t"
    "                                            \n\t"
    "vbroadcastsd       0 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd       1 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
    "                                            \n\t"
    "vbroadcastsd       2 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd       3 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
    "                                            \n\t"
    "vbroadcastsd       4 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd       5 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
    "                                            \n\t"
    "vmovaps           -2 * 32(%%rbx), %%ymm0    \n\t"
    "vmovaps           -1 * 32(%%rbx), %%ymm1    \n\t"
    "                                            \n\t"
    "                                            \n\t" // iteration 1
    "vbroadcastsd       6 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd       7 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
    "                                            \n\t"
    "vbroadcastsd       8 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd       9 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
    "                                            \n\t"
    "vbroadcastsd      10 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd      11 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
    "                                            \n\t"
    "vmovaps            0 * 32(%%rbx), %%ymm0    \n\t"
    "vmovaps            1 * 32(%%rbx), %%ymm1    \n\t"
    "                                            \n\t"
    "                                            \n\t" // iteration 2
    "prefetcht0   76 * 8(%%rax)                  \n\t"
    "                                            \n\t"
    "vbroadcastsd      12 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd      13 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
    "                                            \n\t"
    "vbroadcastsd      14 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd      15 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
    "                                            \n\t"
    "vbroadcastsd      16 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd      17 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
    "                                            \n\t"
    "vmovaps            2 * 32(%%rbx), %%ymm0    \n\t"
    "vmovaps            3 * 32(%%rbx), %%ymm1    \n\t"
    "                                            \n\t"
    "                                            \n\t" // iteration 3
    "vbroadcastsd      18 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd      19 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
    "                                            \n\t"
    "vbroadcastsd      20 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd      21 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
    "                                            \n\t"
    "vbroadcastsd      22 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd      23 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
    "                                            \n\t"
    "addq           $4 * 6 * 8, %%rax            \n\t" // a += 4*6 (unroll x mr)
    "addq           $4 * 8 * 8, %%rbx            \n\t" // b += 4*8 (unroll x nr)
    "                                            \n\t"
    "vmovaps           -4 * 32(%%rbx), %%ymm0    \n\t"
    "vmovaps           -3 * 32(%%rbx), %%ymm1    \n\t"
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
    "prefetcht0   64 * 8(%%rax)                  \n\t"
    "                                            \n\t"
    "vbroadcastsd       0 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd       1 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
    "                                            \n\t"
    "vbroadcastsd       2 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd       3 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
    "                                            \n\t"
    "vbroadcastsd       4 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd       5 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
    "                                            \n\t"
    "addq           $1 * 6 * 8, %%rax            \n\t" // a += 1*6 (unroll x mr)
    "addq           $1 * 8 * 8, %%rbx            \n\t" // b += 1*8 (unroll x nr)
    "                                            \n\t"
    "vmovaps           -4 * 32(%%rbx), %%ymm0    \n\t"
    "vmovaps           -3 * 32(%%rbx), %%ymm1    \n\t"
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
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "movq                %8, %%rsi               \n\t" // load cs_c
    "leaq        (,%%rsi,8), %%rsi               \n\t" // rsi = cs_c * sizeof(double)
    "                                            \n\t"
    "leaq   (%%rcx,%%rsi,4), %%rdx               \n\t" // load address of c +  4*cs_c;
    "                                            \n\t"
    "leaq   (%%rsi,%%rsi,2), %%r13               \n\t" // r13 = 3*cs_c;
    //"leaq   (%%rsi,%%rsi,4), %%r15               \n\t" // r15 = 5*cs_c;
    //"leaq   (%%r13,%%rsi,4), %%r10               \n\t" // r10 = 7*cs_c;
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // now avoid loading C if beta == 0
    "                                            \n\t"
    "vxorpd    %%ymm0,  %%ymm0,  %%ymm0          \n\t" // set ymm0 to zero.
    "vucomisd  %%xmm0,  %%xmm3                   \n\t" // set ZF if beta == 0.
    "je      .DBETAZERO                          \n\t" // if ZF = 1, jump to beta == 0 case
    "                                            \n\t"
    "                                            \n\t"
    "cmpq       $8, %%rsi                        \n\t" // set ZF if (8*cs_c) == 8.
    "jz      .DROWSTORED                         \n\t" // jump to row storage case
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
    "vfmadd213pd      %%ymm6,  %%ymm3,  %%ymm0   \n\t"
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
    "vfmadd213pd      %%ymm10, %%ymm3,  %%ymm0   \n\t"
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
    "vfmadd213pd      %%ymm14, %%ymm3,  %%ymm0   \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
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
    "vfmadd213pd      %%ymm7,  %%ymm3,  %%ymm0   \n\t"
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
    "vfmadd213pd      %%ymm11, %%ymm3,  %%ymm0   \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    DGEMM_INPUT_GS_BETA_NZ
    "vfmadd213pd      %%ymm13, %%ymm3,  %%ymm0   \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    DGEMM_INPUT_GS_BETA_NZ
    "vfmadd213pd      %%ymm15, %%ymm3,  %%ymm0   \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
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
    "vfmadd231pd      (%%rcx), %%ymm3, %%ymm4    \n\t"
    "vmovups          %%ymm4,  (%%rcx)           \n\t"
    "addq      %%rdi, %%rcx                      \n\t"
    "vfmadd231pd      (%%rdx), %%ymm3, %%ymm5    \n\t"
    "vmovups          %%ymm5,  (%%rdx)           \n\t"
    "addq      %%rdi, %%rdx                      \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vfmadd231pd      (%%rcx), %%ymm3, %%ymm6    \n\t"
    "vmovups          %%ymm6,  (%%rcx)           \n\t"
    "addq      %%rdi, %%rcx                      \n\t"
    "vfmadd231pd      (%%rdx), %%ymm3, %%ymm7    \n\t"
    "vmovups          %%ymm7,  (%%rdx)           \n\t"
    "addq      %%rdi, %%rdx                      \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vfmadd231pd      (%%rcx), %%ymm3, %%ymm8    \n\t"
    "vmovups          %%ymm8,  (%%rcx)           \n\t"
    "addq      %%rdi, %%rcx                      \n\t"
    "vfmadd231pd      (%%rdx), %%ymm3, %%ymm9    \n\t"
    "vmovups          %%ymm9,  (%%rdx)           \n\t"
    "addq      %%rdi, %%rdx                      \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vfmadd231pd      (%%rcx), %%ymm3, %%ymm10   \n\t"
    "vmovups          %%ymm10, (%%rcx)           \n\t"
    "addq      %%rdi, %%rcx                      \n\t"
    "vfmadd231pd      (%%rdx), %%ymm3, %%ymm11   \n\t"
    "vmovups          %%ymm11, (%%rdx)           \n\t"
    "addq      %%rdi, %%rdx                      \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vfmadd231pd      (%%rcx), %%ymm3, %%ymm12   \n\t"
    "vmovups          %%ymm12, (%%rcx)           \n\t"
    "addq      %%rdi, %%rcx                      \n\t"
    "vfmadd231pd      (%%rdx), %%ymm3, %%ymm13   \n\t"
    "vmovups          %%ymm13, (%%rdx)           \n\t"
    "addq      %%rdi, %%rdx                      \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vfmadd231pd      (%%rcx), %%ymm3, %%ymm14   \n\t"
    "vmovups          %%ymm14, (%%rcx)           \n\t"
    //"addq      %%rdi, %%rcx                      \n\t"
    "vfmadd231pd      (%%rdx), %%ymm3, %%ymm15   \n\t"
    "vmovups          %%ymm15, (%%rdx)           \n\t"
    //"addq      %%rdi, %%rdx                      \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "jmp    .DDONE                               \n\t" // jump to end.
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".DBETAZERO:                                 \n\t"
    "                                            \n\t"
    "cmpq       $8, %%rsi                        \n\t" // set ZF if (8*cs_c) == 8.
    "jz      .DROWSTORBZ                         \n\t" // jump to row storage case
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".DGENSTORBZ:                                \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm4,  %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm6,  %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm8,  %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm10, %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm12, %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm14, %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "                                            \n\t"
    "                                            \n\t"
    "movq      %%rdx, %%rcx                      \n\t" // rcx = c + 4*cs_c
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm5,  %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm7,  %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm9,  %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm11, %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm13, %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
    "addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm15, %%ymm0           \n\t"
    DGEMM_OUTPUT_GS_BETA_NZ
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
    "vmovups          %%ymm4,  (%%rcx)           \n\t"
    "addq      %%rdi, %%rcx                      \n\t"
    "vmovups          %%ymm5,  (%%rdx)           \n\t"
    "addq      %%rdi, %%rdx                      \n\t"
    "                                            \n\t"
    "vmovups          %%ymm6,  (%%rcx)           \n\t"
    "addq      %%rdi, %%rcx                      \n\t"
    "vmovups          %%ymm7,  (%%rdx)           \n\t"
    "addq      %%rdi, %%rdx                      \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vmovups          %%ymm8,  (%%rcx)           \n\t"
    "addq      %%rdi, %%rcx                      \n\t"
    "vmovups          %%ymm9,  (%%rdx)           \n\t"
    "addq      %%rdi, %%rdx                      \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vmovups          %%ymm10, (%%rcx)           \n\t"
    "addq      %%rdi, %%rcx                      \n\t"
    "vmovups          %%ymm11, (%%rdx)           \n\t"
    "addq      %%rdi, %%rdx                      \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vmovups          %%ymm12, (%%rcx)           \n\t"
    "addq      %%rdi, %%rcx                      \n\t"
    "vmovups          %%ymm13, (%%rdx)           \n\t"
    "addq      %%rdi, %%rdx                      \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vmovups          %%ymm14, (%%rcx)           \n\t"
    //"addq      %%rdi, %%rcx                      \n\t"
    "vmovups          %%ymm15, (%%rdx)           \n\t"
    //"addq      %%rdi, %%rdx                      \n\t"
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
