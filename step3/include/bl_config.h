#ifndef BLISLAB_CONFIG_H
#define BLISLAB_CONFIG_H

// Allow C++ users to include this header file in their source code. However,
// we make the extern "C" conditional on whether we're using a C++ compiler,
// since regular C compilers don't understand the extern "C" construct.
#ifdef __cplusplus
extern "C" {
#endif

#define GEMM_SIMD_ALIGN_SIZE 32

#define DGEMM_MC 96
#define DGEMM_NC 4096
#define DGEMM_KC 256
#define DGEMM_MR 8
#define DGEMM_NR 4

//#define DGEMM_MC 72
//#define DGEMM_NC 4080
//#define DGEMM_KC 256
//#define DGEMM_MR 8
//#define DGEMM_NR 6
////#define DGEMM_MR 12
////#define DGEMM_NR 4
////#define DGEMM_MR 6
////#define DGEMM_NR 8

//#define BL_MICRO_KERNEL bl_dgemm_int_8x4
//#define BL_MICRO_KERNEL bl_dgemm_asm_8x4
//#define BL_MICRO_KERNEL bl_dgemm_asm_8x6
//#define BL_MICRO_KERNEL bl_dgemm_asm_6x8
//#define BL_MICRO_KERNEL bl_dgemm_asm_12x4
#define BL_MICRO_KERNEL bl_dgemm_ukr

// End extern "C" construct block.
#ifdef __cplusplus
}
#endif

#endif
