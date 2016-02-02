#ifndef __RNN_KERNEL_H__
#define __RNN_KERNEL_H__


void rnn_rank_k_asm_d8x4(
        int    k,
        double* a,
        double* b,
        double* c,
        int    ldc,
        aux_t  *aux
        );

void sq2nrm_asm_d8x4(
        int    k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        //    unsigned long long last,
        aux_t  *aux); 
void bli_dgemm_ukr_ref( int k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        //                        unsigned long long last,
        aux_t* data );

void bli_dgemm_int_8x4( int k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t* data );

void bli_dgemm_asm_8x4( int k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t* data );

void (*blis_micro_kernel) (
        int    k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        //    unsigned long long last,
        aux_t  *aux
        ) = {
    sq2nrm_asm_d8x4
        //bli_dgemm_ukr_ref
        //bli_dgemm_asm_8x4
        //bli_dgemm_int_8x4
};

#endif // define __RNN_KERNEL_H__
