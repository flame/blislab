#include <bl_config.h>
#include "bl_dgemm_kernel.h"

// micro-panel a is stored in column major, lda=DGEMM_MR.
#define a(i, j) a[(j)*DGEMM_MR + (i)]
// micro-panel b is stored in row major, ldb=DGEMM_NR.
#define b(i, j) b[(i)*DGEMM_NR + (j)]
// result      c is stored in column major.
#define c(i, j) c[(j)*ldc + (i)]

void bl_dgemm_ukr(
    int k,
    double* a,
    double* b,
    double* c,
    unsigned long long ldc,
    aux_t* data) {
  int l, j, i;

  for (l = 0; l < k; ++l) {
    for (j = 0; j < DGEMM_NR; ++j) {
      for (i = 0; i < DGEMM_MR; ++i) {
        c(i, j) += a(i, l) * b(l, j);
      }
    }
  }
}
