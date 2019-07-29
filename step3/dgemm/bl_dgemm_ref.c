#include <bl_dgemm.h>

#ifdef USE_BLAS
/*
 * dgemm prototype
 *
 */
// void dgemm(char*, char*, int*, int*, int*, double*, double*,
//        int*, double*, int*, double*, double*, int*);
extern void dgemm_(
    char*,
    char*,
    int*,
    int*,
    int*,
    double*,
    double*,
    int*,
    double*,
    int*,
    double*,
    double*,
    int*);
#endif

void bl_dgemm_ref(
    int m,
    int n,
    int k,
    double* XA,
    int lda,
    double* XB,
    int ldb,
    double* XC,
    int ldc) {
  // Local variables.
  int i, j, p;
  double alpha = 1.0, beta = 1.0;

  // Sanity check for early return.
  if (m == 0 || n == 0 || k == 0)
    return;

    // Reference GEMM implementation.

#ifdef USE_BLAS
  dgemm_("N", "N", &m, &n, &k, &alpha, XA, &lda, XB, &ldb, &beta, XC, &ldc);
#else
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      for (p = 0; p < k; p++) {
        XC[j * ldc + i] += XA[p * lda + i] * XB[j * ldb + p];
      }
    }
  }
#endif
}
