BLISlab: A Sandbox for Optimizing GEMM

Matrix-matrix multiplication is a fundamental operation of great
importance to scientific computing and, increasingly, machine learning.
It is a simple enough concept to be introduced in a typical high school
algebra course yet in practice important enough that its implementation
on computers continues to be an active research topic. This note
describes a set of exercises that use this operation to illustrate how
high performance can be attained on modern CPUs with hierarchical
memories (multiple caches). It does so by building on the insights that
underly the BLAS-like Library Instantiation Softare (BLIS) framework by
exposing a simplified “sandbox” that mimics the implementation in BLIS.
As such, it also becomes a vehicle for the “crowd sourcing” of the
optimization of BLIS. We call this set of exercises [BLISlab](https://github.com/flame/blislab).

Check the [tutorial](https://github.com/flame/blislab/blob/master/tutorial.pdf) for more details.

# Related Links
* [How to Optimize GEMM Wiki] (https://github.com/flame/how-to-optimize-gemm/wiki)
* [GEMM: From Pure C to SSE Optimized Micro Kernals] (http://apfel.mathematik.uni-ulm.de/~lehn/sghpc/gemm/)

# Acknowledgement
This material was partially sponsored by grants from the National Science Foundation (Awards ACI-1148125/1340293).

_Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation (NSF)._
