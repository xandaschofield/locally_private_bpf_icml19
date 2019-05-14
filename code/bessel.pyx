#!python
#cython: boundscheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: wraparound=False
#distutils: language = c
#distutils: extra_link_args = ["-lgsl", "-lgslcblas"]
#distutils: extra_compile_args = -Wno-unused-function
"""Code file for sampling from Bessel distribution. Includes boilerplate
code for initializing random seeds and data structures for storing samples
and caching for rejection sampling. Also includes a test of whether the
direct and recursive definitions of the Bessel distribution pmf match. For
actual sampling code, go to the header file bessel.pxd.
"""

import sys
import numpy as np
cimport numpy as np
from libc.math cimport fabs, exp, log, sqrt


cdef extern from "gsl/gsl_errno.h" nogil:
    ctypedef struct gsl_error_handler_t:
        pass
    gsl_error_handler_t * gsl_set_error_handler_off()
gsl_set_error_handler_off()

cpdef double log1pexp(double x) nogil:
    return _log1pexp(x)

cpdef double pmf_unnorm(int y, double v, double a) nogil:
    return _pmf_unnorm(y, v, a)

cpdef double logpmf_unnorm(int y, double v, double a) nogil:
    return _logpmf_unnorm(y, v, a)

cpdef double pmf_norm(double v, double a) nogil:
    return _pmf_norm(v, a)

cpdef double logpmf_norm(double v, double a) nogil:
    return _logpmf_norm(v, a)

cpdef void test_logpmf_norm(double v, double a):
    if isnan(_logpmf_norm(v, a)):
        print 'NAN'

    if isinf(_logpmf_norm(v, a)):
        print 'INF'

cpdef int mode(double v, double a) nogil:
    return _mode(v, a)

cpdef double mean(double v, double a) nogil:
    return _mean(v, a)

cpdef void vec_mean(double[::1] v_N, double[::1] a_N, double[::1] out_N) nogil:
    cdef:
        int N, n

    N = a_N.shape[0]
    for n in range(N):
        out_N[n] = _mean(v_N[n], a_N[n])

cpdef double mean_naive(double v, double a) nogil:
    return _mean_naive(v, a)

cpdef double variance(double v, double a) nogil:
    return _variance(v, a)

cpdef double quotient(double v, double a) nogil:
    return _quotient(v, a)

cpdef void run_sample(double v, double a, int seed, int[::1] out) nogil:
    
    cdef:
        gsl_rng *rng
        int N

    rng = gsl_rng_alloc(gsl_rng_mt19937)
    gsl_rng_set(rng, seed)

    N = out.shape[0]

    for n in range(N):
        out[n] = _sample(rng, v, a)

    gsl_rng_free(rng)


DEF CACHE_SIZE = 4

cpdef void run_sample_from_pmf(double v, double a, int seed, int[::1] out) nogil:
    
    cdef:
        gsl_rng *rng
        int N
        double[CACHE_SIZE] cache

    cache[0] = NAN

    rng = gsl_rng_alloc(gsl_rng_mt19937)
    gsl_rng_set(rng, seed)

    N = out.shape[0]

    for n in range(N):
        out[n] = _sample_from_pmf(rng, v, a, cache)

    gsl_rng_free(rng)

cpdef unsigned int test_log_pmf_update(double v, double a, double tol) nogil:
    """Test function for whether using the recursive definition of the Bessel
    PMF matches the value generated directly by _logpmf_unnorm.
    """
    cdef:
        double z, c, f, g
        unsigned int y
        int m, i

    z = _logpmf_norm(v, a)
    y = m = _mode(v, a)
    f = g = _logpmf_unnorm(y, v, a)

    c = 2 * log(a) - log(4)

    for i in range(m):
        # test each y between 0 and m - 1
        y = m - i - 1
        g += log(y + 1) + log(y + v + 1) - c
        if fabs(_logpmf_unnorm(y, v, a) - g) > tol:
            return 0

        # test each y between m + 1 and 2 * m
        y = m + i + 1
        f += c - log(y) - log(y + v)
        if fabs(_logpmf_unnorm(y, v, a) - f) > tol:
            return 0

    # test the long tail of y above 2 * m
    y = 2 * m + 1
    for i in range(10000):
        f += c - log(y) - log(y + v)
        if fabs(_logpmf_unnorm(y, v, a) - f) > tol:
            return 0
        else:
            y += 1

    return 1 


# Some boilerplate code for initializing variables for running
# different Bessel samplers and computing the expected number of
# iterations required for rejection sampling.

cpdef double expected_iter_rejection_1(double v, double a) nogil:
    cdef:
        double z, fm, pm
        int m

    z = _logpmf_norm(v, a)
    m = _mode(v, a)
    fm = _logpmf_unnorm(m, v, a)
    pm = exp(fm - z)
    return 4 + pm

cpdef void run_rejection_1(double v, double a, int seed, int[::1] out) nogil:
    
    cdef:
        gsl_rng *rng
        int N
        double[CACHE_SIZE] cache

    cache[0] = NAN

    rng = gsl_rng_alloc(gsl_rng_mt19937)
    gsl_rng_set(rng, seed)

    N = out.shape[0]

    for n in range(N):
        out[n] = _rejection_1(rng, v, a, cache)

    gsl_rng_free(rng)


cpdef double expected_iter_rejection_2(double v, double a) nogil:
    cdef:
        double z, fm, pm, s, q
        int m

    m = _mode(v, a)
    z = _logpmf_norm(v, a)
    fm = _logpmf_unnorm(m, v, a)
    pm = exp(fm - z)

    s = sqrt(_variance(v, a))
    q = 1. / (s * sqrt(648))
    if q > (1. / 3):
        q = (1. / 3)

    return pm * (1 + 4. / q)

cpdef void run_rejection_2(double v, double a, int seed, int[::1] out) nogil:
    
    cdef:
        gsl_rng *rng
        int N
        double[CACHE_SIZE] cache

    cache[0] = NAN

    rng = gsl_rng_alloc(gsl_rng_mt19937)
    gsl_rng_set(rng, seed)

    N = out.shape[0]

    for n in range(N):
        out[n] = _rejection_2(rng, v, a, cache)

    gsl_rng_free(rng)


cpdef double expected_iter_rejection_3(double v, double a) nogil:
    cdef:
        double z, fm, pm, s, q
        int m

    m = _mode(v, a)
    z = _logpmf_norm(v, a)
    fm = _logpmf_unnorm(m, v, a)
    pm = exp(fm - z)

    s = sqrt(_variance_bound(v, a))
    q = 1. / (s * sqrt(648))
    if q > (1. / 3):
        q = (1. / 3)

    return pm * (1 + 4. / q)

cpdef void run_rejection_3(double v, double a, int seed, int[::1] out) nogil:
    
    cdef:
        gsl_rng *rng
        int N
        double[CACHE_SIZE] cache

    cache[0] = NAN

    rng = gsl_rng_alloc(gsl_rng_mt19937)
    gsl_rng_set(rng, seed)

    N = out.shape[0]

    for n in range(N):
        out[n] = _rejection_3(rng, v, a, cache)

    gsl_rng_free(rng)


cpdef double expected_iter_double_poisson(double v, double a) nogil:
    return exp(a - _logpmf_norm(v, a))

cpdef void run_double_poisson(int v, double a, int seed, int[::1] out) nogil:
    
    cdef:
        gsl_rng *rng
        int N

    rng = gsl_rng_alloc(gsl_rng_mt19937)
    gsl_rng_set(rng, seed)

    N = out.shape[0]

    for n in range(N):
        out[n] = _double_poisson(rng, v, a)

    gsl_rng_free(rng)


cpdef void run_normal_approx(double v, double a, int seed, int[::1] out) nogil:
    
    cdef:
        gsl_rng *rng
        int N

    rng = gsl_rng_alloc(gsl_rng_mt19937)
    gsl_rng_set(rng, seed)

    N = out.shape[0]

    for n in range(N):
        out[n] = _normal_approx(rng, v, a)

    gsl_rng_free(rng)
