#!python
#cython: boundscheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: wraparound=False
#distutils: extra_link_args = ['-lgsl', '-lgslcblas']
#distutils: extra_compile_args = -Wno-unused-function
"""Implements private inference for Bayesian Private Stochastic Block Model,
both for data generation and inference."""

import sys

from cython.parallel import prange
from libc.math cimport exp, log, sqrt, fabs
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
from openmp cimport omp_get_max_threads, omp_get_thread_num, omp_set_num_threads

from bessel cimport _sample as _sample_bessel
from bessel cimport _mode as _mode_bessel
from mcmc_model cimport MCMCModel
from sample cimport _sample_gamma
from sample cimport _sample_uniformint


cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng:
        pass

cdef extern from "gsl/gsl_randist.h" nogil:
    double gsl_rng_uniform(gsl_rng * r)
    double gsl_ran_exponential(gsl_rng * r, double mu)
    unsigned int gsl_ran_poisson(gsl_rng * r, double mu)
    unsigned int gsl_ran_binomial(gsl_rng * r, double p, unsigned int n)
    void gsl_ran_multinomial(gsl_rng * r,
                             size_t K,
                             unsigned int N,
                             const double p[],
                             unsigned int n[])

cdef unsigned int MAX_THREADS = 35

cdef class BPSBM(MCMCModel):

    cdef:
        int V, C, t, sym, icm, trunc, hybrid, debug, n_threads, hybrid_threshold
        double a, b
        double[::1] Theta_C, P_C
        double[:,::1] Theta_VC, Pi_CC, mu_VV, priv_VV, P_CC
        double[:,:,::1] Lambda_2VV
        int[::1] mask_1_V, mask_2_V
        int[:,::1] Y_CC, Y_VC, Y_VV, data_VV, mask_VV
        int[:,:,::1] G_2VV
        unsigned int[::1] N_c_C, N_d_C

    def __init__(self, int V, int C, double a=0.1, double b=0.1,
                 int icm=0, int sym=1, int trunc=0, int hybrid=0, int hybrid_threshold=0, int debug=0, 
                 object seed=None):

        self.n_threads = omp_get_max_threads()
        if self.n_threads > MAX_THREADS:
            self.n_threads = MAX_THREADS
            omp_set_num_threads(MAX_THREADS)
        super(BPSBM, self).__init__(seed, self.n_threads)
        self.print_every = 25

        # Params
        self.V = self.param_list['V'] = V
        self.C = self.param_list['C'] = C
        self.a = self.param_list['a'] = a
        self.b = self.param_list['b'] = b
        self.sym = self.param_list['sym'] = sym
        self.icm = self.param_list['icm'] = icm
        self.trunc = self.param_list['trunc'] = trunc
        self.hybrid = self.param_list['hybrid'] = hybrid
        self.hybrid_threshold = self.param_list['hybrid_threshold'] = hybrid_threshold
        self.debug = self.param_list['debug'] = debug

        # State variables
        self.Pi_CC = np.zeros((C, C))
        self.Theta_VC = np.zeros((V, C))
        self.Y_VV = np.zeros((V, V), dtype=np.int32)
        self.Y_VC = np.zeros((V, C), dtype=np.int32)
        self.Y_CC = np.zeros((C, C), dtype=np.int32)
        self.Lambda_2VV = np.zeros((2, V, V))
        self.G_2VV = np.zeros((2, V, V), dtype=np.int32)
        # self.f = 1.

        # Cache
        self.mu_VV = np.zeros((V, V))
        self.Theta_C = np.zeros(C)

        # Auxiliary
        self.P_C = np.zeros(C)
        self.P_CC = np.zeros((C, C))
        self.N_c_C = np.zeros(C, dtype=np.uint32)
        self.N_d_C = np.zeros(C, dtype=np.uint32)

        # Data 
        self.data_VV = np.zeros((V, V), dtype=np.int32)
        self.mask_VV = np.ones((V, V), dtype=np.int32)
        self.mask_1_V = np.ones(V, dtype=np.int32)
        self.mask_2_V = np.ones(V, dtype=np.int32)
        self.priv_VV = np.zeros((V, V))

    cdef list _get_variables(self):
        """
        Return variable names, values, and sampling methods for testing.
        """
        variables = []
        if not self.trunc:
            variables = [('Y_VV', self.Y_VV, lambda x: None),
                         ('G_2VV', self.G_2VV, self._update_G_2VV),
                         ('Lambda_2VV', self.Lambda_2VV, self._update_Lambda_2VV)]
        variables += [('Y_VC', self.Y_VC, self._update_Y_VCCV),
                      ('Y_CC', self.Y_CC, lambda x: None),
                      ('Pi_CC', self.Pi_CC, self._update_Pi_CC),
                      ('Theta_VC', self.Theta_VC, self._update_Theta_VC)]
        return variables

    def fit(self, data, alpha=0.25, mask=None, num_itns=1000, verbose=True, initialize=True, schedule={}):
        assert data.shape == (self.V, self.V)
        data_VV = data.astype(np.int32)
        if self.trunc:
            data_VV[data < 0] = 0
            self.priv_VV[:] = 0.
        else:
            if isinstance(alpha, np.ndarray):
                assert alpha.shape == (self.V, self.V)
                assert (0 <= alpha).all() and (alpha < 1).all()
                self.priv_VV = alpha.copy()

            else:
                assert np.isscalar(alpha)
                assert (0 <= alpha).all() and (alpha < 1).all()
                self.priv_VV[:] = alpha

        if self.hybrid:
            data_VV[data < self.hybrid_threshold] = 0
            is_nonzero_data = (data_VV > 0)
            for v in xrange(self.V):
                for w in xrange(self.V):
                    self.priv_VV[v, w] *= is_nonzero_data[v, w]
    
        self.data_VV = data_VV
        Y_VV = np.copy(data_VV)
        Y_VV[Y_VV < 0] = 0
        self.Y_VV = Y_VV

        if mask is None:
            mask = np.ones((self.V, self.V), dtype=np.int32)
        assert mask.shape == (self.V, self.V)
        assert all(x in [0, 1] for x in np.unique(mask))
        self.mask_1_V = mask.prod(axis=1, dtype=np.int32)
        self.mask_2_V = mask.prod(axis=0, dtype=np.int32)
        self.mask_VV = mask.astype(np.int32)

        if initialize:
            self._init_state()

        self._update(num_itns=num_itns, verbose=int(verbose), schedule=schedule)

    def reconstruct(self, subs=(), partial_state={}):
        pass

    cdef void _init_state(self):
        """
        Generate internal state.
        """
        cdef:
            int pc, d, v, i, pi, j, n, thread_idx
            double shp, sca, theta_ic, pi_cd, mu_ij, lam_ijn, alpha_ij
            gsl_rng ** rngs

        with nogil:
            shp = self.a
            sca = 1. / self.b
            rngs = self.rngs

            for pc in prange(self.C, schedule='dynamic'):
                thread_idx = omp_get_thread_num()
                for i in range(self.V):
                    theta_ic = _sample_gamma(rngs[thread_idx], shp, sca)
                    self.Theta_VC[i, pc] = theta_ic
                    self.Theta_C[pc] += theta_ic

                for d in range(self.C):
                    if self.sym and (d < pc):
                        # TODO: check this
                        self.Pi_CC[pc, d] = self.Pi_CC[d, pc]
                    else:
                        pi_cd = _sample_gamma(rngs[thread_idx], shp, sca)
                        self.Pi_CC[pc, d] = pi_cd

            for pi in prange(self.V, schedule='dynamic'):
                thread_idx = omp_get_thread_num()
                for j in range(self.V):
                    if pi == j:
                        continue
                    alpha_ij = self.priv_VV[pi, j]
                    mu_ij = alpha_ij / (1 - alpha_ij)
                    for n in range(2):
                        self.Lambda_2VV[n, pi, j] = lam_ijn = gsl_ran_exponential(rngs[thread_idx], mu_ij)
                        self.G_2VV[n, pi, j] = gsl_ran_poisson(rngs[thread_idx], lam_ijn)

    cdef void _print_state(self):
        print 'ITERATION: %d' % self.total_itns

    cdef void _generate_state(self):
        """
        Generate internal state.
        """
        cdef:
            int c, pc, d, i, j, y_icdj, thread_idx
            double shp, sca, theta_ic, pi_cd, theta_jd, mu_icdj
            gsl_rng * rng
            gsl_rng ** rngs

        with nogil:
            shp = self.a
            sca = 1. / self.b
            rng = self.rng
            rngs = self.rngs

            self.Theta_C[:] = 0
            for pc in prange(self.C, schedule='dynamic'):
                thread_idx = omp_get_thread_num()
                for i in range(self.V):
                    theta_ic = _sample_gamma(rngs[thread_idx], shp, sca)
                    self.Theta_VC[i, pc] = theta_ic
                    self.Theta_C[pc] += theta_ic

                for d in range(self.C):
                    if self.sym and (d < pc):
                        self.Pi_CC[pc, d] = self.Pi_CC[d, pc]
                    else:
                        pi_cd = _sample_gamma(rngs[thread_idx], shp, sca)
                        self.Pi_CC[pc, d] = pi_cd

            self.Y_VC[:] = 0
            self.Y_CC[:] = 0  
            self.Y_VV[:] = 0
            self.mu_VV[:] = 0

            for c in range(self.C):
                for d in range(self.C):
                    pi_cd = self.Pi_CC[c, d]
                    for i in range(self.V):
                        theta_ic = self.Theta_VC[i, c]
                        for j in range(self.V):
                            if i == j:
                                continue
                            theta_jd = self.Theta_VC[j, d]
                            mu_icdj = theta_ic * pi_cd * theta_jd
                            self.mu_VV[i, j] += mu_icdj
                            y_icdj = gsl_ran_poisson(rng, mu_icdj)
                            self.Y_VC[i, c] += y_icdj
                            self.Y_VC[j, d] += y_icdj
                            self.Y_CC[c, d] += y_icdj
                            self.Y_VV[i, j] += y_icdj
                        
                        # QUESTION: is this indentation off?
                        self.priv_VV[i, j] = 0.1

    cdef void _generate_data(self):
        """
        Generate data given internal state.
        """
        cdef:
            int pi, j, g_ij1, g_ij2, thread_idx
            double alpha_ij, mu_ij, lam_ij1, lam_ij2
            gsl_rng ** rngs

        with nogil:
            self.data_VV[:] = 0
            rngs = self.rngs
            for pi in prange(self.V):
                thread_idx = omp_get_thread_num()
                for j in range(self.V):
                    if pi == j:
                        continue
                    
                    self.data_VV[pi, j] = self.Y_VV[pi, j]
                    alpha_ij = self.priv_VV[pi, j]
                    mu_ij = alpha_ij / (1 - alpha_ij)

                    self.Lambda_2VV[0, pi, j] = lam_ij1 = gsl_ran_exponential(rngs[thread_idx], mu_ij)
                    self.Lambda_2VV[1, pi, j] = lam_ij2 = gsl_ran_exponential(rngs[thread_idx], mu_ij)

                    self.G_2VV[0, pi, j] = g_ij1 = gsl_ran_poisson(rngs[thread_idx], lam_ij1)
                    self.G_2VV[1, pi, j] = g_ij2 = gsl_ran_poisson(rngs[thread_idx], lam_ij2)

                    self.data_VV[pi, j] += g_ij1 - g_ij2

    cdef void _update_Y_VCCV(self) nogil:
        cdef:
            int i, j, c, d, y_ij, y_icj, y_icdj
            double theta_ic, theta_jd, pi_cd

        self.Y_CC[:] = 0
        self.Y_VC[:] = 0
        for i in range(self.V):
            for j in range(self.V):
                if i == j:
                    continue
                
                y_ij = self.Y_VV[i, j]
                
                self.P_C[:] = 0
                for c in range(self.C):
                    theta_ic = self.Theta_VC[i, c]
                    for d in range(self.C):
                        theta_jd = self.Theta_VC[j, d]
                        pi_cd = self.Pi_CC[c, d]
                        self.P_CC[c, d] = pi_cd * theta_jd
                        self.P_C[c] += self.P_CC[c, d]
                    self.P_C[c] *= theta_ic

                gsl_ran_multinomial(self.rng,
                                    self.C,
                                    <unsigned int> y_ij,
                                    &self.P_C[0],
                                    &self.N_c_C[0])

                for c in range(self.C):
                    y_icj = self.N_c_C[c]
                    if y_icj == 0:
                        continue

                    gsl_ran_multinomial(self.rng,
                                        self.C,
                                        <unsigned int> y_icj,
                                        &self.P_CC[c, 0],
                                        &self.N_d_C[0])

                    for d in range(self.C):
                        y_icdj = self.N_d_C[d]
                        if y_icdj == 0:
                            continue
                        self.Y_CC[c, d] += y_icdj
                        self.Y_VC[i, c] += y_icdj
                        self.Y_VC[j, d] += y_icdj

    cdef void _update_Theta_VC(self) nogil:
        cdef:
            int i, c, d
            double a, b, theta_ic, shp_ic, rte_ic

        a = self.a
        b = self.b

        for i in range(self.V):
            for c in range(self.C):
                self.Theta_C[c] -= self.Theta_VC[i, c]

            for c in range(self.C):
                shp_ic = a + self.Y_VC[i, c]
                rte_ic = b
                for d in range(self.C):
                    rte_ic += self.Theta_C[d] * (self.Pi_CC[c, d] + self.Pi_CC[d, c])
                
                theta_ic = _sample_gamma(self.rng, shp_ic, 1./rte_ic)
                self.Theta_VC[i, c] = theta_ic
            
            for c in range(self.C):    
                self.Theta_C[c] += self.Theta_VC[i, c]

    cdef void _update_Pi_CC(self):
        cdef:
            int c, d
            double a, b, shp_cd, sca_cd
            double[:,::1] rte_CD

        a = self.a
        b = self.b

        Theta_C = np.sum(self.Theta_VC, axis=0)
        Theta_CI = np.transpose(self.Theta_VC)
        rte_CD = np.outer(Theta_C, Theta_C) - np.dot(Theta_CI, self.Theta_VC)

        for c in range(self.C):
            for d in range(self.C):
                shp_cd = a + self.Y_CC[c, d]
                sca_cd = 1. / (b + rte_CD[c, d])
                self.Pi_CC[c, d] = _sample_gamma(self.rng, shp_cd, sca_cd)

    cdef void _update_Lambda_2VV(self) nogil:
        cdef: 
            int pi, j, n, thread_idx
            double p_ij, shp_ijn
            gsl_rng ** rngs

        rngs = self.rngs
        for pi in prange(self.V, schedule='dynamic'):
            thread_idx = omp_get_thread_num()
            for j in range(self.V):
                if pi == j:
                    continue
                p_ij = self.priv_VV[pi, j]
                for n in range(2):
                    shp_ijn = 1. + self.G_2VV[n, pi, j]
                    self.Lambda_2VV[n, pi, j] = _sample_gamma(self.rngs[thread_idx], shp_ijn, p_ij)

    cdef void _update_G_2VV(self):
        cdef:
            int pi, j, o_ij, m_ij, y_p_ij, g_ij1, g_ij2, y_ij, thread_idx
            double shp_ij, mu_ij, lam_ij1, lam_ij2, p_ij
            gsl_rng ** rngs

        self.mu_VV = np.einsum('ik,kl,jl->ij', self.Theta_VC, self.Pi_CC, self.Theta_VC)
        rngs = self.rngs

        with nogil:
            for pi in prange(self.V, schedule='dynamic'):
                thread_idx = omp_get_thread_num()
                for j in range(self.V):
                    if pi == j:
                        continue

                    if self.priv_VV[pi, j] == 0:
                        continue

                    o_ij = self.data_VV[pi, j]
                    mu_ij = self.mu_VV[pi, j]
                    lam_ij1 = self.Lambda_2VV[0, pi, j]
                    lam_ij2 = self.Lambda_2VV[1, pi, j]
                    shp_ij = 2 * sqrt((lam_ij1 + mu_ij) * lam_ij2)

                    if self.icm:
                        m_ij = _mode_bessel(fabs(o_ij + 0.), shp_ij)
                    else:
                        m_ij = _sample_bessel(self.rngs[thread_idx], fabs(o_ij + 0.), shp_ij)
                    # assert m_dv >= 0
                    
                    if o_ij <= 0:
                        y_p_ij = m_ij
                        g_ij2 = y_p_ij - o_ij
                    else:
                        g_ij2 = m_ij
                        y_p_ij = g_ij2 + o_ij

                    p_ij = mu_ij / (mu_ij + lam_ij1)
                    y_ij = gsl_ran_binomial(self.rngs[thread_idx], p_ij, y_p_ij)
                    g_ij1 = y_p_ij - y_ij

                    self.Y_VV[pi, j] = y_ij
                    self.G_2VV[0, pi, j] = g_ij1
                    self.G_2VV[1, pi, j] = g_ij2
