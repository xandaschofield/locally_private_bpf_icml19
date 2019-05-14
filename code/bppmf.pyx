#!python
#cython: boundscheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: wraparound=False
#distutils: extra_link_args = ['-lgsl', '-lgslcblas']
#distutils: extra_compile_args = -Wno-unused-function
"""Implements private inference for Bayesian Private Poisson
Matrix Factorization, both for data generation and inference."""
import sys

import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython.view cimport array as cvarray
from openmp cimport omp_get_max_threads, omp_get_thread_num, omp_set_num_threads
from libc.stdlib cimport malloc, free
from libc.math cimport exp, log, sqrt, fabs

from mcmc_model cimport MCMCModel
from sample cimport _sample_gamma
from sample cimport _sample_uniformint
from bessel cimport _sample as _sample_bessel
from bessel cimport _mode as _mode_bessel


# import discrete_ars as dars
cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng:
        pass

# QUESTION: why GSL? Any reason we wouldn't call through some
# other library?
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


cdef class BPPMF(MCMCModel):
    """"
    Variable Definitions
    
    This is Poisson Matrix Factorization, so we assume two parameter
    matrices: theta (D x K) and phi (K x V). Our observed variables are
    stored in a D x V matrix. The true data (or inferred true data) is
    stored in Y_DV, Y_DK, and Y_KV; the observed date is stored in data_DV.

    We have added noise to the original counts from our data in the form
    of two-sided geometric noise applied to each entry. There are multiple
    formulations of this noise function; we are using ones that are conditional
    on two gamma variables lambda_1 and lambda_2 for each of the D x V
    count variables (stored in Lambda_2DV). These are then used as parameters
    to generate two Poisson variables for the two "sides" of the noise
    (stored in G_2DV).
    
    For gamma distribution sampling, we use a for our shape parameter and
    b for our rate (inverse scale) parameter. 

    Our privacy level priv is tuned to be between 0 and 1. This can be a
    scalar or entry-wise priv_DV for each count in the D x V matrix. mu_DV
    is the corresponding exponential distribution parameter computed as
    priv/(1 - priv).

    The fit function can apply masks to certain count variables to have them
    ignored during inference, as specified by a boolean (1 or 0) mask variable,
    with 1 meaning unmasked. By default, nothing is masked.
    """ 
    cdef:
        int D, V, K, P, t, trunc, hybrid, hybrid_threshold, icm, debug, n_threads
        double a, b
        double[::1] Theta_K, Phi_K
        double [:,::1] P_PK
        double[:,::1] Theta_DK, Phi_KV, mu_DV, priv_DV
        double[:,:,::1] Lambda_2DV
        int[::1] mask_D, mask_V
        int[:,::1] Y_KV, Y_DK, Y_DV, data_DV, mask_DV
        int[:,:,::1] G_2DV
        int [:,::1] is_nonzero_data
        int [::1] thread_counts
        unsigned int[:,::1] N_PK

    def __init__(self, int D, int V, int K, double a=0.1, double b=0.1,
                 int icm=0, int trunc=0, int hybrid=0, int hybrid_threshold=0, int debug=0, object seed=None):

        self.n_threads = omp_get_max_threads()
        if self.n_threads > MAX_THREADS:
            self.n_threads = MAX_THREADS
            omp_set_num_threads(MAX_THREADS)
        super(BPPMF, self).__init__(seed, self.n_threads)
        self.print_every = 25

        # Params
        self.D = self.param_list['D'] = D
        self.V = self.param_list['V'] = V
        self.K = self.param_list['K'] = K
        self.a = self.param_list['a'] = a
        self.b = self.param_list['b'] = b
        self.icm = self.param_list['icm'] = icm
        self.trunc = self.param_list['trunc'] = trunc
        self.hybrid = self.param_list['hybrid'] = hybrid
        self.hybrid_threshold = self.param_list['hybrid_threshold'] = hybrid_threshold
        self.debug = self.param_list['debug'] = debug

        # State variables
        self.Phi_KV = np.zeros((K, V))
        self.Theta_DK = np.zeros((D, K))
        self.Y_KV = np.zeros((K, V), dtype=np.int32)
        self.Y_DK = np.zeros((D, K), dtype=np.int32)
        self.Y_DV = np.zeros((D, V), dtype=np.int32)
        self.Lambda_2DV = np.zeros((2, D, V))
        self.G_2DV = np.zeros((2, D, V), dtype=np.int32)

        # Cache
        self.Theta_K = np.zeros(K)
        self.Phi_K = np.zeros(K)
        self.mu_DV = np.zeros((D, V))

        # Auxiliary
        self.P_PK = np.zeros((self.n_threads, K))
        self.N_PK = np.zeros((self.n_threads, K), dtype=np.uint32)

        # Data 
        self.data_DV = np.zeros((D, V), dtype=np.int32)
        self.mask_DV = np.ones((D, V), dtype=np.int32)
        self.mask_D = np.ones(D, dtype=np.int32)
        self.mask_V = np.ones(V, dtype=np.int32)
        self.priv_DV = np.zeros((D, V))

    cdef list _get_variables(self):
        """
        Return variable names, values, and sampling methods for testing.
        """
        variables = []
        # QUESTION: Why do we skip these variables for truncation? Can we
        # just not test them correctly?
        # A: In the naive method, we don't need to infer these - and some
        # of the Nones are just for inference that happens elsewhere
        if not self.trunc:
            variables = [('G_2DV', self.G_2DV, self._update_G_2DV),
                         ('Lambda_2DV', self.Lambda_2DV, self._update_Lambda_2DV),
                         ('Y_DV', self.Y_DV, lambda x: None)]
        variables += [('Y_KV', self.Y_KV, self._update_Y_DVK),
                     ('Y_DK', self.Y_DK, lambda x: None),
                     ('Phi_KV', self.Phi_KV, self._update_Phi_KV),
                     ('Theta_DK', self.Theta_DK, self._update_Theta_DK)]
        return variables

    def fit(self, data, priv=0.25, mask=None, num_itns=1000, verbose=True, initialize=True, schedule={}):
        """Infer the original data, noise, and model parameters based on
        noisy data and a privacy level."""
        assert data.shape == (self.D, self.V)
        data_DV = data.astype(np.int32)

        if self.trunc:
            data_DV[data < 0] = 0
            self.priv_DV[:] = 0.

        else:
            if isinstance(priv, np.ndarray):
                assert priv.shape == (self.D, self.V)
                assert (0 <= priv).all() and (priv < 1).all()
                self.priv_DV = priv.copy()
            
            else:
                assert np.isscalar(priv)
                assert (0 <= priv).all() and (priv < 1).all()
                self.priv_DV[:] = priv

        if self.hybrid:
            data_DV[data < self.hybrid_threshold] = 0
            is_nonzero_data = (data_DV > 0)
            for d in xrange(self.D):
                for v in xrange(self.V):
                    self.priv_DV[d,v] *= is_nonzero_data[d, v]
            
        self.data_DV = data_DV
        Y_DV = np.copy(data_DV)
        Y_DV[Y_DV < 0] = 0
        self.Y_DV = Y_DV

        if mask is None:
            mask = np.ones((self.D, self.V), dtype=np.int32)
        assert mask.shape == (self.D, self.V)
        assert all(x in [0, 1] for x in np.unique(mask))
        self.mask_D = mask.prod(axis=1, dtype=np.int32)
        self.mask_V = mask.prod(axis=0, dtype=np.int32)
        self.mask_DV = mask.astype(np.int32)

        if initialize:
            self._init_state()

        self._update(num_itns=num_itns, verbose=int(verbose), schedule=schedule)

    def reconstruct(self, subs=(), partial_state={}):
        pass

    cdef void _init_state(self):
        """
        Initialize internal state randomly for inference.

        Specifically, this initializes
        - Theta_DK and Phi_KV to random samples from Gamma(a, 1/b)
        - Theta_K and Phi_K to sums over all D and V of the above matrices
        - Lambda_2DV to pairs of samples from an exponential distribution
          parameterized by priv/(1 - priv) for each entry's privacy level.
        """
        cdef:
            Py_ssize_t pk, pd, thread_idx, k, d, v
            double shp, sca
            double p_dv, phi_kv, mu_dv, theta_dk
            gsl_rng ** rngs

        with nogil:
            shp = self.a
            sca = 1. / self.b
            rngs = self.rngs

            self.Phi_K[:] = 0
            self.Theta_K[:] = 0
            for pk in prange(self.K, schedule='dynamic'):
                thread_idx = omp_get_thread_num()
                for d in xrange(self.D):
                    theta_dk = _sample_gamma(rngs[thread_idx], shp, sca)
                    self.Theta_DK[d, pk] = theta_dk
                    self.Theta_K[pk] += theta_dk

                for v in xrange(self.V):
                    phi_kv = _sample_gamma(rngs[thread_idx], shp, sca)
                    self.Phi_KV[pk, v] = phi_kv
                    self.Phi_K[pk] += phi_kv

            for pd in prange(self.D, schedule='dynamic'):
                thread_idx = omp_get_thread_num()
                for v in xrange(self.V):
                    p_dv = self.priv_DV[pd, v]
                    mu_dv = p_dv / (1 - p_dv)
                    self.Lambda_2DV[0, pd, v] = gsl_ran_exponential(rngs[thread_idx], mu_dv)
                    self.Lambda_2DV[1, pd, v] = gsl_ran_exponential(rngs[thread_idx], mu_dv)

    cdef void _generate_state(self):
        """
        Initialize internal state randomly to generate true data for test

        Specifically, this initializes
        - Theta_DK and Phi_KV to random samples from Gamma(a, 1/b)
        - Theta_K and Phi_K to sums over all D and V of the above matrices
        - mu_DV to the Poisson parameters from the product of Theta_DK and
          Phi_KV
        - Y_KV, Y_DK, and Y_DV to contain sums over Poisson samples taken
          from each d, k, v.
        - priv_DV to contain a uniform random variable between 0 and 1 for
          each d, v.
        """
        cdef:
            Py_ssize_t pk, pd, d, v, k, thread_idx
            int y_dvk
            double shp, sca
            double theta_dk, phi_kv, mu_dvk
            int[:,:,::1] Y_PKV
            gsl_rng ** rngs

        Y_PKV = np.zeros((self.n_threads, self.K, self.V), dtype=np.int32)
        with nogil:
            shp = self.a
            sca = 1. / self.b
            rngs = self.rngs

            self.Phi_K[:] = 0
            self.Theta_K[:] = 0
            for pk in prange(self.K, schedule='dynamic'):
                thread_idx = omp_get_thread_num()
                for d in xrange(self.D):
                    theta_dk = _sample_gamma(rngs[thread_idx], shp, sca)
                    self.Theta_DK[d, pk] = theta_dk
                    self.Theta_K[pk] += theta_dk

                for v in xrange(self.V):
                    phi_kv = _sample_gamma(rngs[thread_idx], shp, sca)
                    self.Phi_KV[pk, v] = phi_kv
                    self.Phi_K[pk] += phi_kv

            self.Y_KV[:] = 0
            self.Y_DK[:] = 0  
            self.Y_DV[:] = 0
            self.mu_DV[:] = 0

            for pd in prange(self.D, schedule='dynamic'):
                thread_idx = omp_get_thread_num()
                for v in xrange(self.V):
                    for k in xrange(self.K):
                        mu_dvk = self.Theta_DK[pd, k] * self.Phi_KV[k, v]
                        self.mu_DV[pd, v] += mu_dvk
                        
                        # Sampling from all of the Poissons individually
                        # to ensure they are internally consistent.
                        y_dvk = gsl_ran_poisson(rngs[thread_idx], mu_dvk)
                        Y_PKV[thread_idx, k, v] += y_dvk
                        self.Y_DK[pd, k] += y_dvk
                        self.Y_DV[pd, v] += y_dvk

                    self.priv_DV[pd, v] = 0.05

            for p in range(self.n_threads):
                for k in range(self.K):
                    for v in range(self.V):
                        self.Y_KV[k, v] += Y_PKV[p,k,v]

    cdef void _generate_data(self):
        """
        Generate private data given true data and internal state. This
        uses the true data and the privacy variables to generate two-sided
        geometric noise according to Process 2 in the paper, updating
        the Lambda_2DV, G_2DV, and data_DV matrices.
        """
        cdef:
            Py_ssize_t pd, v, thread_idx
            int g_dv1, g_dv2
            double p_dv, mu_dv, lam_dv1, lam_dv2
            gsl_rng ** rngs

        with nogil:
            rngs = self.rngs
            for pd in prange(self.D, schedule='dynamic'):
                thread_idx = omp_get_thread_num()
                for v in xrange(self.V):
                    self.data_DV[pd, v] = self.Y_DV[pd, v]
                    p_dv = self.priv_DV[pd, v]
                    mu_dv = p_dv / (1 - p_dv)

                    self.Lambda_2DV[0, pd, v] = lam_dv1 = gsl_ran_exponential(rngs[thread_idx], mu_dv)
                    self.Lambda_2DV[1, pd, v] = lam_dv2 = gsl_ran_exponential(rngs[thread_idx], mu_dv)

                    self.G_2DV[0, pd, v] = g_dv1 = gsl_ran_poisson(rngs[thread_idx], lam_dv1)
                    self.G_2DV[1, pd, v] = g_dv2 = gsl_ran_poisson(rngs[thread_idx], lam_dv2)

                    self.data_DV[pd, v] += g_dv1 - g_dv2

    cdef void _update_Y_DVK(self):
        """
        Performs one inference step on the true data values expressed in
        Y_KV and Y_DK based on data in Y_DV.
        """
        cdef:
            Py_ssize_t pd, p, v, k, thread_idx
            int y_dv, b_dv, y_dvk
            int [:,:,::1] Y_PKV
            gsl_rng ** rngs

        self.Y_KV[:] = 0
        self.Y_DK[:] = 0
        Y_PKV = np.zeros((self.n_threads, self.K, self.V), dtype=np.int32)

        with nogil:
            rngs = self.rngs
            for pd in prange(self.D, schedule='dynamic'):
                thread_idx = omp_get_thread_num()
                for v in range(self.V):
                    y_dv = self.Y_DV[pd, v]
                    b_dv = self.mask_DV[pd, v]
                    # TODO: missing data?
                    # QUESTION: are the masks being used for missing data, or
                    # for specific cases of ignoring some data in inference?
                    # Why isn't the mask actually being used here?
                    if y_dv == 0:
                        continue

                    for k in range(self.K):
                        self.P_PK[thread_idx, k] = self.Theta_DK[pd, k] * self.Phi_KV[k, v]

                    gsl_ran_multinomial(rngs[thread_idx],
                                        self.K,
                                        <unsigned int> y_dv,
                                        &self.P_PK[thread_idx, 0],
                                        &self.N_PK[thread_idx, 0])

                    for k in range(self.K):
                        y_dvk = self.N_PK[thread_idx, k]
                        if y_dvk > 0:
                            Y_PKV[thread_idx, k, v] += y_dvk
                            self.Y_DK[pd, k] += y_dvk
        
            for p in range(self.n_threads):
                for k in range(self.K):
                    for v in range(self.V):
                        self.Y_KV[k, v] += Y_PKV[p,k,v]

    cdef void _update_Theta_DK(self):
        """Perform an inference step on Theta based on the current
        Phi and Y values."""
        cdef:
            Py_ssize_t k, d
            double a, b
            double sca_k, sca_dk, shp_dk, theta_dk

        a = self.a
        b = self.b

        self.Theta_K[:] = 0
        for k in xrange(self.K):
            sca_k = 1. / (b + self.Phi_K[k])
            for d in xrange(self.D):
                if self.mask_D[d]:
                    sca_dk = sca_k
                else:
                    sca_dk = 1. / (b + np.dot(self.Phi_KV, self.mask_DV[d]))
                shp_dk = a + self.Y_DK[d, k]
                theta_dk = _sample_gamma(self.rng, shp_dk, sca_dk)
                self.Theta_DK[d, k] = theta_dk
                self.Theta_K[k] += theta_dk

    cdef void _update_Phi_KV(self):
        """Perform an inference step on Phi based on the current Theta
        and Y values."""
        cdef:
            Py_ssize_t d, k
            double a, b, sca_k, sca_kv, shp_kv

        a = self.a
        b = self.b

        self.Phi_K[:] = 0
        for k in xrange(self.K):
            sca_k = 1. / (b + self.Theta_K[k])
            for v in xrange(self.V):
                if self.mask_V[v]:
                    sca_kv = sca_k
                else:
                    sca_kv = 1. / (b + np.dot(self.mask_DV[:, v], self.Theta_DK))
                shp_kv = a + self.Y_KV[k, v]
                phi_kv = _sample_gamma(self.rng, shp_kv, sca_kv)
                self.Phi_KV[k, v] = phi_kv
                self.Phi_K[k] += phi_kv

    cdef void _update_Lambda_2DV(self) nogil:
        """Perform an inference step on the lambdas used in generating noise
        based on G."""
        cdef: 
            Py_ssize_t pd, v, n, thread_idx
            double p_dv, shp_dvn
            gsl_rng ** rngs

        rngs = self.rngs
        for pd in prange(self.D, schedule='dynamic'):
            thread_idx = omp_get_thread_num()
            for v in xrange(self.V):
                p_dv = self.priv_DV[pd, v]
                if p_dv == 1.:
                    continue
                for n in xrange(2):
                    shp_dvn = 1. + self.G_2DV[n, pd, v]
                    self.Lambda_2DV[n, pd, v] = _sample_gamma(rngs[thread_idx], shp_dvn, p_dv)

    cdef void _update_G_2DV(self):
        """Perform an inference step on the true data Y_DV and noise G."""
        cdef:
            Py_ssize_t pd, v, thread_idx
            int o_dv, m_dv, y_p_dv, g_dv1, g_dv2, y_dv
            double shp_dv, mu_dv, lam_dv1, lam_dv2, p_dv
            gsl_rng ** rngs

        self.mu_DV = np.dot(self.Theta_DK, self.Phi_KV)
        
        with nogil:
            rngs = self.rngs
            for pd in prange(self.D, schedule='dynamic'):
                for v in range(self.V):
                    if self.priv_DV[pd, v] == 0:
                        continue
                    thread_idx = omp_get_thread_num()
                    o_dv = self.data_DV[pd, v]
                    mu_dv = self.mu_DV[pd, v]
                    lam_dv1 = self.Lambda_2DV[0, pd, v]
                    lam_dv2 = self.Lambda_2DV[1, pd, v]
                    shp_dv = 2 * sqrt((lam_dv1 + mu_dv) * lam_dv2)

                    if self.icm:
                        m_dv = _mode_bessel(fabs(o_dv + 0.), shp_dv)
                    else:
                        m_dv = _sample_bessel(rngs[thread_idx], fabs(o_dv + 0.), shp_dv)
                    # assert m_dv >= 0
                    
                    if o_dv <= 0:
                        y_p_dv = m_dv
                        g_dv2 = y_p_dv - o_dv
                    else:
                        g_dv2 = m_dv
                        y_p_dv = g_dv2 + o_dv

                    p_dv = mu_dv / (mu_dv + lam_dv1)
                    y_dv = gsl_ran_binomial(rngs[thread_idx], p_dv, y_p_dv)
                    g_dv1 = y_p_dv - y_dv

                    self.Y_DV[pd, v] = y_dv
                    self.G_2DV[0, pd, v] = g_dv1
                    self.G_2DV[1, pd, v] = g_dv2