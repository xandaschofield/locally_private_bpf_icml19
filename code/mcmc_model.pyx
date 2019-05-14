#!python
#cython: boundscheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: wraparound=False
#distutils: language = c
#distutils: extra_link_args = ['-lgsl', '-lgslcblas']
#distutils: extra_compile_args = -Wno-unused-function
import os
import shutil
import sys
cimport numpy as np
import numpy as np
import scipy.stats as st
from numpy.random import randint
from pp_plot import pp_plot
from copy import deepcopy
from time import time
from contextlib import contextmanager
from libc.stdlib cimport malloc, free


@contextmanager
def timeit_context(name):
    startTime = time()
    yield
    elapsedTime = time() - startTime
    print '%3.4fms: %s' % (elapsedTime * 1000, name)


cdef class MCMCModel:

    def __init__(self, object seed=None, int num_threads=1):

        self.num_threads = num_threads
        self.rngs = <gsl_rng**>malloc(self.num_threads * sizeof(gsl_rng))
        for t in range(self.num_threads):
            self.rngs[t] = gsl_rng_alloc(gsl_rng_mt19937)
        self.rng = self.rngs[0]

        if seed is None:
            seed = randint(0, sys.maxint) & 0xFFFFFFFF
        gsl_rng_set(self.rng, seed)

        for t in range(1, self.num_threads):
            seed_t = gsl_rng_get(self.rng)
            gsl_rng_set(self.rngs[t], seed_t)
        gsl_rng_set(self.rng, seed)  # reset the initial seed

        self.total_itns = 0
        self.print_every = 10
        self.param_list = {'seed': seed}

    def __dealloc__(self):
        """
        Free GSL random number generators.
        """
        for t in range(self.num_threads):
            gsl_rng_free(self.rngs[t])

    def get_total_itns(self):
        """
        Return the number of itns the model has done inference for.
        """
        return self.total_itns

    def get_params(self):
        """
        Get a copy of the initialization params.

        Inheriting objects should add params to the param_list, e.g.:

        cdef class ExampleModel(MCMCModel):
            
            def __init__(self, double alpha=1., object seed=None):
                
                super(ExampleModel, self).__init__(seed)
                
                self.param_list['alpha'] = alpha

                ...
        """
        return deepcopy(self.param_list)

    cdef list _get_variables(self):
        """
        Return variable names, values, and sampling methods for testing.

        Example:

        return [('foo', self.foo, self._sample_foo),
                ('bar', self.bar, self._sample_bar)]
        """
        pass

    def get_state(self):
        """
        Wrapper around _get_variables(...).

        Returns only the names and values of variables (not update funcs).
        """
        for k, v, update_func in self._get_variables():
            if np.isscalar(v):
                yield k, v
            else:
                yield k, np.array(v)

    cdef void _generate_state(self):
        """
        Generate internal state.
        """
        pass

    cdef void _generate_data(self):
        """
        Generate data given internal state.
        """
        pass

    cdef void _init_state(self):
        """
        Initialize internal state.
        """
        pass

    cdef void _print_state(self):
        """
        Print internal state.
        """
        print 'ITERATION %d\n' % self.total_itns

    cdef void _update(self, int num_itns, int verbose, dict schedule):
        """
        Perform inference.
        """

        cdef:
            int n

        for k, _, _ in self._get_variables():
            if k not in schedule.keys():
                schedule[k] = lambda x: True

        for n in range(self.total_itns, self.total_itns + num_itns):
            for k, _, update_func in self._get_variables():
                if schedule[k](n):
                    if (verbose == 1) and ((n + 1) % self.print_every == 0):
                        with timeit_context('sampling %s' % k):
                            update_func(self)
                    else:
                        update_func(self)
            self.total_itns += 1
            if (verbose == 1) and ((n + 1) % self.print_every == 0):
                self._print_state()

    cpdef void update(self, int num_itns, int verbose, dict schedule={}):
        """
        Thin wrapper around _update(...).
        """
        self._update(num_itns, verbose, schedule)

    cdef void _test(self,
                    int num_samples,
                    str method='geweke',
                    dict var_funcs={},
                    dict schedule={}):
        """
        Perform Geweke testing or single-sample testing.
        """

        cdef:
            int n
            dict default_funcs, fwd, rev

        default_funcs = {'Arith. Mean': np.mean,
                         'Geom. Mean': lambda x: np.exp(np.mean(np.log1p(x))),
                         'Var.': np.var,
                         'Max.': np.max}

        fwd, rev = {}, {}
        var_funcs = deepcopy(var_funcs)  # this method changes var_funcs state
        for k, v, _ in self._get_variables():
            
            if k not in schedule.keys():
                schedule[k] = lambda x: True
            
            if not any(schedule[k](n) for n in xrange(num_samples)):
                if k in var_funcs.keys():
                    del var_funcs[k]
                continue

            if k not in var_funcs.keys():
                var_funcs[k] = default_funcs
            assert len(var_funcs[k].keys()) <= 4

            if np.isscalar(v):
                fwd[k] = np.empty(num_samples)
                rev[k] = np.empty(num_samples)
            else:
                fwd[k] = {}
                rev[k] = {}
                for f in var_funcs[k]:
                    fwd[k][f] = np.empty(num_samples)
                    rev[k][f] = np.empty(num_samples)

        if method == 'single_sample':
            for n in range(num_samples):
                self._generate_state()
                self._generate_data()
                self._calc_funcs(n, var_funcs, fwd)

                self._update(10, 0, schedule)
                self._generate_data()
                self._calc_funcs(n, var_funcs, rev)
                if n % 500 == 0:
                    print n
        else:
            for n in range(num_samples):
                self._generate_state()
                self._generate_data()
                self._calc_funcs(n, var_funcs, fwd)
                if n % 500 == 0:
                    print n

            self._generate_state()
            for n in range(num_samples):
                self._generate_data()
                self._update(10, 0, schedule)
                self._calc_funcs(n, var_funcs, rev)
                if n % 500 == 0:
                    print n

        for k, _, _ in self._get_variables():
            if any(schedule[k](n) for n in xrange(num_samples)):
                pp_plot(fwd[k], rev[k], k, show=False)

    cdef void _calc_funcs(self,
                          int n,
                          dict var_funcs,
                          dict out):
        """
        Helper function for _test. Calculates and stores functions of variables.
        """

        for k, v, _ in self._get_variables():
            if k not in var_funcs.keys():
                continue
            if np.isscalar(v):
                out[k][n] = v
            else:
                for f, func in var_funcs[k].iteritems():
                    out[k][f][n] = func(v)

    cpdef void geweke(self,
                      int num_samples,
                      dict var_funcs={},
                      dict schedule={}):
        """
        Wrapper around _test(...).
        """
        self._test(num_samples=num_samples,
                   method='geweke',
                   var_funcs=var_funcs,
                   schedule=schedule)

    cpdef void single_sample(self,
                      int num_samples,
                      dict var_funcs={},
                      dict schedule={}):
        """
        Wrapper around _test(...).
        """
        self._test(num_samples=num_samples,
                   method='single_sample',
                   var_funcs=var_funcs,
                   schedule=schedule)
