cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng_type:
        pass
    ctypedef struct gsl_rng:
        pass
    gsl_rng_type *gsl_rng_mt19937
    void gsl_rng_free(gsl_rng * r)
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T)
    void gsl_rng_set(gsl_rng * r, unsigned long int)
    unsigned long int gsl_rng_get (const gsl_rng * r)


cdef class MCMCModel:
    cdef:
        gsl_rng *rng
        gsl_rng **rngs
        int total_itns, print_every, num_threads
        dict param_list

    cdef list _get_variables(self)
    cdef void _generate_state(self)
    cdef void _generate_data(self)
    cdef void _init_state(self)
    cdef void _print_state(self)
    cdef void _update(self, int num_itns, int verbose, dict schedule)
    cpdef void update(self, int num_itns, int verbose, dict schedule=?)
    cdef void _test(self,
                    int num_samples,
                    str method=?,
                    dict var_funcs=?,
                    dict schedule=?)
    cdef void _calc_funcs(self, int n, dict var_funcs, dict out)
    cpdef void geweke(self, int num_samples, dict var_funcs=?, dict schedule=?)
    cpdef void single_sample(self, int num_samples, dict var_funcs=?, dict schedule=?)
