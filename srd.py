"""
Sum of ranking differences analysis
Validated by comparison of ranks with random numbers (CRNN)

Requirements:
    1) NumPy
    2) SciPy

References:
    1) Heberger, K. Sum of ranking differences compares methods or models fairly. TRAC 2010, 29, 101-109.
        (doi:10.1016/j.trac.2009.09.009)
    2) Heberger, K.; Kollar-Hunek, K. Sum of ranking differences for method discrimination and its validation:
        comparison of ranks with random numbers. J. Chemom. 2011, 25, 151-158.
        (doi:10.1002/cem.1320)
    3) Kalivas, J. H.; Heberger, K.; Andries, E. Sum of ranking differences (SRD) to ensemble multivariate
        calibration model merits for tuning parameter selection and comparing calibration methods, Anal. Chim. Acta
        2015, 869, 21-33.
        (doi:10.1016/j.aca.2014.12.056)
"""

# Import stuff from numpy, scipy & itertools packages
from numpy import min as np_min
from numpy import max as np_max
from numpy import mean, median, size, ndarray, argsort, argwhere, arange, multiply, zeros, array, hstack
from scipy.stats import mode
from numpy.random import permutation
from itertools import permutations as perms


class SumOfRankingDiffs:

    def __init__(self, a, t='mean'):

        """
        :param a: ndarray
        :param t: ndarray, string (default, 'mean')
        """

        # Define input matrix
        self.A = a  # Matrix A [columns: models, methods; rows: samples]

        """ Define targets """
        # Dictionary of "gold standards"
        self.gold_std_dict = {'mean': mean, 'median': median, 'mode': mode, 'min': np_min, 'max': np_max}

        # If target is a string, apply the function/attribute from the dictionary; else: use the input t as is
        if isinstance(t, str):
            self.T = self.gold_std_dict[t](a, axis=0)  # Target T [default: mean of A]
        else:
            assert isinstance(t, ndarray), '# If target is not a string or None, it must be an ndarray !'
            self.T = t

        # Define size of A
        self.nrows, self.ncols = size(self.A, axis=0), size(self.A, axis=1)

        # Define srd (raw), srd (normalized), maximum srd, srd (random), srd (random, normalized)
        self.srd, self.srd_norm, self.srd_max, self.srd_rnd, self.srd_rnd_norm = [ndarray([])] * 5

    def compute_srd(self):

        # Define T & A indices, and initialize the rank of A
        t_index, a_index, a_ranking = argsort(self.T), argsort(self.A, axis=0), zeros((self.nrows, self.ncols))

        for i in range(self.nrows):
            row = argwhere(a_index == t_index[i])
            a_ranking[i, :] = row[:, 0].T

        ideal_rank = arange(self.nrows).reshape(-1, 1)
        self.srd = sum(abs(a_ranking - ideal_rank))

        return self

    def _srd_max(self):

        if self.nrows % 2 == 1:

            k = (self.nrows - 1) / 2
            self.srd_max = 2 * k * (k + 1)

        else:

            k = self.nrows / 2
            self.srd_max = 2 * (k ** 2)

        return self

    def srd_normalize(self):

        # Assertion to make sure that SRD is ran before normalization !
        assert self.srd.size > 0, '# You must run the SRD method before normalization !'

        # Normalization
        self.srd_norm = self.srd * (100 / self._srd_max().srd_max)

        return self

    @staticmethod
    def _srd_val_normalize(srd_vals, srd_max):
        """
        :param srd_vals: ndarray
        :param srd_max: int
        :return: ndarray
        """
        return multiply(srd_vals, 100 / srd_max)

    @staticmethod
    def _srd_val_restrict(nrows, exact, exact_lim=10, n_rnd_vals=10000):

        """
        :param nrows: int
        :param exact: bool
        :param exact_lim: int
        :param n_rnd_vals: int
        :return: exact, exact_lim, n_rnd_vals
        """

        # Restrict the calculation of true SRD distribution if nrows > exact_lim (default 10)
        exact = False if nrows > exact_lim else exact
        exact_lim, n_rnd_vals = (int(exact_lim), 0) if exact else (0, int(n_rnd_vals))

        return exact, exact_lim, n_rnd_vals

    # SRD validation using the distribution of SRD values of normally-distributed random numbers
    def srd_validate(self, exact=False, **kwargs):

        """
        :param exact: bool
        :return: self
        """

        # Assertion to make sure that SRD is ran before validation !
        assert self.srd.size > 0, '# You must run the SRD method before validation !'
        # Assertion to make sure that exact is a boolean !
        assert isinstance(exact, bool), '# The argument \"exact\" has to be a boolean !'  # Bug fix

        # Input arguments
        exact, exact_lim, n_rnd_vals = self._srd_val_restrict(self.nrows, exact, **kwargs)

        # Ideal ranking
        ideal_rank = arange(self.nrows).reshape(-1, 1)

        # Compute SRD values for "n_rand_vals" random numbers
        rnd_order = hstack([permutation(self.nrows).reshape(-1, 1) for _ in range(n_rnd_vals)]) if not exact else \
            array(list(perms(range(self.nrows)))).T
        srd_rnd = sum(abs(rnd_order - ideal_rank))
        srd_rnd_norm = self._srd_val_normalize(srd_vals=srd_rnd, srd_max=self._srd_max().srd_max)

        self.srd_rnd, self.srd_rnd_norm = srd_rnd, srd_rnd_norm

        return self
