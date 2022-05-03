import numpy as np
import warnings
from diffprivlib.validation import check_bounds, clip_to_bounds
from diffprivlib.utils import PrivacyLeakWarning

from base_gmm import GMM
import local_dist as ld

class LDPGMM(GMM):

    def __init__(
        self,
        n_components=1,
        *,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
        epsilon=1
    ):
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval
        )
        self._epsilon = epsilon
        
    @property
    def epsilon(self):
        return self._epsilon
        
    @epsilon.setter
    def epsilon(self, epsilon):
        if epsilon <= 0:
            raise ValueError("ValueError: invalid epsilon, must be greater than 0")
        self._epsilon = epsilon
    
    def fit(self, X, y=None, bounds=None, boots=2000):
        # TODO: add information about DP and bounds parameter in docstring
        """Estimate model parameters with the EM algorithm.
        The method fits the model ``n_init`` times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for ``max_iter``
        times until the change of likelihood or lower bound is less than
        ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
        If ``warm_start`` is ``True``, then ``n_init`` is ignored and a single
        initialization is performed upon the first call. Upon consecutive
        calls, training starts where it left off.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            The fitted mixture.
        """
        self._trained = True
        n, d = X.shape
        if not bounds:
            warnings.warn("Bounds have not been specified and will be calculated on the data provided.  This will "
                          "result in additional privacy leakage. To ensure differential privacy and no additional "
                          "privacy leakage, specify `bounds` for each dimension.", PrivacyLeakWarning)
            self.bounds = (np.min(X, axis=0), np.max(X, axis=0))

        self.bounds = check_bounds(self.bounds, d, min_separation=1e-5)
        X = clip_to_bounds(X, self.bounds)

        # call local histogram function to create proportions
        bins = list(np.unique(X))
        corrected, true = ld.convert_to_hist(n, self.bounds, X, self._epsilon, bins, boots)

        # get DP X values based off proportions
        dp_list = np.array(ld.make_dist(corrected, X)).reshape(-1, 1)

        # fit and get predictions
        self.fit_predict(dp_list, y)
        return self
