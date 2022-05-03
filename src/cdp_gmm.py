import numpy as np
import warnings
from diffprivlib.models import KMeans
from diffprivlib.validation import check_bounds, clip_to_bounds
from diffprivlib.utils import PrivacyLeakWarning

import mock_dp_library as dpl
from base_gmm import GMM


class CDPGMM(GMM):

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
        # allocate privacy budget
        self._eps_init = 0.1 * epsilon                # privacy budget for parameter initialization
        self._eps_p = 0.04 * epsilon                  # privacy budget for weights
        self._eps_m = (0.16 * epsilon) / n_components # privacy budget for means
        self._eps_s = (0.7 * epsilon) / n_components  # privacy budget for covariances

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        if epsilon <= 0:
            raise ValueError("invalid epsilon, must be greater than 0")
        self._epsilon = epsilon
        self._eps_init = 0.1 * epsilon                     # privacy budget for parameter initialization
        self._eps_p = 0.04 * epsilon                       # privacy budget for weights
        self._eps_m = (0.16 * epsilon) / self.n_components # privacy budget for means
        self._eps_s = (0.7 * epsilon) / self.n_components  # privacy budget for covariances
    
    def fit(self, X, y=None, bounds=None):
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
        _, n_dims = X.shape
        if not bounds:
            warnings.warn("Bounds have not been specified and will be calculated on the data provided.  This will "
                          "result in additional privacy leakage. To ensure differential privacy and no additional "
                          "privacy leakage, specify `bounds` for each dimension.", PrivacyLeakWarning)
            self.bounds = (np.min(X, axis=0), np.max(X, axis=0))

        self.bounds = check_bounds(self.bounds, n_dims, min_separation=1e-5)
        X = clip_to_bounds(X, self.bounds)
        self.fit_predict(X, y)
        return self

    def fit_predict(self, X, y=None):
        n, d = X.shape
        if d > 1:
            raise ValueError(f"Unimplemented for dimension greater than 1")
        # STEP 1: cluster data in DP manner
        if self.init_params != "kmeans":
            raise ValueError(f"Unimplemented initialization method '{self.init_params}'")
        labels = KMeans(n_clusters=self.n_components, epsilon=self._eps_init, bounds=self.bounds).fit(X).labels_
        cluster_dat = []
        for i in range(self.n_components):
            cluster_dat.append(X[labels == i])

        # STEP 2: calculate component weights, add noise, and normalize
        self.weights_ = np.zeros((self.n_components,), dtype=float)
        for i in range(self.n_components):
            self.weights_[i] = cluster_dat[i].shape[0] / n
        weight_sens = 2 / n
        self.weights_ += dpl.laplace(shift=0, scale=(weight_sens / self._eps_p), size=self.weights_.shape)
        self.weights_ /= self.weights_.sum()

        # STEP 3: calculate cluster means and [co]variances
        self.means_ = np.zeros((self.n_components,), dtype=float)
        self.covariances_ = np.zeros((self.n_components,), dtype=float)
        for i in range(self.n_components):
            self.means_[i] = cluster_dat[i].mean()
            self.covariances_[i] = cluster_dat[i].std()**2
        
        # STEP 4: add noise to means and [co]variances
        mean_sens = (self.bounds[1] - self.bounds[0]) / n
        cov_sens = ((n - 1) * (self.bounds[1] - self.bounds[0])**2) / (n**2)
        self.means_ += dpl.laplace(shift=0, scale=(mean_sens / self._eps_m), size=self.means_.shape)
        self.covariances_ += dpl.laplace(shift=0, scale=(cov_sens / self._eps_s), size=self.covariances_.shape)
