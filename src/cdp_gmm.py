import numpy as np
import warnings
from sklearn.utils import check_random_state
from sklearn.exceptions import ConvergenceWarning
from diffprivlib.models import KMeans
from diffprivlib.validation import check_bounds, clip_to_bounds
from diffprivlib.utils import PrivacyLeakWarning

import mock_dp_library as dpl
from base_gmm import GMM


"""
D : original dataset
D' : neighbor dataset of D
d : number of dimensions
n : number of records
x_i : ith record in D
K : number of Gaussian components
T : maximum iterations
B : total privacy budget
R : upper bound L_1-norm of x_i
pi, mu_k, sigma_k : original parameters in each iteration
S(pi), S(mu), S(sigma): L_1-sensitivity of pi, mu_k, sigma_k
pi_bar, mu_bar_k, sigma_bar_k : noisy parameters in each iteration
pi_hat, sigma_hat_k : post-processed output of pi_bar and sigma_bar_k
eps_p, eps_m, eps_s : privacy budget of pi, mu_k, and sigma_k
"""


class CDPGMM(GMM):
    # TODO: add class docstring, reference We et al. and Holohan et al.

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
        self._eps_init = 0.1 * epsilon                             # privacy budget for parameter initialization
        self._eps_p = (0.04 * epsilon) / max_iter                  # privacy budget for weights
        self._eps_m = (0.16 * epsilon) / (max_iter * n_components) # privacy budget for means
        self._eps_s = (0.7 * epsilon) / (max_iter * n_components)  # privacy budget for covariances
    
    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        if epsilon <= 0:
            raise ValueError("invalid epsilon, must be greater than 0")
        self._epsilon = epsilon
        self._eps_p = (0.04 * epsilon) / self.max_iter
        self._eps_m = (0.16 * epsilon) / (self.max_iter * self.n_components)
        self._eps_s = (0.7 * epsilon) / (self.max_iter * self.n_components)

    def _initialize_parameters(self, X, bounds):
        # TODO: add information about DP and bounds parameter in docstring
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)
        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        n_samples, _ = X.shape

        if self.init_params == "kmeans":
            resp = np.zeros((n_samples, self.n_components))
            label = (
                KMeans(
                    n_clusters=self.n_components, epsilon=self._eps_init, bounds=bounds
                )
                .fit(X)
                .labels_
            )
            resp[np.arange(n_samples), label] = 1
        else:
            raise ValueError(
                "Unimplemented initialization method '%s'" % self.init_params
            )
        self._initialize(X, resp)
    
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
        self._X = X
        self._trained = True
        _, n_dims = X.shape
        if not bounds:
            warnings.warn("Bounds have not been specified and will be calculated on the data provided.  This will "
                          "result in additional privacy leakage. To ensure differential privacy and no additional "
                          "privacy leakage, specify `bounds` for each dimension.", PrivacyLeakWarning)
            self.bounds = (np.min(X, axis=0), np.max(X, axis=0))

        self.bounds = check_bounds(self.bounds, n_dims, min_separation=1e-5)
        X = clip_to_bounds(X, self.bounds)
        self.fit_predict(X, y, self.bounds)
        return self

    def fit_predict(self, X, y=None, bounds=None):
        # TODO: add information about DP and bounds parameter in docstring
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_initial_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, bounds)

            lower_bound = -np.inf if do_init else self.lower_bound_

            if self.max_iter == 0:
                best_params = self._get_parameters()
                best_n_iter = 0
            else:
                for n_iter in range(1, self.max_iter + 1):
                    prev_lower_bound = lower_bound

                    log_prob_norm, log_resp = self._e_step(X)
                    self._m_step(X, log_resp)

                    print(self.weights_)
                    print(type(self.weights_))

                    # DP steps here
                    self._noise_step(X)
                    print(self.weights_)
                    self._post_process_step()
                    print(self.weights_)
                    exit()

                    lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)                    

                    change = lower_bound - prev_lower_bound
                    self._print_verbose_msg_iter_end(n_iter, change)

                    if abs(change) < self.tol:
                        self.converged_ = True
                        break

                self._print_verbose_msg_init_end(lower_bound)

                if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                    max_lower_bound = lower_bound
                    best_params = self._get_parameters()
                    best_n_iter = n_iter

        # Should only warn about convergence if max_iter > 0, otherwise
        # the user is assumed to have used 0-iters initialization
        # to get the initial means.
        if not self.converged_ and self.max_iter > 0:
            warnings.warn(
                "Initialization %d did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data." % (init + 1),
                ConvergenceWarning,
            )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X)

        return log_resp.argmax(axis=1)
    
    def _noise_step(self, X):
        # TODO: add docstring
        #   add noise to weights according to Eq. (7)
        #   for k = 1 to K do
        #       add noise to means according to Eq. (8)
        #       add noise to covariances according to Eq. (9)
        n, d = X.shape
        K = self.n_components
        R = np.linalg.norm(X, ord=1, axis=1).max()

        # Calculate sensitivities
        #   weights: S(pi) = K/n
        #   means: S(mu) = 4RK/n
        #   covariances: S(sigma) = (12nKR^2 + 8K^2R^2)/n^2        
        sens_p = K / n
        sens_m = (4 * R * K) / n
        sens_s = (12 * n * K * R**2 + 8 * K**2 * R**2) / (n**2)

        # add noise to weights
        self.weights_ += dpl.laplace(shift=0, scale=(sens_p / self._eps_p), size=self.weights_.shape)
        for k in range(K):
            # add noise to means matrix
            means_noise = dpl.laplace(shift=0, scale=(sens_m / self._eps_m), size=self.means_[k].shape)
            self.means_[k] += means_noise
            # add symmetric noise to covariance matrix
            cov_noise = dpl.laplace(shift=0, scale=(sens_s / self._eps_s), size=self.covariances_[k].shape)
            cov_noise = np.tril(cov_noise) + np.tril(cov_noise, -1).T
            self.covariances_[k] += cov_noise

    def _post_process_step(self):
        # TODO: add docstring
        #   normalize weights
        #   for k = 1 to K do
        #       post-process covariances using Algorithm 1
        # TODO: figure out normalizing weights (currently possible to
        #       get negative weights after noise)
        self.weights_ /= self.weights_.sum()
        for k in range(self.n_components):
            min_eig = np.linalg.eigvals(self.covariances_[k]).min()
            delt = max(0, -1*min_eig)
            self.covariances_[k] += delt * np.identity(n = self.covariances_[k].shape[0])
