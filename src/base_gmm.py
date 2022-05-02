import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.mixture import GaussianMixture

"""
We'll build our DP-GMM implementation off of sklearn's GaussianMixture class.
"""

class GMM(GaussianMixture):

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
        self._trained = False

    def fit(self, X, y=None):
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
        self.fit_predict(X, y)
        return self

    def plot_mixture(self, bins=10):
        """Plot the mixture model on top of the data.

        The method is only designed for datasets of single dimension. A
        higher dimension will raise an error. GMM must have already been
        fit on the dataset.

        Parameters
        ----------
        bins : int, optional
            number of equal-width bins, by default 10

        Raises
        ------
        RuntimeError
            the GMM has not been fit to any data
        ValueError
            the dataset has a dimention higher than 1
        """
        if not self._trained:
            raise RuntimeError("GMM has not been fit to any data, see `fit` method.")
        if self._X.shape[-1] > 1:
            raise ValueError("too many dimensions to plot (expected 1)")
        _, bin_vals, _ = plt.hist(self._X, bins=bins, density=True, color="gray", alpha=0.5)
        xvals = np.linspace(bin_vals[0], bin_vals[-1], 100)
        for i in range(self.n_components):
            pi = self.weights_[i]
            pdf = sp.stats.norm(self.means_.flatten()[i], self.covariances_.flatten()[i]**0.5).pdf(xvals)
            plt.plot(xvals, pi*pdf, label=f"Component {i + 1}")
        plt.legend()
        plt.show()
