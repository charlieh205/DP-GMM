import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.mixture import GaussianMixture

"""
We'll build our DP-GMM implementation off of sklearn's GaussianMixture class.
"""

"""
D : original dataset
D' : neighbor dataset of D
d : number of dimensions
n : number of records
x_i : ith record in D
K : number of Gaussian components
T : maximum iterations
B : total privacy budget
pi, mu_k, sigma_k : original parameters in each iteration
S(pi), S(mu), S(sigma): L_1-sensitivity of pi, mu_k, sigma_k
pi_bar, mu_bar_k, sigma_bar_k : noisy parameters in each iteration
pi_hat, sigma_hat_k : 
"""

class GMM(GaussianMixture):

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
        if not self._trained:
            raise RuntimeError("GMM has not been fitted yet, see `fit` method.")
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
