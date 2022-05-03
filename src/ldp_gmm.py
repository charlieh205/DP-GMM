import numpy as np
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
      
    def fit_ldp(self, X, y = None, boots = 2000):
        
        # Define Local parameters based off data
        n = len(X)
        # TODO: add bounds warning
        bounds = (min(X), max(X))
        bins = list(np.unique(X))
        
        # Call local histogram function to create value of proportions
        corrected, true = ld.convert_to_hist(n,bounds,X,self._epsilon,bins,boots)
        
        # Get DP X values based off proportions
        dp_list = np.array(ld.make_dist(corrected, X))
        
        # Fit and get predictions
        self.fit(dp_list.reshape(-1,1))
        
        return self
