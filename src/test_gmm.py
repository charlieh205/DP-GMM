import numpy as np
import pandas as pd
from base_gmm import GMM
from cdp_gmm import CDPGMM

test_dat = np.random.randint(1, 10, size=(20, 2))
gmm1 = GMM(2)
try:
    gmm1.plot_mixture()
except RuntimeError as e:
    print(e)
gmm1.fit(test_dat)
try:
    gmm1.plot_mixture()
except ValueError as e:
    print(e)

faithful = pd.read_csv("data/old_faithful.csv")
dat = faithful["waiting"].values.astype(int).reshape(-1, 1)
gmm = GMM(2).fit(dat)
gmm.plot_mixture(60)

cdpgmm = CDPGMM(2).fit(dat)
cdpgmm.plot_mixture(60)