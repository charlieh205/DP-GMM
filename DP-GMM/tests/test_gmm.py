import numpy as np
import pandas as pd
from src import GMM

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

faithful = pd.read_csv("../data/faithful.csv")
dat = faithful["waiting"].values.reshape(-1, 1)
gmm = GMM(2).fit(dat)
gmm.plot_mixture(60)
