import numpy as np
import pandas as pd
import warnings
from base_gmm import GMM
from sklearn.utils import check_random_state
from sklearn.exceptions import ConvergenceWarning
import mock_dp_library as dpl

# Local release function used in class
def local_release(x, epsilon):
    x = np.array(x)

    draw = np.random.uniform(size=x.shape)
    cutoff =   1/(1+np.exp(epsilon))
    x[draw<cutoff] *= -1

    return x

# Define function to release local histogram
def local_histogram_release(x, bounds, epsilon, bins):
    
    lower, upper = bounds
    
    x_clamped = dpl.clamp(x, bounds)
    sensitivity = 2

    sensitive_value = []
    dp_release = np.zeros((len(x), len(bins)))

    for i in range(len(x)):
        current_x = x_clamped[i] 

        for j in range(len(np.unique(x))):
            sensitive_value = (current_x == bins[j]) * 2 - 1
            dp_release[i, j] = local_release(sensitive_value, epsilon=epsilon/sensitivity) 
    
    return dp_release



# Define function to make local histogram release from 
def convert_to_hist(n, bounds, data, epsilon, bins, boots, plot = False):
    
    # Bootstrap the data
    #boot_data = dpl.bootstrap(data, n=boots)
    
    #Create local histogram of proportions
    dp_hist = local_histogram_release(x=data, bounds=bounds, epsilon=epsilon, bins=bins) 
    
    raw_dp_values = dp_hist.mean(axis=0) 
    c = (np.exp(epsilon/2) + 1)/(np.exp(epsilon/2) - 1)
    corrected = c * raw_dp_values 
    
    true_values = np.unique(data, return_counts=True)[1] / len(data)
    
    if plot == True:
        pd.DataFrame({
            "true_values": true_values,  
            "corrected_dp_values": (corrected + 1) / 2
        }).plot(kind="bar")
        print(c)
    
    return corrected, true_values


# Create function to make distribution from proportion histograms
def make_dist(corrected_props, data):
    
    # Convert proportions to values
    vals = []
    for val in corrected_props:
        if val<0:
            vals.append((val+1)*len(data)/2)
        else:
            vals.append(0)
    #vals = (corrected_props+1)*len(data)/2
    vals = np.array(vals)
    
    # Clip to 0 and the max to get rid of negative values
    vals = np.clip(vals, 0, max(vals))
    
    #Convert values to integers
    vals = np.around(vals).astype(int)
    
    # Get bins from data
    values = np.unique(data)
    
    #Loop through values and append value for correct iterations
    dp_list = []
    for i in range(len(values)):
        for j in range(vals[i]):
            dp_list.append(values[i])
            
    return dp_list