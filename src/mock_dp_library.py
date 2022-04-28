import numpy as np


def laplace(shift=0, scale=1, size=None):
    """Draw samples from the Laplace or double exponential distribution with
    specified location (or mean) and scale (decay).
    
    Parameters
    ----------
    shift : float or array_like of floats, optional
        The position, :math:`\mu`, of the distribution peak. Default is 0.
    scale : float or array_like of floats, optional
        :math:`\lambda`, the exponential decay. Default is 1. Must be non-
        negative.
    size : int or tuple of ints, optional
        Output shape. If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn. If size is ``None`` (default),
        a single value is returned if ``shift` and ``scale`` are both scalars.
        Otherwise, ``np.broadcast(shift, scale).size`` samples are drawn.
    
    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized Laplace distribution.
    """
    return np.random.laplace(loc=shift, scale=scale, size=size)


def gaussian(shift=0.0, scale=1.0, size=None):
    """Draw random samples from a normal (Gaussian) distribution.
    
    Parameters
    ----------
    loc : float or array_like of floats
        Mean ("centre") of the distribution.
    scale : float or array_like of floats
        Standard deviation (spread or "width") of the distribution. Must be
        non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``loc`` and ``scale`` are both scalars.
        Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized normal distribution.
    """
    return np.random.normal(loc=shift, scale=scale, size=size)


def clamp(x, bounds):
    """Clamp (limit) the values in an array.

    Given an interval, values outside the interval are clamped to
    the interval edges. For example, if an interval of ``[0, 1]``
    is specified, values smaller than 0 become 0, and valeus larger
    then 1 become 1.

    Parameters
    ----------
    x : array_like
        Array containing elements to clip
    bounds : array_like
        Minimum and maximum value.

    Returns
    -------
    out : ndarray
        An array with the elements of `x`, but where values
        < `min(bounds)` are replaced with `min(bounds)`, and those
        > `max(bounds)` are replaced with `max(bounds)`.
    """
    return np.clip(x, *bounds)


def bounded_mean(x, bounds):
    """Returns the average of the array elements after the array
    has been clamped to the given bounds.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose clamped mean is desired.
    bounds : array_like
        Minimum and maximum values for clamping.

    Returns
    -------
    out : float64
        Average of the array elements after clamping.
    """
    x_clamped = clamp(x, bounds)
    return np.mean(x_clamped)


# TODO: convert docstrings to numpy format
def release_dp_mean(x, bounds, epsilon, delta=1e-6, mechanism="laplace"):
    """Release a DP mean. 
    Assumes that the dataset size n is public information.
    """
    sensitive_mean = bounded_mean(x, bounds)

    n = len(x)
    lower, upper = bounds
    # Sensitivity in terms of an absolute distance metric
    # Both the laplace and gaussian mechanisms can noise queries
    #    with sensitivities expressed in absolute distances
    sensitivity = (upper - lower) / n
    
    if mechanism == "laplace":
        scale = sensitivity / epsilon
        dp_mean = sensitive_mean + laplace(scale=scale)
    elif mechanism == "gaussian":
        scale = (sensitivity / epsilon) * np.sqrt(2*np.log(2/delta)) 
        dp_mean = sensitive_mean + gaussian(scale=scale)
    else:
        raise ValueError(f"unrecognized mechanism: {mechanism}")

    return dp_mean


def bootstrap(x, n):
    """Sample n values with replacement from n."""
    index = np.random.randint(low=0., high=len(x), size=n)
    return x[index]


def central_histogram_release(x, epsilon, categories):
    """Release a histogram using Central-DP.
    
    Args:
        x (numpy.ndarray): raw data
        epsilon (int or float): privacy parameter
        categories (list[int]): data categories/bins
    
    Returns:
        numpy.ndarray: locally DP histogram
    """
    sensitivity = 2
    scale = sensitivity / epsilon

    # create a {category: count} hashmap
    counts = dict(zip(*np.unique(x, return_counts=True)))
    # look up the count of each category, or zero if not exists
    sensitive_histogram = np.array([counts.get(cat, 0) for cat in categories])

    dp_histogram = sensitive_histogram + laplace(scale=scale, size=sensitive_histogram.shape)
    return dp_histogram


def local_release(x, epsilon):
    """Create a local release by generating a randomized
    response for the input data.
    
    Args:
        x (numpy.ndarray): raw data
        epsilon (int or float): privacy parameter
    
    Returns:
        numpy.ndarray: local release of x
    """
    # convert to array if not already one
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    
    # get random draw
    draw = np.random.uniform(size=x.shape)
    
    # flip elements of x based on draw and epsilon
    cutoff = 1/(1 + np.exp(epsilon))
    x[draw < cutoff] *= -1
    return x


def expectation_correction(release, epsilon):
    """Correct the expected value of the DP release by
    scaling it by c_epsilon.
    
    Args:
        release (numpy.ndarray): released counts
        epsilon (int or float): privacy parameter
    
    Returns:
        float: corrected expected value
    """
    # calculate c_epsilon
    inflation = (np.exp(epsilon/2) + 1)/(np.exp(epsilon/2) - 1)
    
    # scale expectation
    correct = (release*inflation).mean()
    return correct


def local_histogram_release(x, bounds, epsilon):
    """Release a histogram using Local-DP.
    
    Args:
        x (numpy.ndarray): raw data
        bounds (tup or list[int]): bounds on data
        epsilon (int or float): privacy parameter
    
    Returns:
        numpy.ndarray: locally DP histogram
    """
    # convert to array and clamp
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    x_clamped = clamp(x, bounds)
    
    # create bins
    lower, upper = bounds
    bins = list(range(lower, upper + 1))
    
    # create LDP release
    sensitivity = 2
    dp_release = np.zeros((len(x), upper - lower + 1))
    for i in range(len(x)):
        for j in range(len(bins)):
            sensitive_val = (x_clamped[i] == bins[j])*2 - 1
            dp_release[i, j] = local_release(sensitive_val, epsilon/sensitivity)
    return dp_release


def encode(x, bounds):
    """Convert data array of integers into one-hot
    encoding representation of the data.
    
    Args:
        x (numpy.ndarray): data array
        bounds (tup[int]): data bounds
    
    Returns:
        numpy.ndarray: one-hot encoded data
                       representation
    """
    lower, upper = bounds
    encoding = np.zeros((x.size, upper - lower + 1))
    encoding[np.arange(x.size), x - 1] = 1
    return encoding
