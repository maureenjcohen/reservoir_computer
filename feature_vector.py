# %%
import numpy as np
import pandas as pd
import torch
from torch import nn

# %%
class FeatureVector:
    """
    Input to the reservoir computer
    Constructed from the time series data
    """

    def __init__(self, data, k, s=1, c=1, order=2):
        """
        Generate FeatureVector from time series data

        Args: 
        -----
        data: 2D numpy array
            Time series data
        k: int
            How many previous time samples to include
        s: int
            Number of skips between consecutive samples. Default is 1.
        c: int
            Constant term. Default is 1.
        order: int
            Order of polynomial expansion. Default is 2

        Returns:
        -------
        feature_vector: 1D numpy array
            Constructed feature vector
        """
        linear_comp = self.construct_linear(data=data, k=k, s=s)
        nonlin_comp = self.construct_nonlinear(data=data, k=k, s=s, order=order)

        feature_vector = np.concatenate([np.array([c]), linear_comp, nonlin_comp])
        self.feature_vector = feature_vector
    
    def get_data_dims(self, data):
        """
        Get the dimensions of the data

        Returns:
        -------
        num_x: int
            Number of coupled time series
        len_t: int
            Length of time series data (number of times)
        """
        num_x = data.shape[0]
        len_t = data.shape[1]

        self.num_x = num_x
        self.len_t = len_t

        return num_x, len_t

    def construct_linear(self, data, k, s):
        """
        Construct linear component of feature vector

        Returns:
        -------
        linear_comp: 1D numpy array
            Linear component of feature vector
        """
        __ , len_t = self.get_data_dims(data=data)
        assert k <= len_t, "k must be less than or equal to the length of the time series data"
        assert s > 0, "s must be greater than 0"
        assert k > s, "k must be greater than s"

        linear_comp = data[:,:-k-1:-s].flatten('F')
        return linear_comp

    def construct_nonlinear(self, data, k, s, order):
        """
        Construct non-linear component of feature vector

        Returns:
        -----
        nonlin_comp: 1D numpy array
            Non-linear component of feature vector
        """
        linear_comp = self.construct_linear(data=data, k=k, s=s)
        nonlin_comp = linear_comp
        for i in range(0, order+1):
            nonlin_comp = np.unique(np.outer(nonlin_comp, linear_comp))
        return nonlin_comp
    

if __name__ == "__main__":

    test = np.array([[1,3,8,4],[15,12,13,11]])
    # Create test array containing two time series

    f = FeatureVector(test, k=3)
    # Create feature vector

    f.feature_vector
    # Print feature vector

    # Output:
    # array([    1,     4,    11,     8,    13,     3,    12,    81,   108,
    #      144,   192,   216,   256,   288,   297,   324,   351,   384,
    #      396,   432,   468,   512,   528,   576,   624,   704,   768,
    #      792,   832,   864,   936,  1024,  1056,  1089,  1152,  1188,
    #     1248,  1287,  1296,  1404,  1408,  1452,  1521,  1536,  1584,
    #     1664,  1716,  1728,  1872,  1936,  2028,  2048,  2112,  2288,
    #     2304,  2496,  2704,  2816,  2904,  3072,  3168,  3328,  3432,
    #     3456,  3744,  3872,  3993,  4056,  4096,  4224,  4356,  4576,
    #     4608,  4719,  4752,  4992,  5148,  5184,  5324,  5408,  5577,
    #     5616,  5632,  5808,  6084,  6144,  6292,  6336,  6591,  6656,
    #     6864,  6912,  7436,  7488,  7744,  8112,  8448,  8788,  9152,
    #     9216,  9984, 10648, 10816, 11616, 12584, 12672, 13728, 13824,
    #    14641, 14872, 14976, 15972, 16224, 17303, 17424, 17576, 18876,
    #    19008, 20449, 20592, 20736, 22308, 22464, 24167, 24336, 26364,
    #    28561])