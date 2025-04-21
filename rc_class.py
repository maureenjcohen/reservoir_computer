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

# %%
class NextGenResComp(nn.Module):
    """
    Next-generation reservoir computer, also known as non-linear 
    vector autoregression (NVAR) machine

    nn.Linear(len_feature_vector, len_data) 
    """

    def __init__(self, len_x, len_o):
        """
        Initialise parameters of NextGenResComp

        Args: 
            len_x: torch.tensor
            Length of input data (number of coupled time series)

            len_o: torch.tensor
            Length of feature vector

        Returns:
            Nothing
        """
        super(NextGenResComp, self).__init__()

        # Single layer of trainable weights maps feature vector to output
        self.w = nn.Linear(len_o, len_x)

    
    def forward(self, x):
        """
        Forward pass of NextGenResComp

        Args:
            x:  torch.tensor
            Input features

        Returns:
            output: torch.tensor
            Output/prediction at next time coordinate
        """

        output = self.w(x)

        return output