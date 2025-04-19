# %%
import numpy as np
import pandas as pd
import torch
from torch.nn import nn


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