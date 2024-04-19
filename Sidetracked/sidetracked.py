import yfinance as yf # use yfinance to pull the stock information for these two companies
from torchesn.nn import ESN # the esn stuff
from torchesn.utils import prepare_target
import matplotlib.pyplot as plt
import torch
import numpy as np
import time

class stockESN:
    def __init__(self, num_input_series=1, washout=7, hidden_size=100, output_size=1):
        self.set_dtype()
        self.input_size = num_input_series # just 1 time series
        self.washout = [washout] # washout matching hidden_size, I'm a little dumb washout was not what I thought it was
        self.hidden_size = hidden_size # tweak to my preference
        self.output_size = output_size # only returning a single datapoint
        self.model = ESN(self.input_size, self.hidden_size, self.output_size)
    
    def set_dtype(self, dytpe=torch.float64):
        self.dtype = torch.float64
        torch.set_default_dtype(self.dtype)
        
    def fit(self,trX, trY_flat, washout, loss_function):
        
        """
        Fits the model on training data, prints training error and training time.

        Parameters:
        trX (pyTorch tensor): X training data of shape (n, 1, 1).
        trY_flat (pyTorch tensor): Y training data prepared with prepare_target(), of shape (n,1,1).
        washout (array): Integer array with 1 element, the washout window size

        Returns: output, hidden
        return_type: tensor, tensor
        """
        
        # monitoring training time
        start_time = time.process_time()
        
        self.model(trX, washout, None, trY_flat)
        self.model.fit()
        
        end_time = time.process_time()
        cpu_time = end_time - start_time
        
        print(f"Training completed in {cpu_time} seconds")
        
        # printing training error
        output, hidden = model(trX, washout)
        print("Training error:", loss_function(output, trY[washout[0]:]).item())
        return output, hidden
        
    def predict(x, washout, hidden):
         """
        Predicts, not much else to say.

        Parameters:
        x (pyTorch tensor): X prediction data of shape (n, 1, 1).
        washout (array): Integer array with 1 element, the washout window size
        hidden (pyTorch tensor): hidden layer weight tensor from trained model

        Returns: output, hidden
        return_type: tensor, tensor
        """
        predict_output, predict_hidden = self.model(x, washout, hidden)
        return predict_output, predict_hidden