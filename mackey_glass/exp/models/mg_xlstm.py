import torch
from torch import nn


'''
Scalar xLSTM w/ expo gating and stablizier
State: (h, c, n, m)
'''

class sLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

    def forward():
        pass


class mLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

    def forward():
        pass

'''
Matrix LSTM (matrix memory) + 
covariance update

    State: (h, C, n, m) where
    C: matrix memory d×d, n: normalizer vector, m: stabilizer state

'''


class xLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

    def forward():
        pass