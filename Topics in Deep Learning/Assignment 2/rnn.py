# -*- coding: utf-8 -*-
"""RNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fuBj774OqJ7KNR7cGQ0f9Fs6-EH0G587
"""

import torch

class recurrent_layer(torch.nn.Module):
    def __init__(self, in_features, out_features, nonlinearity="tanh"):
        super().__init__()
        self.reset_parameters(in_features, out_features, nonlinearity)

    def forward(self, input, h):
        output = torch.mm(input, self.weight_ih) + torch.mm(h, self.weight_hh)
        if(self.nonlinearity == "tanh"):
            return torch.tanh(output)
        elif(self.nonlinearity == "relu"):
            return torch.relu(output)
        else:
            raise RuntimeError("Unknown nonlinearity: {}".format(self.nonlinearity))

    def reset_parameters(self, in_features, out_features, nonlinearity="tanh"):
        self.in_features = in_features
        self.out_features = out_features
        self.weight_ih = torch.nn.Parameter(torch.randn(in_features, out_features))
        self.weight_hh = torch.nn.Parameter(torch.randn(out_features, out_features))
        self.nonlinearity = nonlinearity

      
# rnn = recurrent_layer(10, 20)
# input = torch.randn(6, 3, 10)
# hx = torch.randn(3, 20)
# output = []
# for i in range(6):
#  hx = rnn(input[i], hx)
#  output.append(hx)

# print(output)