"""
Play with a small PINN
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class MLP_PINN(nn.Module):
  """MLP with constant number of units per hidden layer"""

  def __init__(self, num_hidden_layers, num_hidden_units, din, dout):
    super(MLP_PINN, self).__init__()
    self.linears = nn.ModuleList(
        [nn.Linear(din, num_hidden_units)] + 
        [nn.Linear(num_hidden_units, num_hidden_units) 
          for i in range(num_hidden_layers-1)] +
        [nn.Linear(num_hidden_units, dout)])

  def forward(self, x):
    for i, lin in enumerate(self.linears[:-1]):
      x = F.silu(lin(x))
    return self.linears[-1](x)

def main():
  din = 1                 # one spatial variable (x)
  dout = 1                # one output variable (T)
  num_hidden_layers = 5
  num_hidden_units = 20
  batch_size = 128
  mlp = MLP_PINN(num_hidden_layers, num_hidden_units, din, dout)

  test_x = torch.zeros([batch_size, 1])
  test_out = mlp(test_x)


  pdb.set_trace()


if __name__ == "__main__":
  main()
