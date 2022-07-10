"""
Play with a small PINN for the 1D heat equation
with constant heat flux and temperature boundary conditions
(solution is a parabola)
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE = 1024
LR_INIT = 1E-4

QDOT = 100
K = 1
T0 = 5
T1 = 10

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

def train(net, opt, c1=100, batch_size=64, niter=10000):
  losses = [None]*niter
  for it in range(niter):
    opt.zero_grad()
    test_x = torch.rand([batch_size, 1])
    internal_loss = pde_loss(net, test_x, QDOT, K)/batch_size
    bound_loss = bc_loss(net, T0, T1)
    total_loss = internal_loss + c1*bound_loss
    total_loss.backward()
    opt.step()
    losses[it] = total_loss.item() 
    print("It. {} loss: {:.7f}".format(it, total_loss.item()))
  return losses

def pde_loss(net, x, qdot, k):
  x.requires_grad = True
  T = net(x)
  dT_dx = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T),
                              retain_graph=True, create_graph=True)[0]
  dT2_dx2 = torch.autograd.grad(dT_dx, x, grad_outputs=torch.ones_like(dT_dx),
                              retain_graph=True, create_graph=True)[0]
  loss = dT2_dx2 + qdot / k
  return torch.sum(loss**2)

def bc_loss(net, T0, T1):
  x0 = torch.tensor([0.0])
  x1 = torch.tensor([1.0])
  T0_pred = net(x0)
  T1_pred = net(x1)
  loss0 = (T0 - T0_pred)**2
  loss1 = (T1 - T1_pred)**2
  return loss0 + loss1

def plot_results(losses, mlp):
  fig, ax = plt.subplots(2)
  ax[0].semilogy(losses)
  ax[0].set_xlabel('No. batches')
  ax[0].set_ylabel('Loss')
  ax[0].set_title('Training Curve')

  xgrid = torch.linspace(0, 1, 500).unsqueeze(1)
  x = xgrid.numpy().squeeze()
  T = mlp(xgrid).detach().numpy().squeeze()
  ax[1].plot(x, T)
  ax[1].set_xlabel('x')
  ax[1].set_ylabel('T')
  ax[1].set_title('Temperature Profile')
  fig.tight_layout()
  fig.show()

def main():

  din = 1                 # one spatial variable (x)
  dout = 1                # one output variable (T)
  num_hidden_layers = 5
  num_hidden_units = 20
  niter = 1E4

  mlp = MLP_PINN(num_hidden_layers, num_hidden_units, din, dout)
  opt = optim.Adam(mlp.parameters(), lr=LR_INIT)

  losses = train(mlp, opt, batch_size=BATCH_SIZE, niter=int(5E3))
  plot_results(losses, mlp)


if __name__ == "__main__":
  main()
