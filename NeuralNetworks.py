import os
import torch
from torch import nn


class ValueNet(nn.Module):
  def __init__(self,in_size=3,outsize=1):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(in_size, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, outsize )
    )

  def forward(self, x):
    return self.layers(x)
  
  def save(self,path):
    torch.save(self.state_dict(), path)

  def load(self,path):
    self.load_state_dict(torch.load(path))
    self.eval()


class PolicyNet(nn.Module):
  def __init__(self,in_size=3,outsize=1):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(in_size, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, outsize),
      nn.Tanh()
    )

  def forward(self, x):
    return self.layers(x)

  def save(self,path):
    torch.save(self.state_dict(), path)

  def load(self,path):
    self.load_state_dict(torch.load(path))
    self.eval()


"""valunet=ValueNet()
policynet=PolicyNet()

valunet.save("DModels/valunet.pth")
policynet.save("DModels/policynet.pth")

del valunet
del policynet

valunet=ValueNet()
policynet=PolicyNet()

valunet.load("DModels/valunet.pth")
policynet.load("DModels/policynet.pth")

print("Done!")"""