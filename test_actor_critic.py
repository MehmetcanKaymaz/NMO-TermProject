from NeuralNetworks import PolicyNet
from model import BaseModel
import torch

model=BaseModel()
policy=PolicyNet()

policy_path="DModels/policynet_2.pth"

policy.load(policy_path)

T=1.7
dt=0.01
N=int(T/dt)

for i in range(N):
    x=model.x
    x_tensor=torch.Tensor(x.reshape((1,3)))
    u=policy(x_tensor).item()
    model.step(u)

model.vis_sim()