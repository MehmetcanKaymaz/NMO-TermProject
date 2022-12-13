from NeuralNetworks import PolicyNet
from model import BaseModel
import torch
import numpy as np

N=10

policies=[]
models=[]

for i in range(N):
    policy=PolicyNet()
    policy_path="DModels/policynet_11_{}.pth".format(i+1)
    policy.load(policy_path)
    policies.append(policy)
    model=BaseModel(x0=np.array([0,0,np.pi/2]).reshape((3,1)))
    models.append(model)

for i in range(N):
    policy=policies[i]
    model=models[i]
    T=2
    dt=0.01
    N=int(T/dt)

    target=np.array([10,10,0])

    for i in range(N):
        x=model.x
        x_tensor=torch.Tensor(x.reshape((1,3)))
        u=policy(x_tensor).item()
        model.step(u)
        if np.sqrt((x[0]-target[0])**2+(x[1]-target[1])**2)<1:
            break
    print(i)

for i in range(N):
    models[i].vis_sim(index=i+1,save=True,show=False)