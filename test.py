from model import BaseModel,ModelComplex
from MPCControl import NMPCComplex,NMPCBase
import numpy as np

model=BaseModel()
x=model.x

refx=11
refy=10

T=1.7
dt=0.01
N=int(T/dt)

traj=np.array([refx,refy,0]).reshape((3,1))

control=NMPCBase(N=N)

u=control.run_controller(x0=x,ref_states=traj)

for i in range(N):
    x=model.step(u=u[i])
    if np.sqrt((x[0]-refx)**2+(x[1]-refy)**2)<1:
        break

model.vis_sim()
    


