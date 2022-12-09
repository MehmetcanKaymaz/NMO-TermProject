from model import Model
from MPCControl import NMPC
import numpy as np

model=Model()
control=NMPC()

x=model.x

refx=10
refy=10

T=10
dt=0.01
N=int(T/dt)

traj=np.array([refx,refy,0,0,0]).reshape((5,1))

control=NMPC(N=N)

u=control.run_controller(x0=x,ref_states=traj)

for i in range(N):
    x=model.step(u=u[:,i])
    #if np.sqrt((x[0]-refx)**2+(x[1]-refy)**2)<1:
        #break

model.vis()
    


