import numpy as np
import torch


class DataLoader:
    def __init__(self,N=1000,dt=0.01,V=10,Q=np.array([1,1,0])):
        self.N=N
        self.random_states=None
        self.next_states=None
        self.actions=None
        self.rewards=None
        self.policy=None

        self.dt=dt
        self.V=V
        self.Q=Q

    def generate_data(self,policy):
        self.policy=policy
        self.randomSamle()
        self.calculate_next_state()

        #print(self.actions)

        return self.random_states.T,self.next_states.T,self.rewards

    def randomSamle(self):
        x=np.random.uniform(-2,12,self.N)
        y=np.random.uniform(-2,12,self.N)
        angle=np.random.uniform(-np.pi,np.pi,self.N)

        self.random_states=np.zeros((3,self.N))
        self.random_states[0,:]=x
        self.random_states[1,:]=y
        self.random_states[2,:]=angle

    def calculate_next_state(self):
        self.next_states=np.zeros((3,self.N))
        self.actions=np.zeros(self.N)
        self.rewards=np.zeros(self.N)

        for i in range(self.N):
            x=self.random_states[:,i].reshape((3,1))
            u=self.run_policy(x=x)
            xnext=self.simple_model(x,u)
            self.actions[i]=u
            self.next_states[:,i]=xnext.reshape(3)
            self.rewards[i]=self.calculate_reward(x=xnext)

    def simple_model(self,x,u):
        xdot=np.zeros((3,1))
        xdot[0]=self.V*np.cos(x[2])
        xdot[1]=self.V*np.sin(x[2])
        xdot[2]=u*np.pi

        x=x+xdot*self.dt

        return x

    def run_policy(self,x):
        xtensor=torch.Tensor(x.reshape((1,3)))
        a=self.policy(xtensor)
        u=a.item()

        return u

    def calculate_reward(self,x):
        reward=-np.sqrt(self.Q[0]*(x[0]-10)**2+self.Q[1]*(x[1]-10)**2)/10
        return reward

        
"""from NeuralNetworks import PolicyNet
policy=PolicyNet()

loader=DataLoader(N=5)

x,xn,r=loader.generate_data(policy=policy)

print(x)
print(xn)
print(r)"""