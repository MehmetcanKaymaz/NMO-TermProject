import gym
from gym import spaces
from model import BaseModel,ModelComplex
import numpy as np

class CustomEnvComplex(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(CustomEnvComplex, self).__init__()

        low=np.array([-1,-1],dtype=np.float64)
        high=np.array([1,1],dtype=np.float64)
        self.action_space = spaces.Box(low,high)
        
        low=np.array([-np.inf,-np.inf,-10,-np.inf,-np.pi],dtype=np.float64)
        high=np.array([np.inf,np.inf,10,np.inf,np.pi],dtype=np.float64)
        self.observation_space = spaces.Box( low , high , shape=(5,) , dtype=np.float64 )

        
    def step(self, action):
        
        action_u=np.array([action[0]*10,action[1]*np.pi])
        x=self.model.step(u=action_u)
        info={}
        reward=-np.sqrt((10-x[0])**2+(10-x[1])**2)/10
        
        done=False
        if -reward<=.05:
            done=True
        self.index+=1
        if self.index>=1000:
            done=True

        
        return x.reshape(5,), reward[0], done, info

    def reset(self):
        self.model=ModelComplex()
        x=self.model.x
        self.index=0

        return x.reshape(5,)  

    def render(self, mode="human"):
        pass
    def close (self):
        pass


class CustomEnvBase(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(CustomEnvBase, self).__init__()

        low=-1
        high=1
        self.action_space = spaces.Box(low,high,shape=(1,))
        
        low=np.array([-np.inf,-np.inf,-np.inf],dtype=np.float64)
        high=np.array([np.inf,np.inf,np.inf],dtype=np.float64)
        self.observation_space = spaces.Box( low , high , shape=(3,) , dtype=np.float64 )

        
    def step(self, action):
        
        action_u=action*np.pi

        x=self.model.step(u=action_u)
        
        info={}
        
        reward=-np.sqrt((10-x[0])**2+(10-x[1])**2)/10
        
        done=False
        if -reward<=.05:
            done=True
        self.index+=1
        if self.index>=200:
            done=True

        
        return x.reshape(3,), reward[0], done, info

    def reset(self,x0=np.zeros((3,1))):
        self.model=BaseModel(x0=x0)
        x=self.model.x
        self.index=0

        return x.reshape(3,)  

    def render(self, mode="human"):
        pass
    def close (self):
        pass