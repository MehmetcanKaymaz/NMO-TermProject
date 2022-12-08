import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self,dt=0.01,N=1,x0=np.zeros((3,1)),V=10):
        
        self.info = "Simple dynamic model"
        self.dt=dt
        self.x=x0
        self.V=V
        self.N=N

        self.states=[]
        self.controls=[]

        self.states.append(self.x)

    def step(self,u):
        self.x=self.__update_states(self.x,u)
        return self.x

    def __update_states(self,x,u):
        for i in range(self.N):
            x=x+self.__calculate_xdot(x,u)*self.dt
            self.states.append(x)
            self.controls.append(u)
        return x
    
    def __calculate_xdot(self,x,u):
        xdot=np.zeros((3,1))

        xdot[0]=self.V*np.cos(x[2])
        xdot[1]=self.V*np.sin(x[2])
        xdot[2]=u

        return xdot

    def vis(self):
        t1=np.linspace(0,len(self.states)*self.dt,len(self.states))
        t2=np.linspace(0,len(self.controls)*self.dt,len(self.controls))

        states=np.array(self.states)
        plt.plot(t1,states[:,0],label="Position x(m)")
        plt.plot(t1,states[:,1],label="Position y(m)")
        plt.plot(t1,states[:,2]*180/np.pi,label="Angle(deg)")

        plt.plot(t2,np.array(self.controls)*180/np.pi,label="Control(deg/sec)")
        plt.legend()
        plt.show()


class ModelTrain:
    def __init__(self,dt=0.01,N=1,x0=np.zeros((3,1)),V=10):
        
        self.info = "Simple dynamic model"
        self.dt=dt
        self.x=x0
        self.V=V
        self.N=N

    def step(self,u):
        self.x=self.__update_states(self.x,u)
        return self.x

    def __update_states(self,x,u):
        for i in range(self.N):
            x=x+self.__calculate_xdot(x,u)*self.dt
        return x
    
    def __calculate_xdot(self,x,u):
        xdot=np.zeros((3,1))

        xdot[0]=self.V*np.cos(x[2])
        xdot[1]=self.V*np.sin(x[2])
        xdot[2]=u

        return xdot






