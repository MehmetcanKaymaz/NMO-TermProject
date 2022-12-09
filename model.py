import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import cv2

class BaseModel:
    def __init__(self,x0=np.zeros((3,1)),dt=0.01,V=10,N=1,t=0):
        self.dt=dt

        self.x=x0
        self.states=[]
        self.controls=[]
        self.V=V
        self.N=N
        self.t=t

        self.states.append(self.x)

    def step(self,u):
        self.x=self.update_states(self.x,u)
        return self.x

    def update_states(self,x,u):
        for i in range(self.N):
            x=x+self.calculate_xdot(x,u)*self.dt
            self.states.append(x)
            self.controls.append(u)
        return x
    
    def calculate_xdot(self,x,u):
        xdot=np.zeros((3,1))

        xdot[0]=self.V*np.cos(x[2])
        xdot[1]=self.V*np.sin(x[2])
        xdot[2]=u
        return xdot

    def vis_sim(self):
        states=np.array(self.states)
        N=len(self.states)

        image_folder="../lamborghini_cut.jpg"
        image=mpimg.imread(image_folder)

        fig, ax = plt.subplots(figsize = (12, 12))

        for i in range(N):
            
            plt.cla()

            ax.scatter(0,0,s=10,c="blue")
            ax.scatter(10,10,s=10,c="red")

            ax.plot(states[:i,0],states[:i,1],color="orange")

            if self.t==0:
                angle=states[i,2]
            elif self.t==1:
                angle=states[i,3]
            else:
                print("!!! Unknown t !!!!!!!")
                break
            image_r=self.rotate_bound(image,int(180-angle*180/np.pi))
            imagebox = OffsetImage(image_r, zoom = 0.08)
            ab = AnnotationBbox(imagebox, (states[i,0], states[i,1]), frameon = False)
            ax.add_artist(ab)
            plt.xlim([-2,12])
            plt.ylim([-2,12])
            
            plt.pause(0.001)


    def rotate_bound(self,image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])


        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        return cv2.warpAffine(image, M, (nW, nH), borderValue=(255,255,255))

class ModelComplex(BaseModel):
    def __init__(self,dt=0.01,N=1,x0=np.zeros((5,1)),V=10):
        super().__init__(x0=x0,dt=dt,V=V,N=N,t=1)

    def calculate_xdot(self,x,u):
        xdot=np.zeros((5,1))

        xdot[0]=x[2]*np.cos(x[3])
        xdot[1]=x[2]*np.sin(x[3])
        xdot[2]=u[0]
        xdot[3]=x[4]
        xdot[4]=u[1]
        return xdot

    def vis(self):
        plt.cla()
        t1=np.linspace(0,len(self.states)*self.dt,len(self.states))
        t2=np.linspace(0,len(self.controls)*self.dt,len(self.controls))

        states=np.array(self.states)
        plt.plot(t1,states[:,0],label="Position x(m)")
        plt.plot(t1,states[:,1],label="Position y(m)")
        plt.plot(t1,states[:,2],label="Velocity(m/s)")
        plt.plot(t1,states[:,3]*180/np.pi,label="Angle(deg)")
        plt.plot(t1,states[:,4]*180/np.pi,label="Angular Rate(deg/sec)")

        plt.plot(t2,np.array(self.controls)[:,1]*180/np.pi,label="Control(deg/sec^2)")
        plt.plot(t2,np.array(self.controls)[:,0],label="Control(m/sec^2)")
        plt.legend()
        plt.show()











