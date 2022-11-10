import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class LoadMap:
    def __init__(self,txtpath,initloc,finalloc,muglocs):
        self.txtpath=txtpath
        self.initloc=initloc
        self.finalloc=finalloc
        self.muglocs=muglocs

        self.map=np.loadtxt(self.txtpath)
        self.add_events()
        self.map=np.rot90(self.map,axes=(1,0))

    def add_events(self):
        self.map[self.initloc[0],self.initloc[1]]=2
        self.map[self.finalloc[0],self.finalloc[1]]=3
        for mugloc in self.muglocs:
            self.map[mugloc[0],mugloc[1]]=4


    def vis(self):
        print(self.map)
        fig, ax = plt.subplots()
        xlim=self.map.shape[0]
        ylim=self.map.shape[1]

        for i in range(xlim):
            for j in range(ylim):
                if self.map[i,j]==0:
                    ax.add_patch(Rectangle((i,j),1,1,edgecolor="black",facecolor = 'white',fill=False))
                elif self.map[i,j]==1:
                    ax.add_patch(Rectangle((i,j),1,1,edgecolor="black",facecolor = 'black',fill=True))
                elif self.map[i,j]==2:
                    ax.add_patch(Rectangle((i,j),1,1,edgecolor="black",facecolor = 'green',fill=True))
                elif self.map[i,j]==3:
                    ax.add_patch(Rectangle((i,j),1,1,edgecolor="black",facecolor = 'yellow',fill=True))
                elif self.map[i,j]==4:
                    ax.add_patch(Rectangle((i,j),1,1,edgecolor="black",facecolor = 'brown',fill=True))
                else:
                    print("Grid id :{} is not known".format(self.map[i,j]))
        plt.xlim([0,xlim])
        plt.ylim([0,ylim])
        plt.show()

mapgen=LoadMap("examplemap.txt",[4,0],[2,0],[[3,1]])

mapgen.vis()