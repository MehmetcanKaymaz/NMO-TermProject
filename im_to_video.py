import cv2
import numpy as np
import glob
import os

N=10
for i in range(N):

    frameSize = (1200, 1200)
    out = cv2.VideoWriter('figs/model_2_{}/output_video.mp4'.format(i+1),cv2.VideoWriter_fourcc(*'mp4v'), 20, frameSize)

    for j in range(200):
        filename="figs/model_2_{}/fig_{}.png".format(i+1,j)
        if os.path.exists(filename):
            img = cv2.imread(filename)
            out.write(img)

    out.release()