import numpy as np
import cv2


N=10

img_arr=[]
fps=30

for i in range(N):
    video_name='figs/model_2_{}/output_video.mp4'.format(i+1)
    cap=cv2.VideoCapture(video_name)
    while cap.isOpened():
        ret,frame=cap.read()

        if ret:
            img_arr.append(frame)
        else:
            break

out=cv2.VideoWriter('figs/scenario2.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,(1200,1200))

for img in img_arr:
    out.write(img)

out.release()