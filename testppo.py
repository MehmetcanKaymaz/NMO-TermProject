from EnvGym import CustomEnvBase
from stable_baselines3 import PPO
import numpy as np

env=CustomEnvBase()

model = PPO.load("modelppo/model_200000")

obs = env.reset(x0=np.array([0,0,np.pi/2]).reshape((3,1)))
i=0
while True:
    i+=1
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    #env.render()
    if dones:
        break

print(i)
env.model.vis_sim()