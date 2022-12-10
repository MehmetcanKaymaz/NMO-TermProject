from EnvGym import CustomEnvBase
from stable_baselines3 import PPO

env=CustomEnvBase()

model = PPO.load("modelppo/model_100000")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    #env.render()
    if dones:
        break
print("done")
env.model.vis_sim()