from EnvGym import CustomEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

env=CustomEnv()
res=check_env(env)

"""print(res)

model = PPO("MlpPolicy", env, verbose=0,tensorboard_log="./tensorboard/")
model.learn(total_timesteps=5000000)
model.save("modelppo/model_5000000_2")"""

model = PPO.load("modelppo/model_5000000")
model.set_env(env)

model.learn(total_timesteps=5000000)
model.save("modelppo/model_10000000")

"""
del model # remove to demonstrate saving and loading

model = PPO.load("modelppo/model0")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    #env.render()
    if dones:
        break

env.model.vis_sim()"""
