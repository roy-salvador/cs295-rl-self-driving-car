import gym
import envs.AirSimSimplifiedActionEnv
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2


# delete if it's registered
env_name = "AirSimNonMeta-v0"
if env_name in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name]

# register environment
gym.register(
    id=env_name,
    entry_point= envs.AirSimSimplifiedActionEnv.AirSimEnv
)

env = gym.make(env_name)
# Vectorized environments allow to easily multiprocess training
# we demonstrate its usefulness in the next examples
env = DummyVecEnv([lambda: env]) 
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Either an MLP LSTM Policy or MLP Policy
model = PPO2( MlpPolicy, env,  nminibatches=1, verbose=1, tensorboard_log="./ppo_tensorboard/")
#model = PPO2( MlpLstmPolicy, env,  nminibatches=1, verbose=1, tensorboard_log="./ppo_tensorboard/")


#model = PPO2.load("save_models/ppo_lidar_simplified_final_mlplstm_80", env=env, verbose=1, tensorboard_log="./ppo_tensorboard/")


# Train the agent
for i in range(0, 100) :
    model.learn(total_timesteps=10000, tb_log_name="lidar_simplified_final_mlp")
    # Save trained model
    model.save("ppo_lidar_simplified_final_mlp_" + str(i))

# Enjoy trained agent
print("Testing")
obs = env.reset()
done = False
state = None
while not done:
    action, state = model.predict(obs, state=state)
    obs, rewards, done, info = env.step(action)
    env.render()
    
    
    


# Train the agent
#model.learn(total_timesteps=100000, tb_log_name="lidar_custom_meta_cs20_orientation")
# Save trained model
#model.save("ppo_lidar_custom_meta_cs20_orientation")
#model.learn(total_timesteps=100000, tb_log_name="lidar_custom_meta_cs20_orientation")
# Save trained model
#model.save("ppo_lidar_custom_meta_cs20_orientation")


