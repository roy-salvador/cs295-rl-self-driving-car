import gym
import envs.AirSimSimplifiedActionMetaRLEnv
import tensorflow as tf

# delete if it's registered
env_name = "AirSimNH-v0"
if env_name in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name]

# register environment
gym.register(
    id=env_name,
    entry_point=envs.AirSimSimplifiedActionMetaRLEnv.AirSimEnv
)



from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, LstmPolicy, register_policy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2

                                                  
                                                          
env = gym.make("AirSimNH-v0")
# Vectorized environments allow to easily multiprocess training
# we demonstrate its usefulness in the next examples
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Meta RL basically uses an LSTM policy
model = PPO2( MlpLstmPolicy, env,  nminibatches=1, verbose=1, tensorboard_log="./ppo_tensorboard/")
#model = PPO2.load("save_models/ppo_lidar_simplified_fixed_99", env=env, verbose=1, tensorboard_log="./ppo_tensorboard/")



# Train the agent
for i in range(0, 100) :
    model.learn(total_timesteps=10000, tb_log_name="lidar_metarl_")
    # Save trained model
    model.save("ppo_lidar_metarl_" + str(i))


# Enjoy trained agent
print("Testing")
obs = env.reset()
done = False
state = None
while not done:
    action, state = model.predict(obs, state=state)
    obs, rewards, done, info = env.step(action)
    env.render()

