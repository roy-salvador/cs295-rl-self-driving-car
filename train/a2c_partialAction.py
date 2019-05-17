import gym
import envs.AirSimPartialActionMlpEnv as AirSimEnv

# delete if it's registered
env_name = "AirSimNH-v0"
if env_name in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name]

# register environment
gym.register(
    id=env_name,
    entry_point=AirSimEnv.AirSimEnv
)


from stable_baselines.common.policies import MlpPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C
from stable_baselines.bench import Monitor


env = gym.make("AirSimNH-v0")
log_dir = "a2c_run_mlpPolicy_parEnv_350k"
# Vectorized environments allow to easily multiprocess training
# we demonstrate its usefulness in the next examples
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = A2C(MlpPolicy, env, verbose=1, tensorboard_log="./a2c_tensorboard/")

# Train the agent
model.learn(total_timesteps=350000, tb_log_name=log_dir)
# Save trained model
model.save(log_dir)

del model
# Load model
model = A2C.load(log_dir)

# Enjoy trained agent
print("Testing")
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()