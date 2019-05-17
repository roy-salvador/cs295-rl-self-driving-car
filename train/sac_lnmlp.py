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



from stable_baselines.sac.policies import FeedForwardPolicy, LnMlpPolicy, register_policy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import SAC


# Custom MLP policy of three layers of size 128 each
class CustomSACPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                           layers=[512, 512, 512],
                                           layer_norm=True,
                                           feature_extraction="mlp")
register_policy('CustomSACPolicy', CustomSACPolicy)                                                           
env = gym.make("AirSimNH-v0")

# Vectorized environments allow to easily multiprocess training
# we demonstrate its usefulness in the next examples
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

#model = SAC( 'CustomSACPolicy', env, verbose=1, tensorboard_log="./ppo_tensorboard/")
# Continue training from saved model 
model = SAC.load("sac_lnmlp_target_normal_52", env=env, verbose=1, tensorboard_log="./ppo_tensorboard/")


# Train the agent
for i in range(30, 100) :
    model.learn(total_timesteps=10000, tb_log_name="sac_lnmlp_target_normal")
    # Save trained model
    model.save("sac_lnmlp_target_normal_" + str(i))



# Enjoy trained agent
print("Testing")
obs = env.reset()
done = False
state = None
while not done:
    action, state = model.predict(obs, state=state)
    obs, rewards, done, info = env.step(action)
    env.render()
