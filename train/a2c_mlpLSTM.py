import gym
import logging, argparse

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C
from stable_baselines.bench import Monitor

def set_env(option):
	if option == 'full':
		import envs.AirSimFullActionMlpLSTMEnv as AirSimEnv
	else:
		import envs.AirSimPartialActionMlpLSTMEnv as AirSimEnv

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)

	parser = argparse.ArgumentParser(add_help=True)
	parser.add_argument('--timesteps', type=int, action='store', 
						dest='timesteps', default=10000,
	                    help='Path to output trained classifier model')
	parser.add_argument('--log-dir', type=str, action='store', 
						dest='log_dir',default='a2c_mlpLSTMPolicy_10k',
	                    help='Size to crop images to')
	parser.add_argument('--env', type=str, action='store',
						dest='airsim_env', default='full')
	parser.add_argument('--last-trained', type=int, action='store',
						dest='cnt_last_model', default=0)

	args = parser.parse_args()
	log_dir = args.log_dir
	tensorboard_log = "./a2c_tensorboard/"
	timesteps = args.timesteps
	if args.airsim_env == 'full':
		import envs.AirSimFullActionMlpLSTMEnv as AirSimEnv
	elif args.airsim_env == 'partial':
		import envs.AirSimPartialActionMlpLSTMEnv as AirSimEnv
	else:
		print('No Env')
	# delete if it's registered
	env_name = "AirSimNH-v0"

	if env_name in gym.envs.registry.env_specs:
		del gym.envs.registry.env_specs[env_name]

	# register environment
	gym.register(
	    id=env_name,
	    entry_point=AirSimEnv.AirSimEnv
	)

	env = gym.make("AirSimNH-v0")
	# Vectorized environments allow to easily multiprocess training
	# we demonstrate its usefulness in the next examples
	env = Monitor(env, log_dir, allow_early_resets=True)
	env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

	if args.cnt_last_model == 0:
		model = A2C(MlpLstmPolicy, env, verbose=1, 
					tensorboard_log=tensorboard_log)
		model.learn(total_timesteps=timesteps, tb_log_name=log_dir,  reset_num_timesteps=False)
		model.save(log_dir+"_"+str(0))

	# model = A2C(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_log)

	# Train the agent
	#model.learn(total_timesteps=timesteps, tb_log_name=log_dir)
	# Save trained model
	#model.save(log_dir)


	count = args.cnt_last_model
	err_count = args.cnt_last_model
	model = A2C.load(log_dir+"_"+str(count),
		env=env, verbose=1, 
		tensorboard_log="./a2c_tensorboard/")
	try:
		print("Loading model: "+ log_dir +"_"+str(count))
		for i in range(count, 200):
			# Train the agent
			model.learn(total_timesteps=timesteps, tb_log_name=log_dir,  reset_num_timesteps=False)
			# Save trained model
			model.save(log_dir+"_"+str(i))
			count += 1
	except:
		err_count = err_count + 1
		print('Latest iteration: ', count)
		count = count + 1

	del model
	# Load model
	model = A2C.load(+log_dir+"_"+str(i)) #"./save_models/"+

	'''

	# Enjoy trained agent
	print("Testing")
	obs = env.reset()
	for i in range(1000):
	    action, _states = model.predict(obs)
	    obs, rewards, dones, info = env.step(action)
	    env.render()
	'''