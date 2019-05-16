'''
    Outputs the statistics (collision, distance covered, total reward per episode) to a file for each experiments performed:
'''
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2, A2C
import gym
import airsim
import math
import os

# Customized AirSim Environments
import envs.AirSimFullActionMlpLSTMEnv
import envs.AirSimPartialActionMlpLSTMEnv
import envs.AirSimFullActionMlpEnv
import envs.AirSimFullActionMlpRew1Env
import envs.AirSimPartialActionMlpEnv
import envs.AirSimPartialActionMlpRew1Env
import envs.AirSimSimplifiedActionEnv
import envs.AirSimSimplifiedActionMetaRLEnv

# Environment Settings for the test
MAX_LIDAR_RANGE = 10 
CLOCK_SPEED = 10
AIRSIM_IP = "127.0.0.10"
MAX_TIME_STEPS = 120
WAY_POINTS = 1
# List of waypoints in the map
POSSIBLE_WAY_POINTS = [(airsim.Vector3r(30 , 0, -1), airsim.utils.to_quaternion(0,0,math.pi)), # northward
                       #(airsim.Vector3r(-30 , 0, -1), airsim.utils.to_quaternion(0,0,0)),     # southward
                       #(airsim.Vector3r(-5, 30, -1), airsim.utils.to_quaternion(0,0,-math.pi/2)),  # eastward
                       #(airsim.Vector3r(-5, -30, -1), airsim.utils.to_quaternion(0,0,math.pi/2))  # westward
                    ]
# Consider successful arrival if within meters of the target way point
GOAL_THRESHOLD = 10 
# Farthest distance in meters from the goal which can be rewarded
MAX_GOAL_DISTANCE = 100 

MODEL_DIR = 'models'
EXPERIMENTS = [
    {'title' : 'A2C_FullAction_MlpLSTM', 'algo': A2C, 'env' : envs.AirSimFullActionMlpLSTMEnv, 'model' : 'a2c_mlpLSTM_fullEnv_rew1_99'},
    {'title' : 'A2C_PartialAction_MlpLSTM', 'algo': A2C, 'env' : envs.AirSimPartialActionMlpLSTMEnv, 'model' : 'a2c_mlpLSTM_partialEnv_rew1_99'},
    {'title' : 'A2C_FullAction_Mlp', 'algo': A2C, 'env' : envs.AirSimFullActionMlpEnv, 'model' : 'a2c_mlpPolicy_fullEnv_99'},
    {'title' : 'A2C_FullAction_Mlp_Rew1', 'algo': A2C, 'env' : envs.AirSimFullActionMlpRew1Env, 'model' : 'a2c_mlpPolicy_fullEnv_reward1_99'},
    {'title' : 'A2C_PartialAction_Mlp', 'algo': A2C, 'env' : envs.AirSimPartialActionMlpEnv, 'model' : 'a2c_mlpPolicy_partialEnv_99'},
    {'title' : 'A2C_PartialAction_Mlp_Rew1', 'algo': A2C, 'env' : envs.AirSimPartialActionMlpRew1Env, 'model' : 'a2c_mlpPolicy_partialEnv_reward1_99'},
    {'title' : 'PPO2_SimplifiedAction_Mlp', 'algo': PPO2, 'env' : envs.AirSimSimplifiedActionEnv, 'model' : 'ppo_mlp_simplified_99'},
    {'title' : 'PPO2_SimplifiedAction_MlpLstm', 'algo': PPO2, 'env' : envs.AirSimSimplifiedActionEnv, 'model' : 'ppo_mlplstm_simplified_99'},
    {'title' : 'PPO2_SimplifiedAction_MetaRL', 'algo': PPO2, 'env' : envs.AirSimSimplifiedActionMetaRLEnv, 'model' : 'ppo_metarl_simplified_99'},
]
TEST_EPISODES = 100

for expt in EXPERIMENTS :
    
    # delete gym environment from registry
    env_name = "AirSimNH-Test-v0"
    if env_name in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[env_name]
    # register environment
    gym.register(
        id=env_name,
        entry_point=expt['env'].AirSimEnv,
        kwargs={'ip' : AIRSIM_IP,
                'max_lidar_range' : MAX_LIDAR_RANGE,
                'clock_speed' : CLOCK_SPEED,
                'max_time_steps' : MAX_TIME_STEPS,
                'goal_threshold' : GOAL_THRESHOLD,
                'way_points' : WAY_POINTS,
                'possible_way_points' : POSSIBLE_WAY_POINTS,
                'max_goal_distance' : MAX_GOAL_DISTANCE}
    )
    env = gym.make(env_name)
    env = DummyVecEnv([lambda: env])
    if expt['algo'] == PPO2 :
        print('HERE')
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    model = expt['algo'].load( os.path.join(MODEL_DIR, expt['model']) ,env=env, verbose=1)
    
    
    # Write statistics to file
    f_trajectory = open(expt['title'] + '_episode_trajectory.csv', "a+")
    f_episode_stat = open(expt['title'] + '_episode_stats.csv', "a+")
    f_trajectory.write("episode_number,timestep,waypoint_x,waypoint_y,car_pos_x,car_pos_y\n")
    f_episode_stat.write("episode_number,total_reward,collisions,trajectory_distance,goal_distance\n")
    f_trajectory.close()
    f_episode_stat.close()
    
    for i in range(0, TEST_EPISODES) :
        print('*********************************************')
        print('Episode ' + str(i+1))
        print('*********************************************')
    
        timestep = 0
        obs = env.reset()
        done = False
        state = None
        total_reward = 0
        
        f_trajectory = open(expt['title'] + '_episode_trajectory.csv', "a+")
        while not done:
            action, state = model.predict(obs, state=state)
            obs, rewards, done, info = env.step(action)
            
            print(rewards)
            print(info)
            
            total_reward += rewards[0]
            info = info[0]
            timestep += 1
            
            f_trajectory.write(str(i+1) 
                + ',' + str(timestep) 
                + ',' + str(info['current_waypoint_x'])
                + ',' + str(info['current_waypoint_y'])
                + ',' + str(info['car_pos_x'])
                + ',' + str(info['car_pos_y'])
                + "\n"
        )
            
        f_trajectory.close()
        f_episode_stat = open(expt['title'] + '_episode_stats.csv', "a+")
        f_episode_stat.write(str(i+1) 
                + ',' + str(total_reward) 
                + ',' + str(info['collisions'])
                + ',' + str(info['trajectory_distance'])
                + ',' + str(info['goal_distance'])
                + "\n"
        )
        f_episode_stat.close()
        
