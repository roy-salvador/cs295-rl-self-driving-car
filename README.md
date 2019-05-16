# cs295-rl-self-driving-car

Implementation of Open AI Gym environment(s) using Microsoft AirSim Simulator for training self-driving cars using Deep Reinforcement Learning. 
A Mini Project requirement for CS 295 course at University of the Philippines Diliman AY 2018-2019 under Sir Prospero Naval.

<p align="center">
  <img width="500" height="500" src="https://github.com/roy-salvador/cs295-rl-self-driving-car/raw/master/images/airsimenv.png">
</p>


## Requirements
* [Microsoft AirSim Simulator](https://github.com/microsoft/AirSim)
* [OpenAI Gym](https://gym.openai.com/)
* [Stable Baselines](https://github.com/hill-a/stable-baselines)

## Instructions

1. Download / Setup the [Microsoft AirSimNH map v 1.2.1](https://github.com/Microsoft/AirSim/releases) which is a small suburban neighborhood.

2. Set the following with your AirSim [settings.json](https://github.com/Microsoft/AirSim/blob/master/docs/settings.md) file. Change with your preferred local host IP Address.

``` 
{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "Car",
  "RpcEnabled": true,
  "LocalHostIp": "127.0.0.10",

  "ClockSpeed": 10,
  "Vehicles": {
        "AgentCar": {
          "VehicleType": "PhysXCar",
          "X": 0, "Y": 0, "Z": 0,
          "Sensors": {
                "LidarSensor1": { 
                    "SensorType": 6,
                    "NumberOfChannels": 16,
                    "Enabled" : true,
                    "Range" : 10,
                    "RotationsPerSecond": 10,
					"PointsPerSecond": 100000,
                    "DrawDebugPoints": true
                }
          }
        },
        "TargetCar": {
          "VehicleType": "PhysXCar",
          "EnableCollisionPassthrogh": false,
          "X": 5, "Y": 0, "Z": 0
        }
         
   } 
}
``` 
3. Run the AirSim simulator.

<p align="center">
  <img width="1000" height="300" src="https://github.com/roy-salvador/cs295-rl-self-driving-car/raw/master/images/airsim.png">
</p>

4. Register by indicating your chosen IP Address and create the environment. You can set options such the number of way points to reach and their position.
```
# delete gym environment from registry
env_name = "<YOUR GYM ENVIRONMENT NAEME>"
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
```
5. Train the model using [Stable Baselines](https://github.com/hill-a/stable-baselines), For the project, we tried [A2C](https://stable-baselines.readthedocs.io/en/master/modules/a2c.html) and [PPO2](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html).
6. For loading and running the trained RL models / agents, check `test_performance.py` as an example.
