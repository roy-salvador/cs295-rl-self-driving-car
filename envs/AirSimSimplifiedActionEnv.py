'''
    AirSim Environment with Simplified Action Controls [Steering, Movement  (Throttle, Brake, Gear merged into one action) ]
'''
from gym import Env, spaces
import airsim
import math
import numpy as np
import time
import random

# Defined in settings.json
MAX_LIDAR_RANGE = 10 
CLOCK_SPEED = 10
AIRSIM_IP = "127.0.0.11"

MAX_TIME_STEPS = 120
MAX_SPEED = 50.0
WAY_POINTS = 4
# List of waypoints in the map
POSSIBLE_WAY_POINTS = [(airsim.Vector3r(30 , 0, -1), airsim.utils.to_quaternion(0,0,math.pi)), # northward
                       (airsim.Vector3r(-30 , 0, -1), airsim.utils.to_quaternion(0,0,0)),     # southward
                       (airsim.Vector3r(-5, 30, -1), airsim.utils.to_quaternion(0,0,-math.pi/2)),  # eastward
                       (airsim.Vector3r(-5, -30, -1), airsim.utils.to_quaternion(0,0,math.pi/2))  # westward
                    ]
    
GOAL_THRESHOLD = 10  # Consider successful arrival if within meters of the target way point
MAX_GOAL_DISTANCE = 50 # Farthest distance in meters from the goal which can be rewarded

class  AirSimEnv(Env) :
    '''
    Gym Environment Wrapper class of AirSimNH.
    '''     
    def __init__(self, 
                 ip = AIRSIM_IP, 
                 way_points = WAY_POINTS, 
                 max_time_steps = MAX_TIME_STEPS,
                 goal_threshold = GOAL_THRESHOLD,
                 max_goal_distance = MAX_GOAL_DISTANCE,
                 possible_way_points = POSSIBLE_WAY_POINTS,
                 max_lidar_range = MAX_LIDAR_RANGE,
                 clock_speed = CLOCK_SPEED
                ) :
        self.max_time_steps=max_time_steps         
        self.way_points = way_points
        self.goal_threshold = goal_threshold
        self.max_goal_distance = max_goal_distance
        self.possible_way_points = possible_way_points
        self.max_lidar_range = max_lidar_range
        self.clock_speed = clock_speed

        # Establish connection
        self.client = airsim.CarClient(ip=ip)
        self.client.confirmConnection() 
        
    
        # Vectorized action space
        # Concatenated Steering, Movement
        self.action_space = spaces.Box(low=np.array([-1,-1]), high=np.array([1,1]), dtype=np.float32)
        

        # Vectorized Observation space
        # Quaternion Orientation  (w, x, y, z) + normalized speed + LIDAR-derived vector
        low = [-1, -1, -1, -1, 0]
        high = [1, 1, 1, 1, 1]
        for i in range(0, 360) :
            low.append(0)
            high.append(1)       
        self.observation_space =  spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        
        self.init_new_episode()
    

    # Initialize 
    def init_new_episode(self) :
        self.last_action = [0,0] 
        self.last_reward = 0
        self.client.enableApiControl(True, "AgentCar")
        self.client.enableApiControl(True, "TargetCar")
        self.car_controls = airsim.CarControls()
        self.is_done = False
        self.time_steps_left=self.max_time_steps
        self.trajectory_distance=0
        self.last_collision_depth=0
        agent_car_pose = self.client.simGetVehiclePose(vehicle_name="TargetCar")
        self.car_pos = [agent_car_pose.position.x_val, agent_car_pose.position.y_val, agent_car_pose.position.z_val]
        self.update_current_state()
        self.init_way_points()
    
    # Initialize way points
    def init_way_points(self) :
        self.current_goal = 0
        self.goals = []
        
        # randomize
        arr = np.arange(len(self.possible_way_points))
        np.random.shuffle(arr)
        for i in range(0, self.way_points) :
            self.goals.append(self.possible_way_points[arr[i]])
        
        # set initial way point
        pose = airsim.Pose(self.goals[0][0], self.goals[0][1])
        self.client.simSetVehiclePose(pose, True, vehicle_name="TargetCar")
        
    
    
    # Computes the euclidean distance between two 3D points
    def _euclidean_distance(self, p1, p2) :
        return math.sqrt( (p1[0]-p2[0])**2 +  (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2 )
        
    # Computes the yaw angle in rad between two coordinates in relation to p1
    def _get_yaw_angle_rad(self, p1, p2) :
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # handle undefined case i.e. math.atan(infinity) approaches 90;  math.atan(negative infinity) approaches -90 degrees
        if dx == 0 :
            if dy >= 0 :
                angle_rad = math.pi / 2
            else :
                angle_rad = -math.pi / 2
        else :
            angle_rad = math.atan( dy / dx )  # In first quadrant
            
        
        if dy < 0 and dx >= 0:
            angle_rad = 2 * math.pi + angle_rad # In fourth quadrant
        elif dx < 0:
            angle_rad = math.pi + angle_rad # in second and third quadrant
             
         
        return angle_rad
    
    # Update the environment's state
    def update_current_state(self) :
        '''
        Observation States
            self.car_state - car orientation, speed
            self.lidar_reading - LIDAR-derived distance vector (0-359 degrees)
        Auxiliary states not part of observation   
            self.current_pos - position of the agent car
            self.collisions - count of collisions in the episode
            self.is_in_collision - True if agent car is in collision for the current timestep
        '''
  
        car_state = self.client.getCarState()
        kinematics = car_state.kinematics_estimated

        # Kinematics state
        obs  = []
        obs.append(kinematics.orientation.x_val)
        obs.append(kinematics.orientation.y_val)
        obs.append(kinematics.orientation.z_val)
        obs.append(kinematics.orientation.x_val)
        obs.append(car_state.speed / MAX_SPEED)

        
        # Compute first the trajectory distance and update car position
        new_pos = [kinematics.position.x_val, kinematics.position.y_val, kinematics.position.z_val]
        self.trajectory_distance +=  self._euclidean_distance(self.car_pos, new_pos)
        self.car_pos = new_pos 
                
        
        # LIDAR-derived distance  (horizontal LIDAR)
        lidar_dist = []
        for i in range(0, 360) :
            lidar_dist.append(self.max_lidar_range / self.max_lidar_range)
            
        pcl = self.client.getLidarData('LidarSensor1').point_cloud
        if (len(pcl) >= 3):
            #print(len(pcl))
            points = np.array(pcl, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0]/3), 3))
            for point in points :
                dist = self._euclidean_distance(self.car_pos, point)
                angle_rad = self._get_yaw_angle_rad(self.car_pos, point)
                angle_deg = angle_rad / math.pi * 180
                lidar_index = int(angle_deg)
                if (dist / self.max_lidar_range) < lidar_dist[lidar_index] :
                     lidar_dist[lidar_index] = dist / self.max_lidar_range
        
        
        # Update observation state
        self.current_state = np.array(obs + lidar_dist)

        
        # Collision details
        collision_info = self.client.simGetCollisionInfo() 
        if not collision_info.has_collided :
            self.collisions = 0
            self.is_in_collision = False
        else :
            if self.last_collision_depth != collision_info.penetration_depth :
                self.collisions += 1
                self.is_in_collision = True 
            else :
                 self.is_in_collision = False
        self.last_collision_depth = collision_info.penetration_depth    
        

    
    # Computes the reward and checks whether the episode ends
    def get_reward_status(self) :
    
        reward = 0
        is_done = False
        
        
        # Penalty for collision
        if self.is_in_collision :
            reward += -5
            
        # Extra Penalty for laziness - not moving or extremely slow to encourage exploration and travel faster
        # Speed
        if abs(self.current_state[4]) < (1.0/MAX_SPEED) :
            reward += -1
        
        # Penalty for driving in reverse to force car to prefer driving forward
        if self.car_controls.manual_gear == -1 :
            reward += -1
            
        # Intermediate penalty to encourage to reach goal in smaller amount of time
        reward += - 1
            
        # Intermediate reward proportional to distance from way point. 
        tcar_state = self.client.simGetVehiclePose(vehicle_name="TargetCar")
        tcar_car_pos = [tcar_state.position.x_val, tcar_state.position.y_val, tcar_state.position.z_val]
        dist = self._euclidean_distance(self.car_pos, tcar_car_pos)
        reward += ((self.max_goal_distance - dist) / (self.max_goal_distance - self.goal_threshold)) 
        
        # Intermediate reward for facing the direction of the way point. Inversely penalizes if facing the opposite direction
        tcar_yaw = self._get_yaw_angle_rad(self.car_pos, tcar_car_pos)
        reward += math.cos( tcar_yaw -  airsim.utils.to_eularian_angles(self.client.getCarState().kinematics_estimated.orientation)[2] )
            
        
        
        # Check if maximum time steps reached
        if self.time_steps_left == 0 :
            is_done = True
        else :

            if dist <= self.goal_threshold :
                self.current_goal += 1
           
                # Last way point reached
                if self.current_goal >= len(self.goals) :
                    is_done = True
                    reward += 25
                # Intermediate way point reached
                else :
                    reward += 10
                    

            # set to next way point
            if self.current_goal < len(self.goals) :
                pose = airsim.Pose(self.goals[self.current_goal][0], self.goals[self.current_goal][1])
                self.client.simSetVehiclePose(pose, True, vehicle_name="TargetCar")
        self.is_done = is_done
        print("Goal Distance:" + str(dist))
        self.goal_distance = dist
        
        return reward, is_done
    
    

    # Performs an action and updates the environment 
    # An action is given by a vector [Steering, Movement]
    def step(self, action):
        # Save current way point
        current_waypoint = self.goals[self.current_goal]  
        self.last_action = action
    
        # Perform the action
        self.car_controls.steering = float(action[0])
        
        # Reverse
        if action[1] < -0.33 : 
            self.car_controls.is_manual_gear = True;
            self.car_controls.manual_gear = -1
            self.car_controls.throttle = 1
            self.car_controls.brake = 0
            #print('REVERSE')
        # Move forward
        elif action[1] > 0.33 :
            self.car_controls.is_manual_gear = False;
            self.car_controls.manual_gear = 0
            self.car_controls.throttle = 1
            self.car_controls.brake = 0
            #print('FORWARD')
        # Brake
        else :
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
            #print('BRAKE')
            
 
        self.client.setCarControls(self.car_controls, "AgentCar")
        time.sleep(1.0/self.clock_speed)
        self.time_steps_left -= 1

        # Get reward and next state (reward)
        self.update_current_state() 
        reward, is_done = self.get_reward_status()
        self.last_reward = reward
        self.current_state[2] = self.last_reward
        
        
        # Prepare informational 
        info = {'collisions' : self.collisions, 
               'trajectory_distance' : self.trajectory_distance, 
               'goal_distance' : self.goal_distance,
               'car_pos_x' : self.car_pos[0], 
               'car_pos_y' : self.car_pos[1],
               'current_waypoint_x' : current_waypoint[0].x_val,
               'current_waypoint_y' : current_waypoint[0].y_val}
        
        print("Timesteps left: " + str(self.time_steps_left) + ", Speed : "  + str( round(self.current_state[4]*MAX_SPEED,2) ) + 
            ", Steering : "  + str( round(float(action[0]),2) ) + ", Reward:" + str(reward))
        return (self.current_state, reward, is_done, info)
    
    
    # Resets the environment. Places agent at starting point
    def reset(self):
    
        self.client.reset()
        self.init_new_episode()
        return self.current_state
    
    # Nothing to render so far. Visualization can be done through the simulator 
    def render(self, mode='human'):   
        None
        
    def seed(self, seed):
        random.seed(seed)
        np.random.seed
        
if __name__ == '__main__':
    None