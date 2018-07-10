import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 1

        self.state_size = self.action_repeat * 2
        self.action_low = 390
        self.action_high = 420
        self.action_range = self.action_high - self.action_low 
        self.action_size = 1

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # Get max reward of 1 if within 1 unit from hover spot
        # Get min reward of -1 if 6+ units from hover spot
        
        distance = np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        
        #max_dist_sq = 4 ** 4
        #dist_sq = distance**4
        #reward = -(min(max(0,dist_sq),max_dist_sq)/(max_dist_sq/2) - 1) * .98
        
        reward = 0
        if (distance < 4):
            reward = min(1,1/(distance + 1)**2)


        #print("target pos " + str(self.target_pos))
        #print("sim.pose " + str(self.sim.pose[:3]))
        #print("dist " + str(dist))
        #print("reward " + str(reward))

        return reward / self.action_repeat

    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds)
            done = self.sim.next_timestep(rotor_speeds)    
            reward += self.get_reward() 
            z_norm = (self.sim.pose[2] - 10)/3
            z_v_norm = (self.sim.v[2]) / 3
            rotor_norm = (rotor_speeds[0] - self.action_low) / self.action_range * 2 - 1
            pose_all.append(z_norm)
            pose_all.append(z_v_norm)

        next_state = np.array(pose_all)

        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        
        #perturb the start by +- one unit
        perturb_unit = 2
        self.sim.pose[2] += (2*np.random.random()-1) * perturb_unit
        #print("Starting height {:7.3f}".format(self.sim.pose[2]))
        z_norm = (self.sim.pose[2] - 10)/5
        z_v_norm = 0
        rotor_norm = 0
        
        state = np.array(([z_norm, z_v_norm]) * self.action_repeat)

        return state