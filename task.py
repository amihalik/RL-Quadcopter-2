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
        self.action_repeat = 3

        self.state_size = self.action_repeat * 3
        self.action_low = 360
        self.action_high = 450
        self.action_range = self.action_high - self.action_low 
        self.action_size = 1

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # Get max reward of 1 if within 1 unit from hover spot
        # Get min reward of -1 if 6+ units from hover spot
        max_dist_sq = 16
        dist_sq = np.linalg.norm(self.sim.pose[:3] - self.target_pos)**2
        reward = -(min(max(0,dist_sq),max_dist_sq)/(max_dist_sq/2) - 1) * .98

        #print("target pos " + str(self.target_pos))
        #print("sim.pose " + str(self.sim.pose[:3]))
        #print("dist " + str(dist))
        #print("reward " + str(reward))

        return reward / self.action_repeat
    

        return (pose - 10) / 10
    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append((self.sim.pose[2] - 10)/10)
            #pose_all.append(self.sim.v[2])
            pose_all.append((rotor_speeds[0] - self.action_low) / self.action_range * 2 - 1)
            pose_all.append((self.sim.v[2]) / 3 - 1)
            #pose_all.append((self.sim.pose[2] - 10)/10)
            #pose_all.append(self.sim.pose[:3])
            #pose_all.append(self.sim.pose[5])
            #pose_all.append(self.sim.pose)
            #pose_all.append(self.sim.v)

        #next_state = np.concatenate(pose_all)
        next_state = np.array(pose_all)

        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        #state = np.concatenate(([self.sim.pose]) * self.action_repeat) 
        #state = np.concatenate(([self.sim.pose,self.sim.v]) * self.action_repeat) 
        #state = np.concatenate(([self.sim.pose[:3]]) * self.action_repeat) 
        #state = (np.array(([self.sim.pose[2]]) * self.action_repeat) - 10 ) / 10
        #state = np.array(([(self.sim.pose[2] - 10 ) / 10, self.sim.v[2]]) * self.action_repeat)
        state = np.array(([(self.sim.pose[2] - 10 ) / 10, 0.,0.]) * self.action_repeat)

        return state