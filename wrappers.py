import gymnasium as gym
import pickle
import os
from utils.utils_lsf import construct_fourier_coeff

class TrajectoryRecorder(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.trajectory = (None, None, None, 0, 0) #(current state, next state, action, reward, done)
        self.trajectories = []
        self.iter = 0
        self.id = str(os.getpid())
        
    def step(self, action):
        #use bestrecorder step
        ret = self.env.step(action)
        
        current_state = self.unwrapped.current_obs
        next_state = self.unwrapped.next_obs
        action = self.unwrapped.action
        reward = self.unwrapped.reward
        done = self.unwrapped.done
        
        self.trajectory = (current_state, next_state, action, reward, done)
        self.trajectories.append(self.trajectory)
        
        return ret
    
    def reset(self, *, options = None, seed = None):
        if self.iter > 1:
            #Add with other episodic trajectories
            with open(f"{self.env.unwrapped.folder_path}/{self.id}_trajectory.pickle", "rb") as f:
                tmp = pickle.load(f)
            
            tmp = tmp + self.trajectories #concat two lists
            with open(f"{self.env.unwrapped.folder_path}/{self.id}_trajectory.pickle", "wb") as f:
                pickle.dump(tmp, f)
                
        elif self.iter == 1:
            #First save (first episode)
            with open(f"{self.env.unwrapped.folder_path}/{self.id}_trajectory.pickle", "wb") as f:
                pickle.dump(self.trajectories, f)
        
        self.trajectories = []
        self.iter += 1
        
        return self.env.reset(options = None, seed = None)
    
    def get_trajectories(self):
        return self.trajectories

class BestRecorder(gym.Wrapper):
    """
        Wrapper to record best eff which will be compiled in every episode ending
    """
    def __init__(self, env):
        super().__init__(env)
        self.best = (0, 0, 0, None)  # efficiency, te eff, tm eff, structure
        self.eps_best = (0, None) #storing best eff on each episode
        self.prev_best = 0 #keeping the previous best eff, to be used in the exploration reward calculation

    def step(self, action):
        ret = self.env.step(action)

        if self.unwrapped.eff > self.best[0]:
            self.prev_best = self.best[0]
            self.best = (self.unwrapped.eff, self.unwrapped.te_eff, self.unwrapped.tm_eff, self.unwrapped.mee.ucell.detach().numpy())
            
            
        if self.unwrapped.eff > self.eps_best[0]:
            self.eps_best = (self.unwrapped.eff, self.unwrapped.mee.ucell.detach().numpy())
            # print(self.eps_best[0])

        return ret
    
    def reset(self, *, options = None, seed = None):
        return self.env.reset(options = None, seed = None)

    def get_best(self):
        return self.best
    
    def get_max_steps(self):
        return self.unwrapped.max_steps
    
    def get_best_eps(self):
        result = self.eps_best
        
        return result
    
    def reset_best_eps(self):
        self.eps_best = (0, None)

    def get_eff(self):
        return self.unwrapped.eff
    
class RandomInitializer(gym.Wrapper):
    """
        Wrapper to randomly initalize the starting structure for exploration
    """

    def __init__(self, env, cutoff):
        super().__init__(env)
        self.reset_counter = 0
        self.cutoff = cutoff
    
    def reset(self, *, options = None, seed = None):
        if self.reset_counter >= self.cutoff:
            #stop initializing randomly
            self.unwrapped.coeff = construct_fourier_coeff(Nx = self.Nx, Ny = self.Ny, cols = 'zeros')
        else:
            #initialize randomly
            self.unwrapped.coeff = construct_fourier_coeff(Nx = self.Nx, Ny = self.Ny, cols = 'random_gaussian')
        
        self.reset_counter += 1

        return self.env.reset(options = None, seed = None)
    
    def get_coeff(self):
        return self.unwrapped.coeff
