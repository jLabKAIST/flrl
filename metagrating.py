import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
from utils.utils_lsf import construct_fourier_coeff, generate_lsf, flatten_fourier_coeff, inverse_flatten_fourier_coeff, to_pairs
from utils.utils_opt import create_solver, get_de, binary_to_index


class Metagrating(gym.Env):
    def __init__(self, env_config, folder_path):
        super(Metagrating, self).__init__()

        self.real_action_dim = env_config['real_action_dim']
        self.imag_action_dim = env_config['imag_action_dim']
        self.action_dim = self.real_action_dim + self.imag_action_dim
        
        #Structure setup
        self.Nx = env_config['Nx']
        self.Ny = env_config['Ny']
        self.tfo_x = env_config['tfo_x']
        self.tfo_y = env_config['tfo_y']
        self.grid_x = env_config['grid_x']
        self.grid_y = env_config['grid_y']
        self.max_steps = env_config['max_steps']
        self.surface_shape = [1, self.grid_y, self.grid_x]
        self.period_x = env_config['wavelength'] / np.sin(np.deg2rad(env_config['angle']))
        self.period_y = env_config['wavelength'] / 2.
        self.period = th.tensor([self.period_x, self.period_y])
        self.wavelength = env_config['wavelength']
        self.angle = env_config['angle']
        self.pol = env_config['pol']

        #Environment setup
        self.obs_bound = env_config['obs_bound']
        self.action_bound = env_config['action_bound']
        self.obs_normalization = env_config['obs_normalization']
        self.action_scaling = env_config['action_scaling']
        self.reward_scaling = env_config['reward_scaling']
        self.num_envs = env_config['n_envs']

        self.folder_path = folder_path

        if self.Ny == 0:
            assert self.tfo_y == 0
            assert self.grid_y == 1
        else:
            assert self.tfo_y != 0
            assert self.grid_y != 1
        
        self.obs_input = env_config['obs_input']
        if self.obs_input == "flat":
            self.observation_space = spaces.Box(low=-self.obs_bound, high=self.obs_bound, shape=([self.action_dim]), dtype=np.float32)
        elif self.obs_input == "pair":
            self.observation_space = spaces.Box(low=-self.obs_bound, high=self.obs_bound, shape=([self.real_action_dim, 2]), dtype=np.float32)
            
        self.action_space = spaces.Box(low=-self.action_bound, high=self.action_bound, shape=([self.action_dim]), dtype=np.float32)
        
        
        #For saving trajectory
        self.obs_vector = None
        self.action = None
        self.reward = 0
        self.current_obs = None
        self.next_obs = None
        self.done = False

        #initalize the structure
        self.reset()


    def calculate_eff(self):
        if self.pol == 2:
            #TE & TM
            self.mee.pol = 0
            self.te_eff = get_de(self.mee, x_order = 1, y_order = 0).item()
            print(f"TE: {self.te_eff}")

            self.mee.pol = 1
            self.tm_eff = get_de(self.mee, x_order = 1, y_order = 0).item()
            print(f"TM: {self.tm_eff}")

            raw_eff = (self.te_eff + self.tm_eff) * 0.5

        else:
            self.mee.pol = self.pol
            raw_eff = get_de(self.mee, x_order = 1, y_order = 0).item()
            if self.pol == 0:
                self.te_eff = raw_eff
                self.tm_eff = 0
            elif self.pol == 1:
                self.te_eff = 0
                self.tm_eff = raw_eff

        return raw_eff
    
    def reset(self, *, options = None, seed = None):
        #Structure and eff initialization
        self.coeff = construct_fourier_coeff(Nx = self.Nx, Ny = self.Ny, cols = 'zeros')
        self.lsf = generate_lsf(self.Nx, self.Ny, self.period_x, self.period_y, self.grid_x, self.grid_y, th.tensor(self.coeff))
        
        self.structure = th.tensor(np.where(self.lsf >= 0, 1, 0)).unsqueeze(dim = 0).float()
        self.structure = binary_to_index(self.structure, self.wavelength)
        
        self.mee = create_solver(fourier_order = [self.tfo_x, self.tfo_y], pol = 0, wavelength = self.wavelength,
                                 period_x = self.period_x, period_y = self.period_y) 
        self.mee.ucell = self.structure.reshape([1, self.grid_y, self.grid_x])
        self.eff = self.calculate_eff()
         
        real_coeff, imag_coeff = flatten_fourier_coeff(self.Nx, self.Ny, self.coeff)
        if self.obs_input == "flat":
            self.obs_vector = np.concatenate([real_coeff, imag_coeff])
        elif self.obs_input == "pair":
            self.obs_vector = to_pairs(real_coeff, imag_coeff)
        
        return self.obs_vector, {}
    
    def step(self, action):  
        self.action = action
        self.current_obs = self.obs_vector
        
        if self.action_scaling != None:
            self.action = self.action * self.action_scaling
        
        
        delta_real_coeff = self.action[:self.real_action_dim]
        delta_imag_coeff = self.action[self.real_action_dim:]
        
        #Alter the coefficients based on the action (delta coeff)
        if self.obs_normalization == 'tanh':
            real_coeff, imag_coeff = flatten_fourier_coeff(self.Nx, self.Ny, self.coeff)
            
            real_coeff = np.tanh(np.array(delta_real_coeff) + real_coeff)
            imag_coeff = np.tanh(np.array(delta_imag_coeff) + imag_coeff)

            
            self.coeff = inverse_flatten_fourier_coeff(self.Nx, self.Ny, real_coeff, imag_coeff)
            
            
            if self.obs_input == 'flat':
                self.obs_vector = np.concatenate([real_coeff, imag_coeff])
            elif self.obs_input == 'pair':
                self.obs_vector = to_pairs(real_coeff, imag_coeff)

        elif self.obs_normalization == 'power':
            real_coeff, imag_coeff = flatten_fourier_coeff(self.Nx, self.Ny, self.coeff)
            
            real_coeff = np.array(delta_real_coeff) + real_coeff
            imag_coeff = np.array(delta_imag_coeff) + imag_coeff
            
            
            norm = np.sqrt(np.power(real_coeff, 2).sum() + np.power(imag_coeff, 2).sum())
            
            real_coeff = real_coeff/norm
            imag_coeff = imag_coeff/norm
            self.coeff = inverse_flatten_fourier_coeff(self.Nx, self.Ny, real_coeff, imag_coeff)
            
            if self.obs_input == 'flat':
                self.obs_vector = np.concatenate([real_coeff, imag_coeff])
            elif self.obs_input == 'pair':
                self.obs_vector = to_pairs(real_coeff, imag_coeff)
            
        else:
            real_coeff, imag_coeff = flatten_fourier_coeff(self.Nx, self.Ny, self.coeff)
            real_coeff = np.clip(real_coeff + np.array(delta_real_coeff), -self.obs_bound, self.obs_bound)
            imag_coeff = np.clip(imag_coeff + np.array(delta_imag_coeff), -self.obs_bound, self.obs_bound)
            self.coeff = inverse_flatten_fourier_coeff(self.Nx, self.Ny, real_coeff, imag_coeff)

            self.obs_vector = np.concatenate([real_coeff, imag_coeff])

        self.lsf = generate_lsf(self.Nx, self.Ny, self.period_x, self.period_y, self.grid_x, self.grid_y, th.tensor(self.coeff))
        self.binary_structure = th.tensor(np.where(self.lsf >= 0, 1, 0)).float()
        
        ucell = binary_to_index(structure = self.binary_structure, wavelength = self.wavelength)
        
        prev_eff = self.eff
        
        self.mee.ucell = ucell.reshape([1, self.grid_y, self.grid_x])
        
        self.eff = self.calculate_eff()
        
            
        print(f"eff: {self.eff}")

        self.reward = self.reward_scaling*(self.eff - prev_eff)
    
        print(f"Reward {self.reward}")

        self.next_obs = self.obs_vector
        self.reward = self.reward
        self.done = False
        
        return self.obs_vector, self.reward, False, False, {}    