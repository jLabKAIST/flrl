from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, latent_dim, max_t_prev, type = '1d'):
        super(CNNExtractor, self).__init__(observation_space, features_dim=latent_dim)
        self.time_dim = max_t_prev
        self.latent_dim = latent_dim
        self.type = type
        
        if type == '2d':
            self.cnn = nn.Sequential(
                #input: (B, 2, T, Fx, Fy)
                nn.Conv3d(2, int(self.latent_dim/4), kernel_size=3, padding=1),  # (B, 32, T, Fx, Fy)
                nn.ReLU(),
                nn.Conv3d(int(self.latent_dim/4), int(self.latent_dim/2), kernel_size=3, padding=1), # (B, 64, T, Fx, Fy)
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2),                 # (F/2, T/2)
                nn.Conv3d(int(self.latent_dim/2), self.latent_dim, kernel_size=3, padding=1),# (B, latent_dim, T, F/2, T/2)
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1))                 # (B, latent_dim, 1, 1, 1)
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, int(self.latent_dim/4), kernel_size=3, padding=1),  # (B, 32, F, T)
                nn.ReLU(),
                nn.Conv2d(int(self.latent_dim/4), int(self.latent_dim/2), kernel_size=3, padding=1), # (B, 64, F, T)
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,1)),                 # (F/2, T/2)
                nn.Conv2d(int(self.latent_dim/2), self.latent_dim, kernel_size=3, padding=1),# (B, latent_dim, F/2, T/2)
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))                 # (B, latent_dim, 1, 1)
            )
    

    def forward(self, obs):     
        #Input preprocessing: From [(t-n), ...., (t)] (a flattened stack of timesteps) to (batch_size, channel, frequency, timesteps)
        #For 2D case: to (batch_size, Re/Im, timestep, Freqx, Freqy)
        if self.type == '2d':
            pass
        else:
            obs = obs.unsqueeze(1)  #(B, :) -> (B, 1, :)
            
            size = obs.size()
            
            obs = obs.reshape(size[0], size[1], -1, self.time_dim)
        
        
        out = self.cnn(obs) 
        out = out.reshape(size[0], -1) # (B * latent_dim)
        
        
        return out