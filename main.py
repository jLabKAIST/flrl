import os
import sys
import torch as th
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('../utils/')

from omegaconf import open_dict

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import FlattenExtractor
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import CheckpointCallback

from utils.utils_rl import CustomCallback, NonLearnableStdCallback, DeterministicPolicyCallback
from wrappers import BestRecorder, TrajectoryRecorder, RandomInitializer
from metagrating import Metagrating
from networks import CNNExtractor

import hydra
from omegaconf import OmegaConf, open_dict
  

def make_env(env_config, folder_path):
    env = Metagrating(env_config, folder_path)
    env = TrajectoryRecorder(env)
    env = BestRecorder(env)
    env = RandomInitializer(env, 0)
    
    env = TimeLimit(env, max_episode_steps=env_config['max_steps'])
    return env

@hydra.main(config_path='.', config_name="test_1d")
def main(cfg):
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        
    real_action_dim = (cfg.env.Nx + 1) * (cfg.env.Ny + 1)
    imag_action_dim = (cfg.env.Nx) * (cfg.env.Ny + 1)

    cfg.env.real_action_dim = real_action_dim
    cfg.env.imag_action_dim = imag_action_dim
    
    # Create multiple environments along with the monitor
    vec_env = make_vec_env(
        make_env, vec_env_cls = SubprocVecEnv, 
        n_envs = cfg.env.n_envs, seed = cfg.ppo.seed,
        env_kwargs = {'env_config': cfg.env, 'folder_path': output_dir},
    )
    vec_env = VecFrameStack(vec_env, cfg.env.max_t_prev)
    vec_env = VecNormalize(vec_env, **cfg.env.vec_norm_kwargs)

    
    # Define a callback method to keep track of the max. eff.
    callback = CustomCallback(output_dir)
    checkpoint = CheckpointCallback(
        save_freq = 10000, save_path = os.path.join(output_dir, "model")
    )
    callbacks = [callback, checkpoint]

    if cfg.feature_extractor == 'mlp':
        features_extractor_class = FlattenExtractor
    if cfg.feature_extractor == 'cnn':
        #Only for stacked observations of several timesteps
        with open_dict(cfg):
            cfg[cfg.feature_extractor].max_t_prev = cfg.env.max_t_prev
        features_extractor_class = CNNExtractor    
    
    #define the actor critic networks
    net_arch_dict = dict( 
            pi=[cfg.pi_layer_size for _ in range(cfg.pi_layer_width)],
            vf=[cfg.vf_layer_size for _ in range(cfg.vf_layer_width)]
        )
    
    if cfg.policy_type == 'stochastic_static':
        # keep the std of the policy constant throughout the training
        callbacks.append(NonLearnableStdCallback())
    elif cfg.policy_type == 'deterministic':
        # set the std of the policy approx. to 0
        callbacks.append(DeterministicPolicyCallback(action_dim=real_action_dim+imag_action_dim))
        
    # RL Training
    policy_kwargs = dict(
            activation_fn = th.nn.ReLU,
            features_extractor_class = features_extractor_class,
            features_extractor_kwargs = cfg[cfg.feature_extractor],
            net_arch = net_arch_dict,
            log_std_init = cfg.log_std_init,   
        )
        
    model = PPO(
            'MlpPolicy', vec_env, policy_kwargs = policy_kwargs,
            tensorboard_log=output_dir,
            batch_size=int(cfg.env.max_steps*cfg.env.n_envs*cfg.minibatch_ratio),
            n_steps=int(cfg.env.max_steps*cfg.nsteps_ratio),
            clip_range = get_linear_fn(start=cfg.clip_range_start, end=cfg.clip_range_end, end_fraction =cfg.end_fraction),
            device='cpu',
            **cfg.ppo
        )

    model.learn(total_timesteps = cfg.total_timesteps, callback = callbacks)
    
    model.save(os.path.join(output_dir, "model", "final.pt"))
    
if __name__ == "__main__":
    main()