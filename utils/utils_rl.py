import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from functools import partial

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, path, verbose=0):
        super(CustomCallback, self).__init__(verbose)

        self.path = path
        

    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_start(self) -> None:
        pass
   
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        
        if self.num_timesteps % self.training_env.env_method("get_max_steps")[0] == 0:
            print('Rollout end for all envs')      
            
            #Max aggregate the efficiency from all envs
            best = max(self.training_env.env_method("get_best"), key = lambda x: x[0])

            best_eps = max(self.training_env.env_method("get_best_eps"), key = lambda x: x[0])
            title_eff = round(best[0] * 100, 3)

            #Logging best efficiency
            self.logger.record("rollout/max_eff", best[0])
            self.logger.record("rollout/max_eff_per_episode", best_eps[0])
            self.logger.record("rollout/te_eff", best[1])
            self.logger.record("rollout/tm_eff", best[2])
            
            #Logging image structure
            fig = plt.figure()
            
            if best[-1][0].shape[-2] == 1:
                #1D structure, needs to be tiled
                fig.add_subplot().imshow(np.tile(best[-1][0], (128, 1)), cmap = 'binary')        
            else:
                fig.add_subplot().imshow(best[-1][0], cmap = 'binary')

            fig.suptitle(f"eff: {title_eff}")       
            self.logger.record("rollout/max_eff_structure", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
            plt.close()
            
            np.save(f"{self.path}/eff_{title_eff}", best[-1]) 

            self.training_env.env_method("reset_best_eps")
                            
            for i in range(len(self.model.policy.log_std)):
                self.logger.record(f"rollout/log_std_{i}", self.model.policy.log_std.detach().cpu().numpy()[i])

        
    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


class NonLearnableStdCallback(BaseCallback):
    """
        A custom callback that derives from ``BaseCallback``.

        :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose: int = 0):
        super(NonLearnableStdCallback, self).__init__(verbose)

    def _on_training_start(self) -> None:
        self.model.policy.log_std.requires_grad = False
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass

class DeterministicPolicyCallback(BaseCallback):
    """
        A custom callback that derives from ``BaseCallback``.

        :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, action_dim: int, verbose: int = 0):
        super(DeterministicPolicyCallback, self).__init__(verbose)
        self.action_dim = action_dim
        
    def _on_training_start(self) -> None:
        self.model.policy.log_std = torch.nn.Parameter(torch.ones(self.action_dim) * -20.0, requires_grad=False)
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass
