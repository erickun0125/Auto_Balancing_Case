"""Manager-based RL tasks for the Suitcase robot.

This package defines a simple balancing environment configuration that can be
used to train a wheeled suitcase robot to maintain balance against external
disturbances using a USD asset.
"""

import gymnasium as gym

from . import agents

# Gym registration: Isaac-Suitecase-Flat-v0 (Training environment)
gym.register(
    id="Isaac-Suitecase-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.suitecase_env_cfg:SuitecaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SuitecasePPORunnerCfg",
        # Other frameworks can be added if needed:
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Gym registration: Isaac-Suitecase-Flat-Play-v0 (Play/Inference environment)
gym.register(
    id="Isaac-Suitecase-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.suitecase_env_cfg:SuitecasePlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SuitecasePPORunnerCfg",
    },
)



