from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import ray
from ray.rllib.agents import dqn
from ray.tune.registry import register_env
from ray.tune import grid_search, run_experiments
from collections import deque
import numpy as np
import gym
from gym import spaces
from gym.spaces.discrete import Discrete
from ple import PLE
from ple.games.catcher import Catcher
import time
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor

env_name = "ple_env"

#screen dimensions, keep consistent for PLE env and ray
screen_wh = 80


class PLEPreprocessor(Preprocessor):
    def _init(self):
        self.shape = self._obs_space.shape #can vary this based on options

    def transform(self, observation):
        observation = observation / 255.0
        return observation

ModelCatalog.register_custom_preprocessor("ple_prep", PLEPreprocessor)


class PLEEnv(gym.Env):
    def __init__(self, env_config):
        game = Catcher(width=screen_wh, height=screen_wh)

        fps = 30  # fps we want to run at
        frame_skip = 2
        num_steps = 2
        force_fps = True  # False for slower speed
        display_screen = False
        # make a PLE instance.
        self.env = PLE(game, fps=fps, frame_skip=frame_skip, num_steps=num_steps,
                  force_fps=force_fps, display_screen=display_screen)
        self.env.init()
        self.action_dict = {0:None,1:97,2:100}
        #PLE env starts with black screen
        self.env.act(self.env.NOOP)

        self.action_space = Discrete(3)
        self.k = 4
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(screen_wh, screen_wh, 1 * self.k))
        self.frames = deque([], maxlen=self.k)

    def reset(self):
        self.env.reset_game()
        # PLE env starts with black screen, NOOP step to get initial screen
        self.env.act(self.env.NOOP)
        ob = np.reshape(self.env.getScreenGrayscale(), (screen_wh, screen_wh, 1))
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        #traditional gym env step
        #_obs, _rew, done, _info = env.step(env.action_space.sample())
        action_value = self.action_dict[action]
        _rew = self.env.act(action_value)
        _obs = np.reshape(self.env.getScreenGrayscale(),(screen_wh,screen_wh,1))
        self.frames.append(_obs)
        _done = self.env.game_over()
        _info = {}

        return self._get_ob(), _rew, _done, _info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)

register_env(env_name, lambda c: PLEEnv(c))

ray.init()

max_timesteps = 2000000
experiment_name = "ple_catcher_coords_{}".format(int(time.time()))

run_experiments({
    experiment_name: {
        'run': 'DQN',
        'env':env_name,
        'stop':{'timesteps_total': max_timesteps},
        'repeat':1,
        'checkpoint_freq': 1000,
        "trial_resources": {
                            'cpu': 1,#lambda spec: spec.config.num_workers,#lambda spec: spec.config.num_workers,
                            'extra_cpu': 1,
                            "gpu": 1
        },
        'config': {
            'num_workers': 1,
            #'num_gpus_per_worker':0,

            # Max num timesteps for annealing schedules. Exploration is annealed from
            # 1.0 to exploration_fraction over this number of timesteps scaled by
            # exploration_fraction
            "schedule_max_timesteps": 2000000,
            # Number of env steps to optimize for before returning
            "timesteps_per_iteration": 100,#10,#1000,#10 is too low, plot graphs is disjoint
            # Fraction of entire training period over which the exploration rate is
            # annealed
            "exploration_fraction": 0.25,
            # Final value of random action probability
            "exploration_final_eps": 0.01,
            # Update the target network every `target_network_update_freq` steps.
            "target_network_update_freq": 1000,

            "dueling": True,
            "double_q": True,
            "n_step": 5,
            "hiddens": [256],
            "buffer_size": 10000,
            "prioritized_replay": True,
            "lr": 1e-4,
            "train_batch_size": 32,
            "learning_starts": 1000,
            "grad_norm_clipping": 40,
            "clip_rewards": True,
            "preprocessor_pref": "deepmind",#"rllib",#"deepmind",
            'model':{
                'dim': screen_wh,
                'grayscale': False, #PLE set up with grayscale
                'zero_mean':False,
                "custom_preprocessor": "ple_prep",
                'custom_options':{
                    'add_coordinates': True,
                    'add_coordinates_with_r':False
                }
            },
            'tf_session_args': {
                'gpu_options': {'allow_growth': True}
            },
        },
    },
})