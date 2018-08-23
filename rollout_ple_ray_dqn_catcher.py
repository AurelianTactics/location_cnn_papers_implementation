from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonic_on_ray
import ray
from ray.rllib.agents import dqn
from ray.tune.registry import register_env
from collections import deque
import numpy as np
import gym
from gym import spaces
from gym.spaces.discrete import Discrete
from ple import PLE
from ple.games.catcher import Catcher

env_name = "ple_env"

screen_wh = 80

class PLEEnv(gym.Env):
    def __init__(self, env_config):
        game = Catcher(width=screen_wh, height=screen_wh)

        fps = 30  # fps we want to run at
        frame_skip = 2
        num_steps = 2
        force_fps = False # False for slower speed
        display_screen = True
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
        #_obs = self.env.getScreenGrayscale()
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

config = dqn.DEFAULT_CONFIG.copy()

#dqn model params
#https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/dqn/dqn.py
#model catalog params
#https://github.com/ray-project/ray/blob/9559873d135de78734a835d9725f2c0dabe4ace7/python/ray/rllib/models/catalog.py


max_timesteps = 100000

config.update({

    'num_workers': 0,
    'num_gpus_per_worker':0,

    # Max num timesteps for annealing schedules. Exploration is annealed from
    # 1.0 to exploration_fraction over this number of timesteps scaled by
    # exploration_fraction
    "schedule_max_timesteps": 10,
    # Number of env steps to optimize for before returning
    "timesteps_per_iteration": 1000,
    # Fraction of entire training period over which the exploration rate is
    # annealed
    "exploration_fraction": 0.1,
    # Final value of random action probability
    "exploration_final_eps": 0.00,
    # Update the target network every `target_network_update_freq` steps.
    "target_network_update_freq": 500000,

    "dueling": True,
    "double_q": True,
    "n_step": 10,
    "hiddens": [256],
    "buffer_size": 500000,
    "prioritized_replay": True,
    "lr": 5e-4,
    "train_batch_size": 32,
    "learning_starts": 100000,
    "grad_norm_clipping": 40,
    "clip_rewards": True,
    "preprocessor_pref": "deepmind",#"rllib",#"deepmind",
    'model':{
        'dim': screen_wh,
        'grayscale': False, #PLE set up with grayscale
        'zero_mean':False,
        'custom_options':{
            'add_coordinates': False,
            'add_coordinates_with_r':False
        }
    },
    # 'trial_resources': {
    #     'gpu': 1
    # },
    'tf_session_args': {
        'gpu_options': {'allow_growth': True}
    }
})

alg = dqn.DQNAgent(config=config, env=env_name)
#LOAD YOUR MODEL HERE
#alg.restore('/home/ray_results/ple_catcher_SOLVED_rs_1534273375/DQN_ple_env_0_2018-08-14_15-02-55604goybf/checkpoint-12000')

env = PLEEnv(config)

rollout_steps = 10000
steps = 0
while steps < rollout_steps:
    state = env.reset()
    done = False
    reward_total = 0.0
    episode_len = 0
    while not done and steps < rollout_steps:
        action = alg.compute_action(state)
        next_state, reward, done, _ = env.step(action)
        reward_total += reward
        steps += 1
        episode_len += 1
        state = next_state
        if episode_len > 200:
            break
    print("Episode reward: {} -- Episode len: {}".format(reward_total,episode_len))

