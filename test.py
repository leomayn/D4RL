import hydra
import time
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.training.train_state import TrainState
import optax
from omegaconf import OmegaConf
from typing import NamedTuple
from functools import partial
import matplotlib.animation as animation
import wandb
import pickle

from utils.networks import ActorRNN, AgentRNN, ScannedRNN, RewardModel, RewardModelFF, print_jnp_shapes
from utils.jax_dataloader import Trajectory, ILDataLoader

import gym
import d4rl

from brax import envs
import os
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

env = gym.make("halfcheetah-expert-v2")

# env = envs.get_environment('halfcheetah')

dataset = env.get_dataset()
print(dataset['rewards'].mean())

print(print)
'''

# keys
print(envs._envs.keys())