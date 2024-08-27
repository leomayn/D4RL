import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

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

import matplotlib.pyplot as plt

class IL_Transition(NamedTuple):
    obs: jnp.ndarray
    done: jnp.ndarray
    teacher_act: jnp.ndarray
    info: jnp.ndarray
    
def make_train(config):
    
    env = gym.make(config["ENV_NAME"])
    # if config["TEACHER_NETWORK_TYPE"] == "Qfn":
    config["NUM_ACTORS"] = 1 * config["NUM_ENVS"]
    config["NUM_TEST_ACTORS"] = 1 * config["NUM_TEST_ENVS"]
    config["NUM_STEPS"] = env._max_episode_steps
    
    # initialize teacher model and student model
    # if config["TEACHER_NETWORK_TYPE"] == "Qfn":
    #     teacher_model = AgentRNN(action_dim=env.action_space(env.agents[0]).n, 
    #                            hidden_dim=config["TEACHER_NETWORK_HIDDEN"],
    #                            init_scale=config["TEACHER_NETWORK_INIT_SCALE"],
    #                            config=config)
    # else:
    #     teacher_model = ActorRNN(action_dim=env.action_space(env.agents[0]).n,
                                #  config=config)
    
    student_model = ActorRNN(action_dim=env.action_space.shape[0],
                             config=config)


    print("Loading dataset...")
    dataset = env.get_dataset()
    
    perm = jax.random.PRNGKey(config["SHUFFLE_SEED"])
    
    data_loader = ILDataLoader(batch_size=config['BATCH_SIZE'], shuffle=True, random_state=perm, max_steps=env._max_episode_steps, for_jax=True, load_file=dataset)
    config["DATASET_SIZE"] = len(data_loader)
    print("dataset:", config["DATASET_SIZE"])
    
    # print("dataset_obs.shape", dataset_obs.shape)
    # print("dataset_done.shape", dataset_done.shape)
    # print("dataset_teacher_act.shape", dataset_teacher_act.shape)
    # raise ValueError
    # config["DATASET_SIZE"] = dataset_obs.shape[1]
    def train(rng):
        # initialize environment
        rng, _env_rng = jax.random.split(rng)
        # _env_rngs = jax.random.split(_env_rng, config["NUM_ENVS"])
        # obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(_env_rngs)
        # initialize learner model
        rng, _s_init_rng = jax.random.split(rng)
        _s_init_x = (
            jnp.zeros((1, 1, env.observation_space.shape[0])),
            jnp.zeros((1, 1))
        )
        _s_init_h = ScannedRNN.initialize_carry(1, config["STUDENT_NETWORK_HIDDEN"])
        
        s_network_params = student_model.init(_s_init_rng, _s_init_h, _s_init_x)
        
        
        # initialize train state and optimizer
        student_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=optax.linear_schedule(init_value=config["LR"], end_value=1e-4, transition_steps=config["NUM_EPOCHS"] * config["DATASET_SIZE"] / config["BATCH_SIZE"]), eps=1e-5),
            )
        student_train_state = TrainState.create(
            apply_fn=student_model.apply,
            params=s_network_params,
            tx=student_tx,
        )
        
        def _update_step(runner_state, tested_times):
            student_train_state, rng = runner_state
            def _train_epoch(IL_training_state, after_test):
                student_train_state, init_s_h_state, rng = IL_training_state
                num_iterations = data_loader._data_size // config["BATCH_SIZE"]
                print("num itr:", num_iterations)
                def _train_minibatch(student_train_state, unused):
                    minibatch = data_loader._get_batch()
                    minibatch = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), minibatch)
                    s_h_state = ScannedRNN.initialize_carry(config["BATCH_SIZE"], config["STUDENT_NETWORK_HIDDEN"])
                    def _IL_loss_fn(student_params, s_h_state, minibatch):
                        obs = minibatch['obs']
                        done = minibatch['done']
                        teacher_act = minibatch['action']
                        
                        # forward pass
                        in_data = (obs, done)
                        s_h_state, pi = student_model.apply(student_params, s_h_state, in_data)
                        
                        # compute loss
                        loss = -pi.log_prob(teacher_act)
                        loss = jnp.mean(loss)
                        
                        # student_act_mean = pi.mean()  # Mean of the distribution
                        # loss = jnp.mean(jnp.square(student_act_mean - teacher_act))  # MSE loss
                        
                        return loss
                    grad_fn = jax.value_and_grad(_IL_loss_fn)
                    loss, grad = grad_fn(student_train_state.params, s_h_state, minibatch)
                    student_train_state = student_train_state.apply_gradients(grads=grad)
                    return student_train_state, loss

                student_train_state, IL_losses = jax.lax.scan(
                    _train_minibatch, student_train_state, jnp.arange(num_iterations)
                )
                IL_training_state = (
                    student_train_state,
                    init_s_h_state,
                    rng,
                )
                return IL_training_state, IL_losses

            IL_training_state = (
                student_train_state,
                ScannedRNN.initialize_carry(config["DATASET_SIZE"], config["STUDENT_NETWORK_HIDDEN"]),
                rng,
            )
            
            IL_training_state, IL_losses = jax.lax.scan(
                _train_epoch, IL_training_state, jnp.arange(config["TEST_INTERVAL"])
            )
            IL_loss = IL_losses.mean()
            # def print_loss(loss):
            #     print("IL loss:", loss)
            # jax.experimental.io_callback(print_loss, None, IL_losses)
            
            student_train_state, new_s_h_state, rng = IL_training_state
            new_s_h_state = new_s_h_state.squeeze()
            # test the student model
            

            
            
            runner_state = (
                student_train_state,
                rng,
            )
            
            return runner_state, IL_loss
        
        runner_state = (
            student_train_state,
            rng,
        )
        
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, np.arange(config["NUM_EPOCHS"]//config["TEST_INTERVAL"])
        )
        print(f"Training finished after {config['NUM_EPOCHS']} epochs.")
        return {"runner_state": runner_state, "metric": metric}
    
    return train

@hydra.main(version_base=None, config_path="config", config_name="offline_IL_config")
def main(config):
    config = OmegaConf.to_container(config)
    # if config["USE_UNILATERAL"]:
    #     config["STUDENT_NETWORK_SAVE_PATH"] = f'results/ILagent/{config["ENV_NAME"].split("_")[2]}/unilateral'
    # else:
    #     config["STUDENT_NETWORK_SAVE_PATH"] = f'results/ILagent/{config["ENV_NAME"].split("_")[2]}'
    env_name_str = config["ENV_NAME"]

    wandb_project = str(config["PROJECT_PREFIX"]) + env_name_str + "_IL"

    if config["DEBUG"]:
        config["WANDB_MODE"] = "disabled"
        config["TOTAL_TIMESTEPS"] = 1e5
    wandb_name = "offline_IL"
    wandb.init(
        entity=config["ENTITY"],
        project=wandb_project,
        tags=["None"],
        name=wandb_name,
        config=config,
        mode=config["WANDB_MODE"],
        # mode='disabled',
    )
    
    train = make_train(config)
    with jax.disable_jit(config["DISABLE_JIT"]):
        jit_train = jax.jit(train)
        rng = jax.random.PRNGKey(config["SEED"])
        output = jit_train(rng)
    wandb.finish()
    
    
            
    
        
if __name__ == "__main__":
    main()