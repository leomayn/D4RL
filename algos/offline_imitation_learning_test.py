import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'
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

from brax import envs
import mediapy as media
import imageio

class IL_Transition(NamedTuple):
    obs: jnp.ndarray
    done: jnp.ndarray
    teacher_act: jnp.ndarray
    info: jnp.ndarray
    
def save_video(frames, filename='trajectory.mp4'):
    # Append .mp4 to ensure the correct format
    if not filename.endswith('.mp4'):
        filename += '.mp4'
    
    # Ensure the frames are in the correct format (e.g., numpy arrays, correct color channels)
    with imageio.get_writer(filename, fps=20, format='mp4') as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Video saved as {filename}")

def get_rollout(params, config):
    env = envs.get_environment(config["TEST_ENV_NAME"])
    
    student_model = ActorRNN(action_dim=env.action_size,
                             config=config)
    
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    
    key = jax.random.PRNGKey(0)
    key, key_r = jax.random.split(key)
    state = reset_fn(key_r)
    obs = state.obs
    done = state.done
    h_state = ScannedRNN.initialize_carry(1, config["STUDENT_NETWORK_HIDDEN"])
    network_params = params
    rollout = [state.pipeline_state]
    timestep = 0
    # grab a trajectory
    frames = []
    # while not done:
    for i in range(1000):
        key, key_a = jax.random.split(key)
        obs = obs[np.newaxis, np.newaxis, :]
        done = jnp.array(done)[np.newaxis, np.newaxis]
        ins = (obs, done)
        h_state, pi = student_model.apply(network_params, h_state, ins)
        actions = pi.sample(seed=key_a)
        actions = jnp.squeeze(actions)
        state = step_fn(state, actions)
        done = state.done
        obs = state.obs
        rollout.append(state.pipeline_state)
        
    frames = env.render(trajectory=rollout, height=240, width=320)

    return frames
    
def make_train(config):
    
    env = gym.make(config["ENV_NAME"])
    test_env = envs.get_environment(config["TEST_ENV_NAME"])
    test_reset_fn = jax.jit(jax.vmap(test_env.reset))
    test_step_fn = jax.jit(jax.vmap(test_env.step))
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
        
        def _train_epochs_and_one_test(runner_state, tested_times):
            student_train_state, rng, best_test_return = runner_state
            def _train_epoch(IL_training_state, after_test):
                student_train_state, rng = IL_training_state
                num_iterations = data_loader._data_size // config["BATCH_SIZE"]
                print("num itr:", num_iterations)
                def _train_minibatch(student_train_state, unused):
                    minibatch, s_h_state = data_loader._get_batch()
                    minibatch = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), minibatch)
                    def _IL_loss_fn(student_params, s_h_state, minibatch):
                        obs = minibatch['obs']
                        done = minibatch['done']
                        teacher_act = minibatch['action']
                        
                        # forward pass
                        in_data = (obs, done)
                        s_h_state, pi = student_model.apply(student_params, s_h_state, in_data)
                        
                        print("act dim:", teacher_act.shape)
                        
                        # compute loss
                        # loss = -pi.log_prob(teacher_act)
                        # loss = jnp.mean(loss)
                        
                        student_act_mean = pi.mean()  # Mean of the distribution
                        loss = jnp.mean(jnp.square(student_act_mean - teacher_act))  # MSE loss
                        
                        return loss
                    grad_fn = jax.value_and_grad(_IL_loss_fn)
                    loss, grad = grad_fn(student_train_state.params, s_h_state, minibatch)
                    
                    data_loader.update_hidden_state(s_h_state)
                    
                    student_train_state = student_train_state.apply_gradients(grads=grad)
                    return student_train_state, loss
                
                student_train_state, IL_losses = jax.lax.scan(
                    _train_minibatch, student_train_state, jnp.arange(num_iterations)
                )
                IL_training_state = (
                    student_train_state,
                    rng,
                )
                return IL_training_state, IL_losses
            
            IL_training_state = (
                student_train_state,
                rng,
            )
            
            IL_training_state, IL_losses = jax.lax.scan(
                _train_epoch, IL_training_state, jnp.arange(config["TEST_INTERVAL"])
            )
            IL_loss = IL_losses.mean()
            # def print_loss(loss):
            #     print("IL loss:", loss)
            # jax.experimental.io_callback(print_loss, None, IL_losses)
            
            student_train_state, rng = IL_training_state
            # test the student model
            def _test_step(test_state, unused):
                s_state, test_env_state, test_h_state, rng = test_state
                test_obs = test_env_state.obs
                test_done = test_env_state.done
                # test_obs_batch = batchify(test_obs, env.agents, config["NUM_TEST_ACTORS"])
                test_in = (test_obs[np.newaxis, :], test_done[np.newaxis, :])
                
                # test_h_state, q_vals = teacher_model.apply(t_network_params, test_h_state, test_in)
                # test_act = jnp.argmax(q_vals, axis=-1)[0]
                # print("test_act", test_act)
                # raise ValueError
                
                test_h_state, pi = student_model.apply(s_state.params, test_h_state, test_in)
                test_act = pi.sample(seed=rng)[0]
                
                # test_act = unbatchify(test_act, env.agents, config["NUM_TEST_ENVS"], 1)
                test_env_state = test_step_fn(test_env_state, test_act)
                test_info = test_env_state.info
                test_info = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_TEST_ACTORS"])), test_info)
                test_rewards = test_env_state.reward
                # test_done_batch = batchify(test_dones, env.agents, config["NUM_TEST_ACTORS"]).squeeze()
                
                test_state = (s_state, test_env_state, test_h_state, rng)
                # return test_state, (batchify(test_rewards, env.agents, config["NUM_TEST_ACTORS"]).mean(), test_env_state, test_obs)
                return test_state, (test_rewards.mean(), test_env_state, test_obs)
            _test_rng, rng = jax.random.split(rng)
            _test_rngs = jax.random.split(_test_rng, config["NUM_TEST_ENVS"])
            test_env_state  = test_reset_fn(_test_rngs)

            test_state = (
                student_train_state,
                test_env_state,
                ScannedRNN.initialize_carry(config["NUM_TEST_ACTORS"], config["STUDENT_NETWORK_HIDDEN"]),
                rng,
            )
            test_state, test_rewards_states=jax.lax.scan(
                _test_step, test_state, jnp.arange(config["NUM_TEST_STEPS"])
            )
            test_rewards, test_states, test_obs = test_rewards_states
            student_train_state = test_state[0]
            test_return = test_rewards.sum()
            def callback(params, tested_times, test_return, best_test_return, IL_loss, test_states, test_obs):
                epoch = tested_times * config["TEST_INTERVAL"]
                wandb.log({"test_return": test_return,
                           "IL_loss": IL_loss,},
                          step=epoch)
                print(f"Epoch: {epoch}, test return: {test_return}, IL loss: {IL_loss}")
                # if update_step == 1 or test_return > max(best_test_return, -15) and update_step > 500:
                # if update_step == 200 or test_return > max(best_test_return, -15) and update_step > 200:
                '''
                if tested_times == config["NUM_EPOCHS"]//config["TEST_INTERVAL"] - 1:
                    if not os.path.exists(config["STUDENT_NETWORK_SAVE_PATH"]):
                        os.makedirs(config["STUDENT_NETWORK_SAVE_PATH"])
                    file_path = os.path.join(config["STUDENT_NETWORK_SAVE_PATH"], 'final.msgpack')
                    with open(file_path, "wb") as f:
                        f.write(serialization.to_bytes(params))
                    print(f"Saved the best model to {file_path} with test return {test_return}")
                '''
            jax.experimental.io_callback(callback, None, student_train_state.params, tested_times, test_return, best_test_return, IL_loss, test_states, test_obs)
            
            runner_state = (
                student_train_state,
                rng,
                jnp.maximum(best_test_return, test_return),
            )
            
            return runner_state, (test_return, IL_loss)
        
        runner_state = (
            student_train_state,
            rng,
            float('-inf') # best return
        )
        
        runner_state, metric = jax.lax.scan(
            _train_epochs_and_one_test, runner_state, np.arange(config["NUM_EPOCHS"]//config["TEST_INTERVAL"])
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
    
    if config["TRAIN"]:
        train = make_train(config)
        with jax.disable_jit(config["DISABLE_JIT"]):
            jit_train = jax.jit(train)
            rng = jax.random.PRNGKey(config["SEED"])
            output = jit_train(rng)
        print("training finished")
        
    if config["RENDER"]:
        # Load model parameters
        student_model = ActorRNN(action_dim=gym.make(config["ENV_NAME"]).action_space.shape[0], config=config)
        rng = jax.random.PRNGKey(0)
        _s_init_x = (
            jnp.zeros((1, 1, gym.make(config["ENV_NAME"]).observation_space.shape[0])),
            jnp.zeros((1, 1))
        )
        _s_init_h = ScannedRNN.initialize_carry(1, config["STUDENT_NETWORK_HIDDEN"])

        init_params = student_model.init(rng, _s_init_h, _s_init_x)

        params_path = os.path.join(config["STUDENT_NETWORK_SAVE_PATH"], 'final.msgpack')
        if os.path.exists(params_path):
            with open(params_path, "rb") as f:
                params_bytes = f.read()
            params = serialization.from_bytes(init_params, params_bytes)
            print(f"Loaded parameters from {params_path}")
        else:
            print(f"Parameters file not found at {params_path}. Please check the path and try again.")
            return
    # rendering code
        start_time = time.time()
        frames = get_rollout(params, config)
        save_video(frames, f'{config["ENV_NAME"]}_{config["ALG_NAME"]}.mp4')
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Time for rendering: {total_time:.2f}")
    


        
if __name__ == "__main__":
    main()