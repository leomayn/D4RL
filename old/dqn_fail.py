import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from jax_rl.algorithms import DQN
from jax_rl.utils import Transition

import gym
import d4rl

from brax import envs
import mediapy as media
import imageio

def generate_reward(il_model, il_params, obs, action, done, il_h_state):
    obs_ = obs[np.newaxis, :]  # Add time_step dimension
    dones_ = done[np.newaxis, :]  # Add time_step dimension
    print("obs:", obs_.shape)
    print("dones:", dones_.shape)
    print("Action shape in generate_reward before log_prob:", action.shape)
    in_data = (obs_, dones_)
    il_h_state, pi = il_model.apply(il_params, il_h_state, in_data)
    return (pi.log_prob(action), il_h_state)
    
def save_video(frames, filename='trajectory.mp4'):
    # Ensure the frames are in the correct format (e.g., numpy arrays, correct color channels)
    with imageio.get_writer(filename, fps=20, format='mp4') as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Video saved as {filename}")

def get_rollout(params, config):
    env = envs.get_environment(config["TEST_ENV_NAME"])
    
    student_model = ActorRNN(action_dim=gym.make(config["ENV_NAME"]).action_space.shape[0],
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
    for i in range(5000):
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
    
# need to fix loop logic 
def make_train(config):
    
    env = envs.create(config["ENV_NAME"])
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))
    
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    
    agent_rnn = AgentRNN(
        action_dim=env.action_size,
        hidden_dim=config['AGENT_HIDDEN_DIM'],
        init_scale=config['AGENT_INIT_SCALE'],
        config=config
    )

    init_hidden = ScannedRNN.initialize_carry(config["NUM_ENVS"], config['AGENT_HIDDEN_DIM'])
    dqn_agent = DQN(
        key=jax.random.PRNGKey(0),
        n_states=env.observation_size,
        n_actions=env.action_size,
        gamma=config['GAMMA'],
        buffer_size=config['BUFFER_SIZE'],
        model=agent_rnn,
        lr=config['LR'],
        num_envs=config['NUM_ENVS']
    )

    # IL
    il_model = ActorRNN(action_dim=env.action_size, config=config)
    rng = jax.random.PRNGKey(0)
    _s_init_x = (
        jnp.zeros((1, 1, env.observation_size)),
        jnp.zeros((1, 1))
    )
    _s_init_h = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["IL_NETWORK_HIDDEN"])

    init_params = il_model.init(rng, _s_init_h, _s_init_x)
    params_path = os.path.join(config["IL_NETWORK_SAVE_PATH"], 'final.msgpack')

    if os.path.exists(params_path):
        with open(params_path, "rb") as f:
            params_bytes = f.read()
        il_params = serialization.from_bytes(init_params, params_bytes)
        print(f"Loaded parameters from {params_path}")
    else:
        print(f"Parameters file not found at {params_path}. Please check the path and try again.")
        return
    # END OF IL
    def train(rng):
        reset_rng, rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
        env_state = reset_fn(reset_rngs)
        
        def _train_updates_and_one_test(runner_state, tested_times):
            student_train_state, rng, best_test_return = runner_state
            reset_rng, rng = jax.random.split(rng)
            reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
            env_state = reset_fn(reset_rngs)
            def update_step(episode_state, episode_idx):
                state, rng, episode_rewards, best_test_return = episode_state
                total_reward = jnp.zeros(config['NUM_ENVS'])
                rng, step_rng = jax.random.split(rng)
                hidden_state = init_hidden
                def env_step(step_state, t):
                    state, rng, total_reward, il_h_state, hidden_state = step_state
                    rng, key = jax.random.split(rng)
                    prev_obs = state.obs
                    prev_done = state.done

                    # Get actions
                    rng, hidden_state, actions = dqn_agent.act(key, prev_obs, prev_done, hidden_state, exploration=True)
                    
                    # Ensure actions are correctly shaped and of the right type
                    actions = jnp.asarray(actions)  # Convert to JAX array if not already one
                    
                    # Step the environment with the actions
                    next_state = step_fn(state, actions)
                    done = next_state.done
                    obs = next_state.obs
                    reward, il_h_state = generate_reward(il_model, il_params, prev_obs, actions, prev_done, il_h_state)
                    
                    dqn_agent.update_buffer(Transition(s=state, a=actions, r=reward, d=done, s_next=obs))
                    loss = dqn_agent.train(batch_size=config['BUFFER_BATCH_SIZE'])
                    
                    total_reward += reward
                    total_reward = jnp.squeeze(total_reward)
                    new_state = next_state
                    train_state = fail

                    return (new_state, rng, total_reward, il_h_state, hidden_state), loss
                
                train_state, IL_losses = jax.lax.scan(
                    env_step, train_state, jnp.arange(config["NUM_STEPS"])
                )
                IL_training_state = (
                    student_train_state,
                    rng,
                )
                return IL_training_state, IL_losses
            
            DQN_training_state = (
                train_state,
                rng,
            )
            
            IL_training_state, IL_losses = jax.lax.scan(
                update_step, IL_training_state, jnp.arange(config["TEST_INTERVAL"])
            )
            IL_loss = IL_losses.mean()
            # def print_loss(loss):
            #     print("IL loss:", loss)
            # jax.experimental.io_callback(print_loss, None, IL_losses)
            
            train_state, rng = IL_training_state
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
                if tested_times == config["NUM_EPOCHS"]//config["TEST_INTERVAL"] - 1:
                    if not os.path.exists(config["STUDENT_NETWORK_SAVE_PATH"]):
                        os.makedirs(config["STUDENT_NETWORK_SAVE_PATH"])
                    file_path = os.path.join(config["STUDENT_NETWORK_SAVE_PATH"], 'final.msgpack')
                    with open(file_path, "wb") as f:
                        f.write(serialization.to_bytes(params))
                    print(f"Saved the best model to {file_path} with test return {test_return}")
            jax.experimental.io_callback(callback, None, student_train_state.params, tested_times, test_return, best_test_return, IL_loss, test_states, test_obs)
            
            runner_state = (
                student_train_state,
                rng,
                jnp.maximum(best_test_return, test_return),
            )
            
            return runner_state, (test_return, IL_loss)
        
        runner_state = (
            train_state,
            rng,
            float('-inf') # best return
        )
        
        runner_state, metric = jax.lax.scan(
            _train_updates_and_one_test, runner_state, np.arange(config["NUM_UPDATES"]//config["TEST_INTERVAL"])
        )
        print(f"Training finished after {config['NUM_UPDATES']} updates.")
        return {"runner_state": runner_state, "metric": metric}
    
    return train

@hydra.main(version_base=None, config_path="config", config_name="dqn_config")
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
        # mode=config["WANDB_MODE"],
        mode='disabled',
    )
    
    train = make_train(config)
    with jax.disable_jit(config["DISABLE_JIT"]):
        jit_train = jax.jit(train)
        rng = jax.random.PRNGKey(config["SEED"])
        output = jit_train(rng)
    print("training finished")
    train_state = jax.tree_util.tree_map(lambda x: x[0], output["runner_state"][0])
    params = train_state.params
    
    
    # rendering code
    
    start_time = time.time()
    frames = get_rollout(params, config)
    save_video(frames, config["ENV_NAME"])
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time for rendering: {total_time:.2f}")
    


        
if __name__ == "__main__":
    main()