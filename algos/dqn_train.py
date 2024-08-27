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
import wandb

from utils.networks import AgentRNN, ActorRNN, ScannedRNN
from jax_rl.algorithms import DQN

import gym
import d4rl

from brax import envs

from jax_rl.utils import Transition

def generate_reward(il_model, il_params, obs, action, done):
    in_data = (obs, done)
    s_h_state, pi = il_model.apply(il_params, s_h_state, in_data)
    return pi.log_prob(action)

def make_train(config):
    env = envs.get_environment(config["ENV_NAME"])
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))
    
    # Initialize DQN agent
    dqn_agent = DQN(key=jax.random.PRNGKey(0),
                    n_states=env.observation_space.shape[0],
                    n_actions=env.action_space.n,
                    gamma=config['GAMMA'],
                    buffer_size=config['BUFFER_SIZE'],
                    model=AgentRNN(action_dim=env.action_space.n, hidden_dim=config["HIDDEN_DIM"]),
                    lr=config['LR'])
    
    # Load IL model parameters
    il_model = ActorRNN(action_dim=gym.make(config["ENV_NAME"]).action_space.shape[0], config=config)
    rng = jax.random.PRNGKey(0)
    _s_init_x = (
        jnp.zeros((1, 1, gym.make(config["ENV_NAME"]).observation_space.shape[0])),
        jnp.zeros((1, 1))
    )
    _s_init_h = ScannedRNN.initialize_carry(1, config["STUDENT_NETWORK_HIDDEN"])

    init_params = il_model.init(rng, _s_init_h, _s_init_x)

    params_path = os.path.join(config["STUDENT_NETWORK_SAVE_PATH"], 'final.msgpack')
    if os.path.exists(params_path):
        with open(params_path, "rb") as f:
            params_bytes = f.read()
        il_params = serialization.from_bytes(init_params, params_bytes)
        print(f"Loaded parameters from {params_path}")
    else:
        print(f"Parameters file not found at {params_path}. Please check the path and try again.")
        return

    def train(rng):
        reset_rng, rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
        env_state  = reset_fn(reset_rngs)
        def episode_step(episode_state, episode_idx):
            state, rng, episode_rewards, best_test_return = episode_state
            total_reward = 0
            rng, step_rng = jax.random.split(rng)

            def step(step_state, t):
                state, rng, total_reward = step_state
                prev_obs = state.obs
                prev_done = state.done
                action = dqn_agent.act(jnp.array(prev_obs)[jnp.newaxis, :], exploration=True)
                next_state = step_fn(state, action)
                done = next_state.done
                obs = next_state.obs
                reward = generate_reward(il_model, il_params, prev_obs, action, prev_done)
                
                dqn_agent.update_buffer(Transition(s=state, a=action, r=reward, d=done, s_next=obs))
                loss = dqn_agent.train(batch_size=config['BATCH_SIZE'])
                
                total_reward += reward
                new_state = next_state if not done else env.reset()

                return (new_state, rng, total_reward), (state, reward, done, loss)

            # Using jax.lax.scan to iterate over timesteps
            (final_state, _, total_reward), _ = jax.lax.scan(step, (state, step_rng, 0), jnp.arange(config['MAX_TIMESTEPS']))

            episode_rewards = episode_rewards.at[episode_idx].set(total_reward)

            # Update target network periodically
            def update_target_fn(val):
                dqn_agent.update_target()
                return val

            update_condition = (episode_idx % config['TARGET_UPDATE_FREQ']) == 0
            episode_rewards = jax.lax.cond(update_condition, update_target_fn, lambda x: x, episode_rewards)

            # Log to wandb periodically
            def log_wandb_fn(val):
                print(f'Episode {episode_idx}: Total Reward = {total_reward}, Avg Reward = {jnp.mean(val[-100:])}')
                wandb.log({"Total Reward": total_reward, "Avg Reward (last 100)": jnp.mean(val[-100:])})
                return val

            log_condition = (episode_idx % config['LOG_FREQ']) == 0
            episode_rewards = jax.lax.cond(log_condition, log_wandb_fn, lambda x: x, episode_rewards)

            # Test the model periodically
            def test_and_save_model_fn(val):
                test_return = test_model(dqn_agent, dqn_agent.params, config, rng)
                new_best_test_return = jnp.maximum(best_test_return, test_return)
                wandb.log({"Test Return": test_return}, step=episode_idx)
                
                def save_model_fn(val):
                    save_model(dqn_agent.params, config["MODEL_SAVE_PATH"], 'best_model.msgpack')
                    return val
                
                save_condition = test_return > best_test_return
                jax.lax.cond(save_condition, save_model_fn, lambda x: x, val)
                return new_best_test_return

            test_condition = (episode_idx % config['TEST_INTERVAL']) == 0
            best_test_return = jax.lax.cond(test_condition, test_and_save_model_fn, lambda x: x, best_test_return)

            return (final_state, rng, episode_rewards, best_test_return), total_reward

        episode_rewards = jnp.zeros(config['NUM_EPISODES'])
        best_test_return = float('-inf')

        (final_state, _, episode_rewards, best_test_return), _ = jax.lax.scan(
            episode_step, 
            (env_state, rng, episode_rewards, best_test_return), 
            jnp.arange(config['NUM_EPISODES'])
        )

        return episode_rewards

    def test_model(dqn_agent, params, config, rng):
        test_env = envs.get_environment(config["TEST_ENV_NAME"])
        reset_fn = jax.jit(jax.vmap(test_env.reset))
        step_fn = jax.jit(jax.vmap(test_env.step))
        reset_rng, rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, config["NUM_TEST_ENVS"])
        test_state  = reset_fn(reset_rngs)
        total_test_reward = 0

        def _test_step(test_state, t):
            state, rng, total_reward = test_state
            prev_obs = state.obs
            action = dqn_agent.act(jnp.array(prev_obs)[jnp.newaxis, :], exploration=False)
            next_state = step_fn(state, action)
            done = state.done
            obs = state.obs
            reward = state.reward
            total_reward += reward

            return (next_state if not done else test_env.reset(), rng, total_reward), (next_state, reward, done)

        (final_state, _, total_test_reward), _ = jax.lax.scan(_test_step, (test_state, rng, 0), jnp.arange(config['MAX_TEST_STEPS']))

        return total_test_reward

    def save_model(params, save_path, filename):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, filename)
        with open(file_path, "wb") as f:
            f.write(serialization.to_bytes(params))
        print(f"Model saved to {file_path}")

    return train

@hydra.main(version_base=None, config_path="config", config_name="dqn_config")
def main(config):
    config = OmegaConf.to_container(config)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT_PREFIX"] + config["ENV_NAME"] + "_DQN",
        tags=["DQN"],
        config=config,
        mode=config["WANDB_MODE"],
    )
    
    train = make_train(config)
    rng = jax.random.PRNGKey(config["SEED"])
    rewards = train(rng)
    print("Training finished")
    
    # Save the final model after training
    # save_model(dqn_agent.params, config["MODEL_SAVE_PATH"], 'dqn_final.msgpack')

if __name__ == "__main__":
    main()
