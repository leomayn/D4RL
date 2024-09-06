import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple, Dict, Union
import haiku as hk

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

import chex
import optax
from flax import serialization
from flax.training.train_state import TrainState
import wandb
import hydra
from omegaconf import OmegaConf
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict

from jax_rl.algorithms import DQN
from utils.networks import AgentRNN, ActorRNN, ScannedRNN
from jax_rl.utils import Transition
from brax import envs


def save_model(params, save_path, filename):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, filename)
    with open(file_path, "wb") as f:
        f.write(serialization.to_bytes(params))
    print(f"Model saved to {file_path}")

def generate_reward(il_model, il_params, obs, action, done, il_h_state):
    obs_ = obs[np.newaxis, :]  # Add time_step dimension
    dones_ = done[np.newaxis, :]  # Add time_step dimension
    print("obs:", obs_.shape)
    print("dones:", dones_.shape)
    print("Action shape in generate_reward before log_prob:", action.shape)
    in_data = (obs_, dones_)
    il_h_state, pi = il_model.apply(il_params, il_h_state, in_data)
    return (pi.log_prob(action), il_h_state)

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
    
    student_tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(learning_rate=optax.linear_schedule(init_value=config["LR"], end_value=1e-4, transition_steps=config["NUM_EPOCHS"] * config["DATASET_SIZE"] / config["BATCH_SIZE"]), eps=1e-5),
    )
    student_train_state = TrainState.create(
        apply_fn=student_model.apply,
        params=s_network_params,
        tx=student_tx,
    )

    def train(rng):
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

                return (new_state, rng, total_reward, il_h_state, hidden_state), (state, reward, done, loss)

            (final_state, _, total_reward, _, _), _ = jax.lax.scan(env_step, (state, step_rng, total_reward, _s_init_h, hidden_state), jnp.arange(config['NUM_STEPS']))

            episode_rewards = episode_rewards.at[episode_idx].set(jnp.mean(total_reward))

            def update_target_fn(val):
                dqn_agent.update_target()
                return val

            update_condition = (episode_idx % config['TARGET_UPDATE_INTERVAL']) == 0
            episode_rewards = jax.lax.cond(update_condition, update_target_fn, lambda x: x, episode_rewards)

            def log_wandb_fn(val):
                print(f'Episode {episode_idx}: Total Reward = {total_reward}, Avg Reward = {jnp.mean(val[-100:])}')
                wandb.log({"Total Reward": total_reward, "Avg Reward (last 100)": jnp.mean(val[-100:])})
                return val

            log_condition = (episode_idx % config['LOG_FREQ']) == 0
            episode_rewards = jax.lax.cond(log_condition, log_wandb_fn, lambda x: x, episode_rewards)

            def test_and_save_model_fn(val):
                test_return = test_model(dqn_agent, dqn_agent.params, config, rng)
                print(f"Test return: {test_return}")
                wandb.log({"Test Return": test_return}, step=episode_idx)
                new_best_test_return = jnp.maximum(best_test_return, test_return)

                def save_model_fn(val):
                    save_model(dqn_agent.params, config["MODEL_SAVE_PATH"], 'best_model.msgpack')
                    return val

                save_condition = test_return > best_test_return
                jax.lax.cond(save_condition, save_model_fn, lambda x: x, val)

                return new_best_test_return

            test_condition = (episode_idx % config['TEST_INTERVAL']) == 0
            best_test_return = jax.lax.cond(test_condition, test_and_save_model_fn, lambda x: x, best_test_return)

            return (final_state, rng, episode_rewards, best_test_return), total_reward

        episode_rewards = jnp.zeros(config['NUM_UPDATES'])
        best_test_return = float('-inf')

        (final_state, _, episode_rewards, best_test_return), _ = jax.lax.scan(
            update_step, 
            (env_state, rng, episode_rewards, best_test_return), 
            jnp.arange(config['NUM_UPDATES'])
        )

        return episode_rewards

    def test_model(dqn_agent, params, config, rng):
        test_env = envs.get_environment(config["TEST_ENV_NAME"])

        # Initialize the environment and hidden states
        reset_fn = jax.jit(test_env.reset)
        step_fn = jax.jit(test_env.step)

        key = jax.random.PRNGKey(0)
        key, key_r = jax.random.split(key)
        
        # Reset the environment
        test_state = reset_fn(key_r)
        obs = test_state.obs
        done = test_state.done
        
        hidden_state = ScannedRNN.initialize_carry(1, config['AGENT_HIDDEN_DIM'])  # Assuming single test environment
        network_params = params
        total_test_reward = 0
        
        # Run a rollout
        for i in range(config["NUM_TEST_STEPS"]):  # or use config["MAX_TEST_STEPS"]
            key, key_a = jax.random.split(key)
            obs = obs[np.newaxis, np.newaxis, :]  # Add dimensions for time step and batch
            done = jnp.array(done)[np.newaxis, np.newaxis]
            
            # Get actions from the policy
            ins = (obs, done)
            hidden_state, pi = dqn_agent.q_network.apply(network_params, hidden_state, ins)
            actions = pi.sample(seed=key_a)
            actions = jnp.squeeze(actions)

            # Step the environment
            test_state = step_fn(test_state, actions)
            done = test_state.done
            obs = test_state.obs
            reward = test_state.reward

            # Accumulate rewards
            total_test_reward += reward

        avg_test_reward = total_test_reward.sum()
        
        # Return the average test reward
        return avg_test_reward

    return train


@hydra.main(version_base=None, config_path="config", config_name="dqn_config")
def main(config):
    config = OmegaConf.to_container(config)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT_PREFIX"] + config["ENV_NAME"] + "_DQN",
        tags=["DQN"],
        config=config,
        # mode=config["WANDB_MODE"],
        mode='disabled',
    )
    
    train = make_train(config)
    rng = jax.random.PRNGKey(config["SEED"])
    outs = train(rng)
    print("Training finished")
    
    # Save the final model after training
    save_model(outs['runner_state'][0].params, config["MODEL_SAVE_PATH"], 'dqn_final.msgpack')


if __name__ == "__main__":
    main()
