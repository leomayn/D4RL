import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple, Union
import time
import gym

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import flashbax as fbx
import wandb
import hydra
from omegaconf import OmegaConf
from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict
import matplotlib.pyplot as plt

from flax import serialization

from brax import envs

from utils.networks import ActorRNN

import imageio

class ScannedRNN(nn.Module):
    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        hidden_size = ins.shape[-1]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(hidden_size, *ins.shape[:-1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        return nn.GRUCell(hidden_size).initialize_carry(jax.random.PRNGKey(0), (*batch_size, hidden_size))


class AgentRNN(nn.Module):
    action_dim: int
    hidden_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(embedding)
        log_std = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(embedding)
        std = jnp.exp(log_std)  # Exponentiate to get the standard deviation

        return hidden, mean, std

class EpsilonGreedy:
    def __init__(self, start_e: float, end_e: float, duration: int):
        self.start_e = start_e
        self.end_e = end_e
        self.duration = duration
        self.slope = (end_e - start_e) / duration

    def get_epsilon(self, t: int):
        epsilon = self.start_e + self.slope * t
        return max(self.end_e, epsilon)

def policy_extractor(key, mean, exploration=True, epsilon=0.1):
    """
    Extract continuous actions based on the mean action with exploration noise.

    mean: Mean action predicted by the network.
    key: JAX random key for any stochastic operation (like sampling).
    exploration: Whether to apply exploration noise.
    epsilon: The scale of exploration noise (how much noise to add).

    Returns:
    Continuous actions of shape (batch_size, action_dim).
    """
    if exploration:
        # Apply Gaussian noise for exploration, scaled by epsilon
        action_noise = jax.random.normal(key, mean.shape) * epsilon
        actions = mean + action_noise
    else:
        # Simply use the mean as the deterministic action output
        actions = mean

    return actions

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
    
    agent = AgentRNN(action_dim=env.action_size, hidden_dim=config["AGENT_HIDDEN_DIM"], init_scale=config['AGENT_INIT_SCALE'])
    
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    
    key = jax.random.PRNGKey(0)
    key, key_r = jax.random.split(key)
    state = reset_fn(key_r)
    obs = state.obs
    done = state.done
    hstate = ScannedRNN.initialize_carry(config["AGENT_HIDDEN_DIM"], 1)
    
    def homogeneous_pass(params, hidden_state, obs, dones):
            hidden_state, mean, std = agent.apply(params, hidden_state, (obs, dones))
            return hidden_state, mean, std
        
    network_params = params
    rollout = [state.pipeline_state]
    timestep = 0
    # grab a trajectory
    frames = []
    # while not done:
    for i in range(config["NUM_STEPS"]):
        key, key_a = jax.random.split(key)
        obs = obs[np.newaxis, np.newaxis, :]
        done = jnp.array(done)[np.newaxis, np.newaxis]
        hstate, mean, std = homogeneous_pass(params, hstate, obs, done)
        action = mean.squeeze(0)
        action = action.squeeze()
        state = step_fn(state, action)
        done = state.done
        obs = state.obs
        rollout.append(state.pipeline_state)
        
    frames = env.render(trajectory=rollout, height=240, width=320)

    return frames

class Transition(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    infos: dict

def generate_reward(il_model, il_params, obs, action, done, il_h_state):
    in_data = (obs, done)
    il_h_state, pi = il_model.apply(il_params, il_h_state, in_data)
    reward = pi.log_prob(action)
    reward = jnp.squeeze(reward, axis=0)
    return (reward, il_h_state)

def plot_rewards(metrics, filename, num_seeds, alg_name, env_name):
    test_metrics = metrics["test_metrics"]
    test_returns = test_metrics["test_returns"]
    
    reward_mean = test_returns.mean(axis=0)
    reward_std = test_returns.std(axis=0) / np.sqrt(num_seeds)
    
    plt.plot(reward_mean)
    plt.fill_between(range(len(reward_mean)), reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.title(f"{env_name}_{alg_name}")
    plt.savefig(f'{filename}.png')

def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    
    print("num_updates: ", config["NUM_UPDATES"])
    
    env = envs.create(config["ENV_NAME"])
    test_env = envs.create(config["TEST_ENV_NAME"])
    
    # Beginning of IL
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

    # End of IL
    
    # epsilon greedy
    
    epsilon_greedy = EpsilonGreedy(start_e=config["EPSILON_START"], end_e=config["EPSILON_FINISH"], duration=config["EPSILON_ANNEAL_TIME"])

    def train(rng):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        
        reset_fn = jax.jit(jax.vmap(env.reset))
        step_fn = jax.jit(jax.vmap(env.step))
        
        test_reset_fn = jax.jit(jax.vmap(test_env.reset))
        test_step_fn = jax.jit(jax.vmap(test_env.step))
        
        reset_rngs = jax.random.split(_rng, config["NUM_ENVS"])
        env_state = reset_fn(reset_rngs)
        init_obs = env_state.obs
        init_dones = jnp.zeros((config["NUM_ENVS"]), dtype=bool)

        # INIT BUFFER
        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3)
            
            # Sample a random action from the environment's action space
            action = jax.random.uniform(key_a, shape=(config["NUM_ENVS"], env.action_size), minval=-1, maxval=1)
            
            # Step the environment with the sampled action
            env_state = step_fn(env_state, action)
            obs = env_state.obs
            rewards = env_state.reward
            dones = env_state.done
            infos = env_state.info
            
            transition = Transition(obs, action, rewards, dones, infos)
            return env_state, transition
        
        _, sample_traj = jax.lax.scan(_env_sample_step, env_state, None, config["NUM_STEPS"])
        sample_traj_unbatched = jax.tree_util.tree_map(lambda x: x[:, 0], sample_traj)
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config['BUFFER_SIZE']//config['NUM_ENVS'],
            min_length_time_axis=config['BUFFER_BATCH_SIZE'],
            sample_batch_size=config['BUFFER_BATCH_SIZE'],
            add_batch_size=config['NUM_ENVS'],
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_unbatched)

        # INIT NETWORK
        agent = AgentRNN(action_dim=env.action_size, hidden_dim=config["AGENT_HIDDEN_DIM"], init_scale=config['AGENT_INIT_SCALE'])
        rng, _rng = jax.random.split(rng)
        init_x = (jnp.zeros((1, 1, env.observation_size)), jnp.zeros((1, 1)))
        init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], 1)
        network_params = agent.init(_rng, init_hs, init_x)

        # INIT TRAIN STATE AND OPTIMIZER
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=config['LR'], eps=config['EPS_ADAM']),
        )
        train_state = TrainState.create(
            apply_fn=agent.apply,
            params=network_params,
            tx=tx,
        )
        target_agent_params = jax.tree_util.tree_map(lambda x: jnp.copy(x), train_state.params)

        # HOMOGENEOUS PASS FUNCTION
        def homogeneous_pass(params, hidden_state, obs, dones):
            hidden_state, mean, std = agent.apply(params, hidden_state, (obs, dones))
            return hidden_state, mean, std

        # TRAINING LOOP
        def _update_step(runner_state, unused):
            train_state, target_agent_params, env_state, buffer_state, time_state, init_obs, init_dones, test_metrics, rng = runner_state

            def _env_step(step_state, unused):
                params, env_state, last_obs, last_dones, hstate, il_h_state, rng, t = step_state
                rng, key_a, key_s = jax.random.split(rng, 3)
                
                obs_ = last_obs[np.newaxis, :]
                dones_ = last_dones[np.newaxis, :]

                # SELECT ACTION
                hstate, mean, std = homogeneous_pass(params, hstate, obs_, dones_)
                epsilon = epsilon_greedy.get_epsilon(t)
                action = policy_extractor(key_a, mean, exploration=True, epsilon=epsilon).squeeze(0)
                reward, il_h_state = generate_reward(il_model, il_params, obs_, action, dones_, il_h_state)

                # STEP ENV
                env_state = step_fn(env_state, action)
                obs = env_state.obs
                done = env_state.done
                info = env_state.info
                transition = Transition(last_obs, action, reward, done, info)  # Empty info dict

                step_state = (params, env_state, obs, done.astype(bool), hstate, il_h_state, rng, t+1)
                return step_state, transition

            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], config["NUM_ENVS"])
            il_h_state = ScannedRNN.initialize_carry(config['IL_NETWORK_HIDDEN'], config["NUM_ENVS"])

            step_state = (
                train_state.params,
                env_state,
                init_obs,
                init_dones,
                hstate,
                il_h_state, 
                _rng,
                time_state['timesteps']
            )

            step_state, traj_batch = jax.lax.scan(_env_step, step_state, None, config["NUM_STEPS"])

            buffer_traj_batch = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 0, 1)[:, np.newaxis], 
                traj_batch
            )
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            # LEARNING PHASE
            def _loss_fn(params, target_agent_params, init_hs, learn_traj):
                obs_ = learn_traj.obs
                dones_ = learn_traj.dones
                hstate, mean, std = homogeneous_pass(params, init_hs, obs_, dones_)
                _, target_mean, target_std = homogeneous_pass(target_agent_params, init_hs, obs_, dones_)

                # Compute value estimates (e.g., using critic network)
                # Since DQN is not directly applicable to continuous actions,
                # you may need to adjust this to use a suitable algorithm like DDPG, TD3, or SAC

                # Placeholder loss (since DQN isn't suitable for continuous actions)
                loss = jnp.array(0.0)

                return loss

            rng, _rng = jax.random.split(rng)
            learn_traj = buffer.sample(buffer_state, _rng).experience
            learn_traj = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x[:, 0], 0, 1), 
                learn_traj
            )
            init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], config["BUFFER_BATCH_SIZE"])

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=False)
            loss, grads = grad_fn(train_state.params, target_agent_params, init_hs, learn_traj)
            train_state = train_state.apply_gradients(grads=grads)

            rng, _rng = jax.random.split(rng)
            reset_rngs = jax.random.split(_rng, config["NUM_ENVS"])
            env_state = reset_fn(reset_rngs)
            init_dones = jnp.zeros((config["NUM_ENVS"]), dtype=bool)

            time_state['timesteps'] = step_state[-1]
            time_state['updates'] += 1

            target_agent_params = jax.lax.cond(
                time_state['updates'] % config['TARGET_UPDATE_INTERVAL'] == 0,
                lambda _: jax.tree_util.tree_map(lambda x: jnp.copy(x), train_state.params),
                lambda _: target_agent_params,
                operand=None
            )

            # TESTING AND METRICS
            rng, _rng = jax.random.split(rng)
            test_metrics = jax.lax.cond(
                time_state['updates'] % (config["TEST_INTERVAL"] // config["NUM_STEPS"] // config["NUM_ENVS"]) == 0,
                lambda _: get_greedy_metrics(_rng, train_state.params, time_state),
                lambda _: test_metrics,
                operand=None
            )

            # update the returning metrics
            metrics = {
                'timesteps': time_state['timesteps']*config['NUM_ENVS'],
                'updates' : time_state['updates'],
                'loss': loss,
                'rewards': traj_batch.rewards.sum(),
            }
            metrics['test_metrics'] = test_metrics # add the test metrics dictionary

            if config.get('WANDB_ONLINE_REPORT', False):
                def callback(metrics, infos):
                    wandb.log(
                        {
                            "returns": metrics['rewards'],
                            "timestep": metrics['timesteps'],
                            "updates": metrics['updates'],
                            "loss": metrics['loss'],
                        }
                    )
                jax.debug.callback(callback, metrics, traj_batch.infos)

            runner_state = (
                train_state,
                target_agent_params,
                env_state,
                buffer_state,
                time_state,
                init_obs,
                init_dones,
                test_metrics,
                rng
            )

            return runner_state, metrics

        def get_greedy_metrics(rng, params, time_state):
            """Help function to test greedy policy during training"""
            def _greedy_env_step(step_state, unused):
                params, env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                obs_ = last_obs[np.newaxis, :]
                dones_ = last_dones[np.newaxis, :]
                hstate, mean, std = homogeneous_pass(params, hstate, obs_, dones_)
                # Use mean as the action for deterministic evaluation
                action = mean.squeeze(0)
                env_state = test_step_fn(env_state, action)
                step_state = (params, env_state, env_state.obs, env_state.done.astype(bool), hstate, rng)
                return step_state, (env_state.reward, env_state.done, env_state.info)
            rng, _rng = jax.random.split(rng)
            reset_rngs = jax.random.split(_rng, config["NUM_TEST_EPISODES"])
            env_state = test_reset_fn(reset_rngs)
            init_dones = jnp.zeros((config["NUM_TEST_EPISODES"]), dtype=bool)
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], config["NUM_TEST_EPISODES"])
            step_state = (
                params,
                env_state,
                env_state.obs,
                init_dones,
                hstate, 
                _rng,
            )
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["NUM_STEPS"]
            )
            first_returns = rewards.sum()
            metrics = {
                'test_returns': first_returns,# episode returns
            }
            if config.get('VERBOSE', False):
                def callback(timestep, val):
                    print(f"Timestep: {timestep}, return: {val}")
                jax.debug.callback(callback, time_state['timesteps']*config['NUM_ENVS'], first_returns.mean())
            return metrics

        time_state = {
            'timesteps':jnp.array(0),
            'updates':  jnp.array(0)
        }
        rng, _rng = jax.random.split(rng)
        test_metrics = get_greedy_metrics(_rng, train_state.params, time_state)
        
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            target_agent_params,
            env_state,
            buffer_state,
            time_state,
            init_obs,
            init_dones,
            test_metrics,
            _rng
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {'runner_state': runner_state, 'metrics': metrics}
    
    return train



# debug functions
def debug_plot_rewards(metrics, filename, num_seeds):
    test_metrics = metrics["test_metrics"]
    print(f"Test Metrics: {test_metrics}")  # Debug print to see the full structure of test metrics

    test_returns = test_metrics["test_returns"]
    print(f"Test Returns (before reshape): {test_returns.shape}")  # Debug to check test returns before reshaping

    reward_mean = test_returns.mean(axis=0)
    reward_std = test_returns.std(axis=0) / np.sqrt(num_seeds)

    print(f"Reward Mean: {reward_mean.shape}")  # Debug to check reward mean
    print(f"Reward Std: {reward_std.shape}")    # Debug to check reward standard deviation

    plt.plot(reward_mean)
    plt.fill_between(range(len(reward_mean)), reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.savefig(f'{filename}.png')

@hydra.main(version_base=None, config_path="./config", config_name="dqn_config")
def main(config):
    config = OmegaConf.to_container(config)

    print('Config:\n', OmegaConf.to_yaml(config))

    env_name = config["ENV_NAME"]
    
    config["NUM_STEPS"] = gym.make(config["GYM_NAME"])._max_episode_steps

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            "DQN",
            env_name.upper(),
            "RNN",
            f"jax_{jax.__version__}",
        ],
        name=f'dqn_{env_name}',
        config=config,
        # mode=config["WANDB_MODE"],
        mode='disabled',
    )
    
    
    
    if config["TRAIN"]:
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        
        start_time = time.time()
        outs = jax.block_until_ready(train_vjit(rngs))
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Time for training: {total_time:.2f}")
        
        if config['SAVE_PATH'] is not None:
            def save_params(params: dict, filename: Union[str, os.PathLike]) -> None:
                flattened_dict = flatten_dict(params, sep=',')
                save_file(flattened_dict, filename)

            model_state = outs['runner_state'][0]
            params = jax.tree_util.tree_map(lambda x: x[0], model_state.params)
            save_dir = os.path.join(config['SAVE_PATH'], env_name)
            os.makedirs(save_dir, exist_ok=True)
            save_params(params, f'{save_dir}/dqn.safetensors')
            print(f'Parameters of first batch saved in {save_dir}/dqn.safetensors')
            
        plot_rewards(outs["metrics"], f'{env_name}_dqn', config["NUM_SEEDS"], config["ENV_NAME"], config["ALG_NAME"])
    elif config["DEBUG"]:
        config["TOTAL_TIMESTEPS"] = config["DEBUG_STEPS"]
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        
        start_time = time.time()
        outs = jax.block_until_ready(train_vjit(rngs))
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Time for training: {total_time:.2f}")
        
        debug_plot_rewards(outs["metrics"], f'{env_name}_dqn', config["NUM_SEEDS"])
    else:
        # Load model parameters
        def load_params(filename: Union[str, os.PathLike]) -> dict:
            # Load the flattened dictionary from file
            flattened_dict = load_file(filename)  # Assuming you have a load_file function
            # Reconstruct the tree structure
            params = unflatten_dict(flattened_dict, sep=',')
            return params
        save_dir = os.path.join(config['SAVE_PATH'], env_name)    
        filename = f'{save_dir}/dqn.safetensors'
        params = load_params(filename)
    
    if config["RENDER"]:
        start_time = time.time()
        frames = get_rollout(params, config)
        save_video(frames, f'{config["ENV_NAME"]}_{config["ALG_NAME"]}.mp4')
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Time for rendering: {total_time:.2f}")

if __name__ == "__main__":
    main()
