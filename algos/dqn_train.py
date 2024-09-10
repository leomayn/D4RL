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

from typing import Sequence, Dict

import imageio

import distrax

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


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        
        if len(x) == 3:
            obs, dones, avail_actions = x
        else:
            obs, dones = x
            avail_actions = jnp.ones((obs.shape[0], obs.shape[1], self.action_dim))
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        

        
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        
        # unavail_actions = 1 - avail_actions
        # action_logits = actor_mean - (unavail_actions * 1e10)
        
        actor_log_std = self.param('actor_log_std', lambda key, shape: -0.5 * jnp.ones(shape), (self.action_dim,))
        action_std = jnp.exp(actor_log_std)

        # pi = distrax.Categorical(logits=action_logits)
        pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=action_std)

        return hidden, pi

class Qnetwork(nn.Module):
    action_dim: int
    hidden_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, obs, action):
        # Concatenate the observation and action
        x = jnp.concatenate([obs, action], axis=-1)
        # Pass the concatenated input through the network
        embedding = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(x)
        embedding = nn.relu(embedding)
        
        # Compute Q-values
        q_vals = nn.Dense(1, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(embedding)

        return jnp.squeeze(q_vals, axis=-1)


    
class Vnetwork(nn.Module):
    hidden_dim: int  # Add a hidden dimension for consistency

    @nn.compact
    def __call__(self, x):
        world_state = x
        embedding = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(world_state)
        embedding = nn.relu(embedding)
        
        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        
        return jnp.squeeze(critic, axis=-1)


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
    
    policy_actor = ActorRNN(action_dim=env.action_size, config=config)
    
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    
    key = jax.random.PRNGKey(0)
    key, key_r = jax.random.split(key)
    state = reset_fn(key_r)
    obs = state.obs
    done = state.done
    hstate = ScannedRNN.initialize_carry(config["AGENT_HIDDEN_DIM"], 1)
        
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
        hstate, pi = policy_actor.apply(params, hstate, (obs, done))
        action = pi.sample(seed=key_a)[0]
        state = step_fn(state, action)
        done = state.done
        obs = state.obs
        rollout.append(state.pipeline_state)
        
    frames = env.render(trajectory=rollout, height=240, width=320)

    return frames

class Transition(NamedTuple):
    obs: jnp.ndarray
    next_obs: jnp.ndarray
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
    rng, _s_init_rng = jax.random.split(rng)
    _s_init_x = (
        jnp.zeros((1, 1, env.observation_size)),
        jnp.zeros((1, 1))
    )
    _s_init_h = ScannedRNN.initialize_carry(config["IL_NETWORK_HIDDEN"], config["NUM_ENVS"])

    init_params = il_model.init(_s_init_rng, _s_init_h, _s_init_x)
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
    
    # initalize networks
    
    # INIT NETWORK
    policy_actor = ActorRNN(action_dim=env.action_size, config=config)
    # q network, v network
    q_network = Qnetwork(action_dim=env.action_size, hidden_dim=config["AGENT_HIDDEN_DIM"], init_scale=config['AGENT_INIT_SCALE'])
    v_network = Vnetwork(hidden_dim=config["AGENT_HIDDEN_DIM"])

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
            last_obs = env_state.obs
            
            # Sample a random action from the environment's action space
            action = jax.random.uniform(key_a, shape=(config["NUM_ENVS"], env.action_size), minval=-1, maxval=1)
            
            # Step the environment with the sampled action
            env_state = step_fn(env_state, action)
            obs = env_state.obs
            rewards = env_state.reward
            dones = env_state.done
            infos = env_state.info
            
            transition = Transition(last_obs, obs, action, rewards, dones, infos)
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

        rng, _rng = jax.random.split(rng)
        init_x = (jnp.zeros((1, 1, env.observation_size)), jnp.zeros((1, 1)))
        init_hs = ScannedRNN.initialize_carry(config["POLICY_HIDDEN"], 1)
        network_params = policy_actor.init(_rng, init_hs, init_x)
        
        rng, q_rng = jax.random.split(rng)
        rng, v_rng = jax.random.split(rng)
        dummy_obs = jnp.zeros((1, 1, env.observation_size))
        dummy_action = jnp.zeros((1, 1, env.action_size))
        q_params = q_network.init(q_rng, dummy_obs, dummy_action)
        v_params = v_network.init(v_rng, dummy_obs)

        # INIT TRAIN STATE AND OPTIMIZER
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=config['LR'], eps=config['EPS_ADAM']),
        )
        train_state = TrainState.create(
            apply_fn=policy_actor.apply,
            params=network_params,
            tx=tx,
        )
        target_agent_params = jax.tree_util.tree_map(lambda x: jnp.copy(x), q_params)

        # TRAINING LOOP
        def _update_step(runner_state, unused):
            train_state, q_params, v_params, target_agent_params, env_state, buffer_state, time_state, init_obs, init_dones, test_metrics, rng = runner_state

            def _env_step(step_state, unused):
                params, env_state, last_obs, last_dones, hstate, il_h_state, rng, t = step_state
                rng, key_a, key_s = jax.random.split(rng, 3)
                
                obs_ = last_obs[np.newaxis, :]
                dones_ = last_dones[np.newaxis, :]

                # SELECT ACTION
                hstate, pi = policy_actor.apply(params, hstate, (obs_, dones_))
                action = pi.sample(seed=key_a)[0]
                reward, il_h_state = generate_reward(il_model, il_params, obs_, action, dones_, il_h_state)

                # STEP ENV
                env_state = step_fn(env_state, action)
                obs = env_state.obs
                done = env_state.done
                info = env_state.info
                transition = Transition(last_obs, obs, action, reward, done, info)  # Empty info dict

                step_state = (params, env_state, obs, done.astype(bool), hstate, il_h_state, rng, t+1)
                return step_state, transition

            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(config["POLICY_HIDDEN"], config["NUM_ENVS"])
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
            
            def v_function_loss(v_params, params, obs, actions, dones, target_agent_params, expectile):
                """Compute the loss for the V-function using expectile regression."""
                # V-function output: V(s) = E[Q(s, a)]
                v_values = v_network.apply(v_params, obs)
                target_values = q_network.apply(target_agent_params, obs, actions)
                # Asymmetric expectile regression loss
                diff = target_values - v_values
                loss = jnp.where(diff > 0, expectile * diff**2, (1 - expectile) * diff**2)
                
                return jnp.mean(loss)
            
            def q_function_loss(q_params, target_agent_params, obs, actions, rewards, dones, next_obs, discount, v_params):
                """Compute the loss for the Q-function."""
                q_values = q_network.apply(q_params, obs, actions)  # Q-function values
                
                # Get V-values for the next state (using target V network or current network)
                next_v_values = v_network.apply(v_params, next_obs)
                
                # Bellman target for the Q-function
                bellman_target = rewards + discount * next_v_values * (1 - dones)
                
                # TD error
                td_error = q_values - bellman_target
                loss = jnp.mean(td_error**2)
                
                return loss
            
            def advantage_weighted_regression_loss(policy_actor, params, obs, actions, dones, h_state, advantages):
                """AWR loss for policy extraction"""
                # Get log probabilities of the actions under the current policy
                h_state, pi = policy_actor.apply(params, h_state, (obs, dones))
                log_probs = pi.log_prob(actions)
                
                # Calculate the weighted log-likelihood loss
                weights = jnp.exp(advantages)  # Exponentiate the advantages to form the weights
                loss = -jnp.mean(weights * log_probs)  # Negative log-likelihood weighted by the advantages
                
                return loss
            
            def compute_advantage(q_values, v_values):
                """Compute the advantage as A(s, a) = Q(s, a) - V(s)"""
                return q_values - v_values


            def _loss_fn(params, target_agent_params, init_hs, learn_traj, expectile, discount, v_params, q_params):
                obs_ = learn_traj.obs
                actions_ = learn_traj.actions
                rewards_ = learn_traj.rewards
                dones_ = learn_traj.dones
                next_obs_ = learn_traj.next_obs

                # Q-function loss
                q_loss = q_function_loss(
                    q_params,
                    target_agent_params,
                    obs_,
                    actions_,
                    rewards_,
                    dones_,
                    next_obs_,
                    discount,
                    v_params
                )

                # V-function loss (expectile regression)
                q_vals = q_network.apply(q_params, obs_, actions_)
                v_loss = v_function_loss(
                    v_params,
                    q_vals,
                    obs_,
                    actions_,
                    dones_,
                    target_agent_params,
                    expectile
                )

                # Compute advantage: A(s, a) = Q(s, a) - V(s)
                v_values = v_network.apply(v_params, obs_)
                advantages = compute_advantage(q_vals, v_values)

                # Policy loss (AWR)
                awr_loss = advantage_weighted_regression_loss(
                    policy_actor,  # Replace with your policy network (if it's different)
                    params,
                    obs_,
                    actions_,
                    dones_,
                    init_hs,
                    advantages
                )

                # Combine all losses
                total_loss = q_loss + v_loss + awr_loss

                return total_loss



            rng, _rng = jax.random.split(rng)
            learn_traj = buffer.sample(buffer_state, _rng).experience
            learn_traj = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x[:, 0], 0, 1), 
                learn_traj
            )
            init_hs = ScannedRNN.initialize_carry(config["POLICY_HIDDEN"], config["BUFFER_BATCH_SIZE"])

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=False)
            loss, grads = grad_fn(train_state.params, target_agent_params, init_hs, learn_traj, config['EXPECTILE'], config['GAMMA'], v_params, q_params)
            train_state = train_state.apply_gradients(grads=grads)

            rng, _rng = jax.random.split(rng)
            reset_rngs = jax.random.split(_rng, config["NUM_ENVS"])
            env_state = reset_fn(reset_rngs)
            init_dones = jnp.zeros((config["NUM_ENVS"]), dtype=bool)

            time_state['timesteps'] = step_state[-1]
            time_state['updates'] += 1

            target_agent_params = jax.lax.cond(
                time_state['updates'] % config['TARGET_UPDATE_INTERVAL'] == 0,
                lambda _: jax.tree_util.tree_map(lambda x: jnp.copy(x), q_params),
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
                q_params,
                v_params,
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
                rng, key_a = jax.random.split(rng)
                obs_ = last_obs[np.newaxis, :]
                dones_ = last_dones[np.newaxis, :]
                hstate, pi = policy_actor.apply(params, hstate, (obs_, dones_))
                action = pi.sample(seed=key_a)[0]
                env_state = test_step_fn(env_state, action)
                step_state = (params, env_state, env_state.obs, env_state.done.astype(bool), hstate, rng)
                return step_state, (env_state.reward, env_state.done, env_state.info)
            rng, _rng = jax.random.split(rng)
            reset_rngs = jax.random.split(_rng, config["NUM_TEST_EPISODES"])
            env_state = test_reset_fn(reset_rngs)
            init_dones = jnp.zeros((config["NUM_TEST_EPISODES"]), dtype=bool)
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(config["POLICY_HIDDEN"], config["NUM_TEST_EPISODES"])
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
            q_params,
            v_params,
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
