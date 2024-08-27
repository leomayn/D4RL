
from flax import linen as nn
import functools
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
import numpy as np
import distrax
from typing import Sequence, Dict
class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(carry.shape[0], carry.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

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


class CriticRNN(nn.Module):
    
    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            hidden.shape[1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        embedding = nn.relu(embedding)
        
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        
        return hidden, jnp.squeeze(critic, axis=-1)
    
class ActorCriticFF(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)
    

class RewardModel(nn.Module):
    action_dim: int
    hidden_size: int = 256
    action_embedding_size: int = 64
    obs_embedding_size: int = 64

    @nn.compact
    def __call__(self, hidden, trajectories):
        action = trajectories.action
        obs = trajectories.obs  # (max_seq_len, batch_size, obs_dim)
        dones = trajectories.done
        world_state = trajectories.world_state

        # change action into one-hot encoding
        action = jax.nn.one_hot(action, self.action_dim)

        embedded_action = nn.Dense(
            self.action_embedding_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(action)
        embedded_obs = nn.Dense(
            self.obs_embedding_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)

        x = jnp.concatenate([embedded_action, embedded_obs, world_state], axis=-1)

        embedding = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        prediction_layer = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        prediction_layer = nn.relu(prediction_layer)

        prediction = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(prediction_layer)
        prediction = jnp.squeeze(prediction, axis=-1)
        return hidden, prediction
    
class RewardModelFF(nn.Module):
    action_dim: int
    layer_dim: int = 128 # 64
    action_embedding_size: int = 64
    obs_embedding_size: int = 64
    
    @nn.compact
    def __call__(self, unused_hidden, trajectories):
        action = trajectories.action
        obs = trajectories.obs  # (max_seq_len, batch_size, obs_dim)

        action = nn.one_hot(action, self.action_dim)  # (max_seq_len, batch_size, action_dim)
        embedded_action = nn.Dense(
            features=self.action_embedding_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(action)
        embedded_action = nn.relu(embedded_action)
        embedded_action = nn.Dense(
            features=self.action_embedding_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(action)
        embedded_action = nn.relu(embedded_action)

        embedded_obs = nn.Dense(
            features=self.obs_embedding_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(obs)
        embedded_obs = nn.relu(embedded_obs)
        embedded_obs = nn.Dense(
            features=self.obs_embedding_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(obs)
        embedded_obs = nn.relu(embedded_obs)

        x = jnp.concatenate([embedded_action, embedded_obs], axis=-1)

        embedding = nn.Dense(
            features=self.layer_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        embedding = nn.Dense(
            features=self.layer_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)    # deeper layer added
        embedding = nn.relu(embedding)
        prediction = nn.Dense(
            features=1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(embedding)
        prediction = jnp.squeeze(prediction, axis=-1)
        return unused_hidden, prediction
 
class AgentRNN(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    init_scale: float
    config: dict = None
    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        q_vals = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(embedding)

        return hidden, q_vals
    
import jax
import jax.numpy as jnp
import numpy as np 
import haiku as hk 
import optax

from functools import partial

from jax_rl.algorithms import BaseAgent
from jax_rl.buffer import ExperienceReplay
from jax_rl.utils import Transition

class DQN(BaseAgent):
	''' Deep Q-network'''

	def __init__(self, key, n_states, n_actions, gamma, buffer_size, policy, model, lr):
		'''
		model must take sequential inputs and a hidden state.
		init_state must provide the initial state for a given batch_size.
		'''
		super(DQN, self).__init__(key, n_states, n_actions, gamma)

		self.buffer = ExperienceReplay(buffer_size)
		self.policy = policy

		# Q-network and parameters
		self.params = model.init(next(self.prng), jnp.ones((1, n_states)))
		self.update_target()
		self.q_network = model.apply

		# optimiser
		self.opt_update, self.opt_state = self.init_optimiser(lr, self.params)

	def act(self, s, exploration = True):
		''' Get an action from the q-network given the state.
		s - torch.FloatTensor shape (1, 1, n_states) current state
		exploration - bool - wether to choose greedy action or use epsilon greedy.
		Returns : action - int
		'''
		assert s.shape == (1, self.n_states)
		q_values = self.q_network(self.params, s)

		return self.policy(next(self.prng), q_values, exploration)

	def train(self, batch_size):
		''' Train the agent on a single episode. Uses the double q-learning target.
		Returns: td loss - float.'''
		if len(self.buffer.buffer) > batch_size:
			s, a, r, d, s_next = self.buffer.sample(batch_size)

			# compute gradients
			loss, gradients = jax.value_and_grad(self.loss_fn)(
				self.params, self.target_params, s, a, r, d, s_next)

			# apply gradients
			updates, self.opt_state = self.opt_update(gradients, self.opt_state, self.params)
			self.params = optax.apply_updates(self.params, updates)

			return loss

	def init_optimiser(self, lr, params):
		opt_init, opt_update = optax.adam(lr)
		opt_state = opt_init(params)
		return opt_update, opt_state

	@partial(jax.jit, static_argnums = 0)
	def loss_fn(self, params, target_params, s, a, r, d, s_next):
		# td targets
		y = r + self.gamma * self.q_network(target_params, s_next).max(-1) * (1.0 - d)

		# q-values
		q_values = self.q_network(params, s)
		q_values = q_values[jnp.arange(a.shape[0]), a]

		return jnp.square((q_values - y)).mean()

	def update_target(self):
		''' Update the weights of the target network.'''
		self.target_params = hk.data_structures.to_immutable_dict(self.params)

	def update_buffer(self, transition):
		''' Update the buffer with a transition.'''
		self.buffer.update(transition)

	def train_on_env(self, env, episodes, batch_size, target_freq, verbose = None):
		''' Train on a given environment.
		env : environment with methods reset and step e.g. gym CartPole-v1
		episodes : number of training episodes
		update_freq : frequency of training episodes
		train_steps : number of gradient descent steps each training epsiode
		verbose : wether to print current rewards. Given as int refering to print frequency.
		'''
		ep_rewards = []
		losses = []
		for episode in range(episodes):
			s = env.reset()
			d = False
			ep_reward = 0.0
			while not d:
				a = self.act(jnp.array(s)[jnp.newaxis, :], exploration = True)
				s_next, r, d, _ = env.step(a)
				ep_reward += r
				self.update_buffer(Transition(s=s, a=a, r=r, d=d, s_next=s_next))
				loss = self.train(batch_size)
				if loss is not None: losses.append(loss)
				s = s_next

			# ep end
			ep_rewards.append(ep_reward)

			if episode % target_freq == 0:
				self.update_target()

			if verbose is not None:
				if episode % verbose == 0:
					print('Episode {} Reward {:.4f} Loss {:.4f}'.format(episode,
						np.mean(ep_rewards[-verbose:]),
						np.mean(losses[-verbose:]))) # need to * by ep-length here

		return ep_rewards, losses


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def timestep_batchify(x: dict, agent_list, num_actors=None):
    x = jnp.concatenate([x[a] for a in agent_list], axis=1)
    return x

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
    x = x.reshape([num_agents, num_envs] + list(x.shape[1:]))
    return {a: x[i] for i, a in enumerate(agent_list)}

def timestep_unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
    x = x.reshape((x.shape[0], num_agents, num_envs, -1)).squeeze()
    return {a: x[:, i] for i, a in enumerate(agent_list)}

def print_jnp_shapes(d, key_path=None):
    if key_path is None:
        key_path = []
    for key, value in d.items():
        current_path = key_path + [key]
        if isinstance(value, dict):
            print_jnp_shapes(value, current_path)
        elif isinstance(value, jnp.ndarray):
            print("Key path:", " -> ".join(current_path), "Shape:", value.shape)
        else:
            print("Key path:", " -> ".join(current_path), "Shape:", value.shape)