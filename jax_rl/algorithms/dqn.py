import jax
import jax.numpy as jnp
import numpy as np 
import haiku as hk 
import optax

from functools import partial

from jax_rl.algorithms import BaseAgent
from jax_rl.buffer import ExperienceReplay
from jax_rl.utils import Transition

from utils.networks import AgentRNN, ActorRNN, ScannedRNN

import jax
from jax import random

def policy_extractor(key, q_values, exploration=True, epsilon=0.1):
    """
    Extract continuous actions based on the Q-values with exploration noise.
    
    q_values: Q-values predicted by the network, should correspond to continuous action parameters.
    key: JAX random key for any stochastic operation (like sampling).
    exploration: Whether to apply exploration noise.
    epsilon: The scale of exploration noise (how much noise to add).
    
    Returns:
    Continuous actions of shape (batch_size, action_dim).
    """
    if exploration:
        # Apply Gaussian noise for exploration, scaled by epsilon
        action_noise = jax.random.normal(key, q_values.shape) * epsilon
        actions = q_values + action_noise
    else:
        # Simply use the q_values as the deterministic action output
        actions = q_values

    # Remove the time-step dimension to get the shape (batch_size, action_dim)
    actions = jnp.squeeze(actions, axis=0)
    
    return actions





class DQN(BaseAgent):
	''' Deep Q-network'''

	def __init__(self, key, n_states, n_actions, gamma, buffer_size, model, lr, num_envs):
		super(DQN, self).__init__(key, n_states, n_actions, gamma)

		self.buffer = ExperienceReplay(buffer_size)
		self.q_network = model  # Use the custom AgentRNN model

		# Initialize the model parameters
		init_x = (
			jnp.zeros((1, num_envs, n_states)),  # (time_step, batch_size, obs_size)
			jnp.zeros((1, num_envs))  # (time_step, batch size)
		)
  
		init_hidden_state = ScannedRNN.initialize_carry(num_envs, model.hidden_dim)
		self.params = self.q_network.init(key, init_hidden_state, init_x)
		self.update_target()

		# Optimizer
		self.opt_update, self.opt_state = self.init_optimiser(lr, self.params)
		self.num_envs = num_envs

	def act(self, key, s, dones, hidden_state, exploration=True, epsilon=0.1):
		""" Get an action from the q-network given the state and done flags.
		s - shape (batch_size, n_states) current state
		dones - shape (batch_size, 1) done flags
		exploration - bool - whether to choose greedy action or use epsilon greedy.
		Returns: hidden_state - updated hidden state after RNN pass
				actions - array of ints (one per environment)
		"""

		# Prepare the inputs for the network: add a time_step dimension
		obs_ = s[np.newaxis, :]  # Add time_step dimension
		dones_ = dones[np.newaxis, :]  # Add time_step dimension

		# Pass the inputs through the network to get q-values
		hidden_state, q_values = self.q_network.apply(self.params, hidden_state, (obs_, dones_))

		# Select actions using epsilon-greedy policy
		actions = policy_extractor(key, q_values, exploration=exploration, epsilon=epsilon)

		print("Actions shape before squeeze:", actions.shape)

		return key, hidden_state, actions


	def train(self, batch_size):
		"""Train the agent on a single episode. Uses the double q-learning target.
		Returns: td loss - float."""
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

class DDQN(DQN):
	''' DQN with double Q-learning. Overwrites the loss function with the DDQN target.'''

	def __init__(self, *args):
		super(DDQN, self).__init__(*args)

	@partial(jax.jit, static_argnums = 0)
	def loss_fn(self, params, target_params, s, a, r, d, s_next):
		# td targets
		a_targ = jnp.argmax(jax.lax.stop_gradient(self.q_network(params, s_next)), axis = -1)
		q_targ = self.q_network(target_params, s_next)[jnp.arange(a_targ.shape[0]), a_targ]
		y = r + self.gamma * q_targ * (1.0 - d)

		# q-values
		q_values = self.q_network(params, s)
		q_values = q_values[jnp.arange(a.shape[0]), a]

		return jnp.square((q_values - y)).mean()
