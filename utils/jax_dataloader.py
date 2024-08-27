import pickle
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import jax.numpy as jnp
import numpy as np
import time
import jax.random
from typing import NamedTuple
from utils.networks import timestep_batchify, batchify
from utils.networks import ScannedRNN

class Transition(NamedTuple):
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    infos: dict

class Trajectory(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    world_state: jnp.ndarray = None
    done: jnp.ndarray = None
    reward: jnp.ndarray = None
    log_prob: jnp.ndarray = None
    avail_actions: jnp.ndarray = None
    
class TransitionEnvState(NamedTuple):
    obs: dict
    actions: dict
    state: dict
    rewards: dict
    dones: dict
    infos: dict


class JaxDataLoader:
    def __init__(self, dir_path, file_list, env, seed=0, debug=True, batch_size=128, need_reward=False, vdn=False, device='cpu'):
        self.dir_path = dir_path
        self.file_list = file_list
        self.env = env
        self.seed = seed
        self.max_length = None
        self.batch_size = batch_size
        self.debug = debug
        self.rng = jax.random.PRNGKey(seed)
        self.need_reward = need_reward
        self.batchs = None
        self.vdn = vdn
        self.device = device
        
        if debug:
            print("Loading data from ", dir_path)
            print("file_list: ", file_list)
            start = time.time()
        if self.vdn:
            self.load_vdn_data()
        else:
            print("Loading converted data...")
            self.load_converted_data()        
        
    def load_converted_data(self):
        for file in self.file_list:
            with open(os.path.join(self.dir_path, file), 'rb') as f:
                new_data = pickle.load(f)
                if 'trajs' in locals():
                    trajs += new_data['trajs']
                    rewards += new_data['rewards']
                    traj_lens += new_data['traj_lens']
                else:
                    trajs = new_data['trajs']
                    rewards = new_data['rewards']
                    traj_lens = new_data['traj_lens']
        self.data = {'trajs': trajs, 
                     'rewards': rewards, 
                     'traj_lens': traj_lens}
        self.max_length = max(traj_lens)
        if self.debug:
            print("len(rewards): ", len(rewards))
        return
        
    def load_vdn_data(self):
        start_time = time.time()
        obs = None
        action = None
        done = None
        reward = None
        world_state = None
        agents = self.env.agents
        def convert_data(data):
            obs = timestep_batchify(data.obs, agents)[..., :-len(agents)] # remove the agents one-hot encoding
            obs = jnp.swapaxes(obs.reshape([-1, 26] + list(obs.shape[1:])), 1, 0).reshape([26, -1] + list(obs.shape[2:]))
            action = timestep_batchify(data.actions, agents)
            action = jnp.swapaxes(action.reshape([-1, 1] + list(action.shape[1:])), 1, 0).reshape([26, -1] + list(action.shape[2:]))
            
            done = timestep_batchify(data.dones, agents)
            done = jnp.swapaxes(done.reshape([-1, 1] + list(done.shape[1:])), 1, 0).reshape([26, -1] + list(done.shape[2:]))
            reward = timestep_batchify(data.rewards, agents)
            reward = jnp.swapaxes(reward.reshape([-1, 1] + list(reward.shape[1:])), 1, 0).reshape([26, -1] + list(reward.shape[2:]))
            reward = jnp.sum(reward, axis=0, keepdims=True)
            world_state = data.obs["__all__"].repeat(len(agents), axis=1)
            world_state = jnp.swapaxes(world_state.reshape([-1, 26] + list(world_state.shape[1:])), 1, 0).reshape([26, -1] + list(world_state.shape[2:]))
            
            return obs, action, done, reward, world_state
        
        if len(self.file_list) == 0:
            self.vdn_data = None
            raise ValueError("No files to load")
        
        for file in self.file_list:
            print("Loading file from ", os.path.join(self.dir_path, file))
            with open(os.path.join(self.dir_path, file), 'rb') as f:
                new_data = pickle.load(f)
                new_obs, new_action, new_done, new_reward, new_world_state = convert_data(new_data)
                obs = jnp.concatenate([obs, new_obs], axis=1) if obs is not None else new_obs
                action = jnp.concatenate([action, new_action], axis=1) if action is not None else new_action
                done = jnp.concatenate([done, new_done], axis=1) if done is not None else new_done
                reward = jnp.concatenate([reward, new_reward], axis=1) if reward is not None else new_reward
                world_state = jnp.concatenate([world_state, new_world_state], axis=1) if world_state is not None else new_world_state

        obs = obs.swapaxes(0, 1)        
        action = action.swapaxes(0, 1).squeeze()
        done = done.swapaxes(0, 1)
        reward = reward.squeeze()
        world_state = world_state.swapaxes(0, 1)
        traj_lengths = jnp.ones((len(obs),)) * 26    
        self.vdn_data = (jnp.array(obs), jnp.array(action), jnp.array(world_state), jnp.array(done), jnp.array(reward), traj_lengths)
        if self.debug:
            print("Vdn data from {} files loaded".format(len(self.file_list)))
            print("Data loaded in ", time.time() - start_time, " seconds")
            
    def __len__(self):
        if self.vdn_data is None:
            return 0
        if self.vdn:
            return self.vdn_data[0].shape[0]
        else:
            return len(self.data['trajs'])
    
    def get_data_for_jit(self):
        if self.vdn:
            obs, action, world_state, done, reward, traj_lengths= self.vdn_data
            return obs, action, world_state, done, None, None, reward, traj_lengths
        """convert all the data into a huge array for jit"""
        if self.debug:
            print("Converting data into a huge array for jit...")
            start_time = time.time()
        obs = []
        action = []
        world_state = []
        done = []
        reward = []
        log_prob = []
        avail_actions = []
        for traj in self.data['trajs'][:]:
            # filter out too-long trajs
            if traj.obs.shape[0] > 64:
                continue
            obs.append(traj.obs)
            action.append(traj.action)
            world_state.append(traj.world_state)
            done.append(traj.done)
            log_prob.append(traj.log_prob)
            if traj.avail_actions is not None:
                avail_actions.append(traj.avail_actions)
            if self.need_reward:
                reward.append(traj.reward)
        maximum_data_idx = len(obs)//self.batch_size * self.batch_size
        obs = obs[:maximum_data_idx]
        action = action[:maximum_data_idx]
        world_state = world_state[:maximum_data_idx]
        done = done[:maximum_data_idx]
        log_prob = log_prob[:maximum_data_idx]
        if self.need_reward:
            reward = reward[:maximum_data_idx]
        if len(avail_actions) > 0:
            avail_actions = avail_actions[:maximum_data_idx]
        

        def pad_and_concatenate(data, max_length=None):
            if max_length is None:
                max_length = max([d.shape[0] for d in data])
            padded_data = []
            if len(data[0].shape) == 1:
                for d in data:
                    padded_data.append(jnp.pad(d, (0, max_length - d.shape[0]), mode='constant', constant_values=0))
                return jnp.array(padded_data)
            else:
                for d in data:
                    padded_data.append(jnp.pad(d, ((0, max_length - d.shape[0]), (0, 0)), mode='constant', constant_values=0))
            return jnp.array(padded_data)
        
        obs = pad_and_concatenate(obs)
        action = pad_and_concatenate(action)
        world_state = pad_and_concatenate(world_state)
        done = pad_and_concatenate(done)
        log_prob = pad_and_concatenate(log_prob)
        returns = jnp.array(self.data['rewards'][:maximum_data_idx])
        if len(avail_actions) > 0:
            avail_actions = pad_and_concatenate(avail_actions)
        if self.need_reward:
            reward = pad_and_concatenate(reward)
        traj_lengths = jnp.array(self.data['traj_lens'][:maximum_data_idx])
        if self.debug:
            print("obs shape: ", obs.shape)
            print("action shape: ", action.shape)
            print("world_state shape: ", world_state.shape)
            print("done shape: ", done.shape)
            print("rewards shape: ", returns.shape)
            print("log_prob shape: ", log_prob.shape)
            print("traj_lengths shape: ", traj_lengths.shape)
            if self.need_reward:
                print("reward shape: ", reward.shape)
            else:
                print("reward shape: Not needed")
            if len(avail_actions) > 0:
                print("avail_actions shape: ", avail_actions.shape)
            else:
                print("avail_actions shape: None")
            print("Data converted in ", time.time() - start_time, " seconds")
        if self.need_reward:
            return obs, action, world_state, done, log_prob, avail_actions, returns, traj_lengths, reward
        else:
            return obs, action, world_state, done, log_prob, avail_actions, returns, traj_lengths
    
    
    def get_dummy_batch(self, size=1, need_avail_actions=False):
        # return a dummy batch of data with the same shape as the real batch
        if self.vdn:
            obs_dim = self.vdn_data[0].shape[-1]
            world_state_dim = self.vdn_data[2].shape[-1]
        else:
            obs_dim = self.data['trajs'][0].obs.shape[-1]
            world_state_dim = self.data['trajs'][0].world_state.shape[-1]
        if self.debug:
            print("obs_dim: ", obs_dim)
            print("world_state_dim: ", world_state_dim)
        dummy_obs = jnp.zeros((1, size, obs_dim))
        dummy_action = jnp.zeros((1, size)) # mpe agent action is a scalar
        dummy_world_state = jnp.zeros((1, size, world_state_dim))
        dummy_done = jnp.zeros((1, size))
        dummy_log_prob = jnp.zeros((1, size))
        dummy_reward = jnp.zeros((1, size))
        if need_avail_actions:
            avail_actions = jnp.zeros((1, size, self.data['trajs'][0].avail_actions.shape[-1]))
        else:
            avail_actions = None
        dummy_trajs = Trajectory(obs=dummy_obs, 
                                 action=dummy_action, 
                                 world_state=dummy_world_state,
                                 done=dummy_done,
                                 log_prob=dummy_log_prob,
                                 reward=dummy_reward,
                                 avail_actions=avail_actions)
        dummy_rewards = jnp.zeros((size,))
        
        return dummy_trajs, dummy_rewards, jnp.ones((size,))

class ILDataLoader:
    def __init__(self, **kwargs):
        '''
        Initialize the dataloader.
        Dataloader is used to load and manage the data for training the model.
        Some basic functions of dataloader include:
        - Load the data
        - Shuffle the data (if needed)
        - Get the next batch of data
        '''
        self.batch_size = kwargs.get('batch_size', 32)
        self.shuffle = kwargs.get('shuffle', True)
        self.random_state = kwargs.get('random_state', None)
        self.for_jax = kwargs.get('for_jax', False) # If True, the batch data should be converted to jax arrays.
        self.max_steps = kwargs.get('max_steps', 128)
        self.hidden_size = kwargs.get('hidden_size', 128)  # Add hidden size

        # Initialize variables that should be determined by the data
        self._data = None
        self._hidden_state = None  # Add hidden state
        self._data_size = None
        self._idx = 0 # Index of the current batch. This should be reset to 0 after each epoch.
        
        # Load the dataset
        self._load_file = kwargs.get('load_file', None)
        if self._load_file is not None:
            self._load_data()
        
        # Initialize the parameters
        self._reset()
        print("Dataloader initialized.")
    
    def _load_data(self):
        '''
        Load the data into the dataloader. 
        Your implementation should set the following variables:
        - self._data: the data to be loaded
        - self._data_size: the size of the data
        self._data should be a dictionary with the following keys:
        - 'obs': the observation data, a numpy arrays:(num_episode, max_episode_length, obs_size) 
        - 'action': the action data, a numpy arrays: (num_episode, max_episode_length, 1)
        - 'reward': the reward data, a numpy arrays: (num_episode, max_episode_length, 1)
        - 'done': the done data, a boolean numpy arrays: (num_episode, max_episode_length, 1)
        '''
        
        if isinstance(self._load_file, str):
            data = np.load(self._load_file, allow_pickle=True).item()
        elif isinstance(self._load_file, dict):
            data = self._load_file
        else:
            raise ValueError("Invalid data source. Provide a file path as a string or a dataset as a dictionary.")
        
        traj_length = self.max_steps
        obs = data['observations']
        actions = data['actions']
        rewards = data['rewards']
        terminals = data['terminals'] + data['timeouts']
        terminals = terminals.astype(bool)
        
        print("original actions shape: ", actions.shape)
        
        # Split the data into trajectories based on terminals (done flags)
        episode_starts = np.where(terminals)[0] + 1
        if episode_starts[-1] != len(terminals):
            episode_starts = np.concatenate(([0], episode_starts, [len(terminals)]))
        else:
            episode_starts = np.concatenate(([0], episode_starts))
        episode_lengths = np.diff(episode_starts)
        
        max_episode_length = self.max_steps
        
        # Initialize lists to store padded trajectories
        padded_obs = []
        padded_actions = []
        padded_rewards = []
        padded_terminals = []
        
        for start, length in zip(episode_starts, episode_lengths):
            end = start + length
            padded_obs.append(np.pad(obs[start:end], ((0, max_episode_length - length), (0, 0)), mode='constant'))
            padded_actions.append(np.pad(actions[start:end], ((0, max_episode_length - length), (0, 0)), mode='constant'))
            padded_rewards.append(np.pad(rewards[start:end], (0, max_episode_length - length), mode='constant').reshape(-1, 1))
            padded_terminals.append(np.pad(terminals[start:end], (0, max_episode_length - length), mode='constant').reshape(-1, 1))
        
        # Stack padded trajectories into numpy arrays
        obs = np.stack(padded_obs)
        actions = np.stack(padded_actions)
        rewards = np.stack(padded_rewards)
        terminals = np.stack(padded_terminals)
        terminals = np.squeeze(terminals, axis=-1)
        
        print(terminals.shape)
        
        # Store the padded data in CPU memory
        self._data = {
            'obs': obs,
            'action': actions,
            'reward': rewards,
            'done': terminals
        }
        
        # Initialize hidden states using ScannedRNN.initialize_carry
        self._hidden_state = ScannedRNN.initialize_carry(len(obs), self.hidden_size)

        # Set the data size to the number of trajectories
        self._data_size = len(self._data['obs'])
    
    def _get_batch(self):
        '''
        Get the next batch of data, including the hidden state.
        '''
        if self._idx >= self._data_size:
            UserWarning("End of data. Resetting the dataloader.")
            self._reset()
        
        start_idx = self._idx
        end_idx = min(start_idx + self.batch_size, self._data_size)
        if self.for_jax:
            start_idx = end_idx - self.batch_size  # make sure that all batches share the same shape for JAX JIT
        batch_data = {k: v[start_idx:end_idx] for k,v in self._data.items()}
        batch_hidden_state = self._hidden_state[start_idx:end_idx]  # Include hidden state in batch
        
        if self.for_jax:
            # Convert data and hidden state to JAX arrays and move them to the GPU
            batch_data = jax.tree_map(lambda x: jax.device_put(jnp.array(x), device=jax.devices('gpu')[0]), batch_data)
            batch_hidden_state = jax.device_put(jnp.array(batch_hidden_state), device=jax.devices('gpu')[0])
        else:
            # Keep the data and hidden state as numpy arrays in CPU memory
            batch_data = jax.tree_map(lambda x: jnp.array(x), batch_data)
            batch_hidden_state = jnp.array(batch_hidden_state)
        
        self._idx = end_idx
        
        return batch_data, batch_hidden_state
    
    def _shuffle(self):
        '''
        Shuffle the data and the hidden states.
        '''
        shuffled_indices = jax.random.permutation(self.random_state, jnp.arange(self._data_size))
        self._data = {k: v[shuffled_indices] for k,v in self._data.items()}
        self._hidden_state = jnp.take(self._hidden_state, shuffled_indices, axis=0)  # Shuffle hidden state
        
    def update_hidden_state(self, updated_hidden_state):
        '''
        Update the hidden state in the dataloader.
        '''
        start_idx = self._idx - self.batch_size
        end_idx = start_idx + updated_hidden_state.shape[0]
        
        # Use JAX's slice update instead of index_update
        self._hidden_state = self._hidden_state.at[start_idx:end_idx].set(updated_hidden_state)
    
    def _reset(self):
        '''
        Reset the dataloader.
        '''
        self._idx = 0
        if self.shuffle:
            self._shuffle()
    
    def __len__(self):
        if self._data_size is None:
            raise ValueError("Data size is not determined. Check the data loading process.")
        return self._data_size



def main():
    dir_path = "data/vdn104"
    file_list = ["traj_batch_" + str(x) + ".pkl" for x in range(10)]
    dataloader = JaxDataLoader(dir_path, file_list, vdn=True)
    obs, action, world_state, done, log_prob, avail_actions, rewards, traj_lens = dataloader.get_data_for_jit()
    
if __name__ == "__main__":
    main()