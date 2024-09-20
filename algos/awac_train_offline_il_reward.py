import os

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from jaxrl.agents import BCLearner, AWACLearner
from jaxrl.datasets import make_env_and_dataset
from jaxrl.evaluation import evaluate

import jax
import jax.numpy as jnp
from utils.networks import ActorRNN, ScannedRNN
from safetensors.flax import save_file, load_file
from flax import serialization

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('il_env_name', 'halfcheetah-medium-expert-v2', 'IL Environment name.')
flags.DEFINE_integer('il_network_hidden', 128, 'IL network hidden.')
flags.DEFINE_string('il_network_save_path', 'results/IL_configs/student_network', 'IL network hidden.')
flags.DEFINE_enum('dataset_name', 'd4rl', ['d4rl', 'awac', 'rl_unplugged'],
                  'Dataset name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 18, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_float(
    'percentile', 100.0,
    'Dataset percentile (see https://arxiv.org/abs/2106.01345).')
flags.DEFINE_float('percentage', 100.0,
                   'Pencentage of the dataset to use for training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    'configs/bc_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)



def main(_):
    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', FLAGS.env_name, 'virtual'))

    video_save_folder = None if not FLAGS.save_video else os.path.join(
        FLAGS.save_dir, 'video', 'eval')
    
    
    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed,
                                        FLAGS.dataset_name, video_save_folder)
    
    # Beginning of IL
    
    il_network_hidden = FLAGS.il_network_hidden
    il_network_save_path = FLAGS.il_network_save_path
    
    def generate_reward(il_model, il_params, obs, action, done, il_h_state):
        in_data = (obs, done)
        il_h_state, pi = il_model.apply(il_params, il_h_state, in_data)
        reward = pi.log_prob(action)
        return (reward, il_h_state)
    il_model = ActorRNN(action_dim=env.action_space.shape[0], config=None)
    rng = jax.random.PRNGKey(0)
    rng, _s_init_rng = jax.random.split(rng)
    _s_init_x = (
        jnp.zeros((1, 1, env.observation_space.shape[0])),
        jnp.zeros((1, 1))
    )
    _s_init_h = ScannedRNN.initialize_carry(1, il_network_hidden)

    init_params = il_model.init(_s_init_rng, _s_init_h, _s_init_x)
    params_path = os.path.join(il_network_save_path, f'{FLAGS.il_env_name}_final.msgpack')

    if os.path.exists(params_path):
        with open(params_path, "rb") as f:
            params_bytes = f.read()
        il_params = serialization.from_bytes(init_params, params_bytes)
        print(f"Loaded parameters from {params_path}")
    else:
        print(f"Parameters file not found at {params_path}. Please check the path and try again.")
        return

    # End of IL

    
    obs = dataset.observations
    action = dataset.actions
    dones = dataset.dones_float
    num_samples = obs.shape[0]
    
    '''
    trajectory_length = 1000 # hardcoded for halfcheetah
    batch_size = num_samples // trajectory_length

    obs = obs.reshape(trajectory_length, batch_size, obs.shape[1])
    action = action.reshape(trajectory_length, batch_size, action.shape[1])
    dones = dones.reshape(trajectory_length, batch_size)
    '''
    
    obs = obs[np.newaxis, :]
    action = action[np.newaxis, :]
    dones = dones[np.newaxis, :]
    
    print("obs: ", obs.shape)
    
    # init_hs = ScannedRNN.initialize_carry(batch_size, il_network_hidden)
    init_hs = ScannedRNN.initialize_carry(num_samples, il_network_hidden)
    
    reward, _ = generate_reward(il_model, il_params, obs, action, dones, init_hs)
    
    # reward = reward.reshape(num_samples)
    
    reward = jnp.squeeze(reward, axis=0)
    print("reward: ", reward.shape)

    dataset.rewards = reward


    if FLAGS.percentage < 100.0:
        dataset.take_random(FLAGS.percentage)

    if FLAGS.percentile < 100.0:
        dataset.take_top(FLAGS.percentile)

    kwargs = dict(FLAGS.config)
    kwargs.pop('algo', None)
    kwargs.pop('replay_buffer_size', None)
    # kwargs['num_steps'] = FLAGS.max_steps
    agent = AWACLearner(FLAGS.seed,
                      env.observation_space.sample()[np.newaxis],
                      env.action_space.sample()[np.newaxis], **kwargs)

    eval_returns = []
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)

        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                summary_writer.add_scalar(f'training/{k}', v, i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.env_name}_virtual.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)
