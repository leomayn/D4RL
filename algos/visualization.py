import hydra
import time
import jax
import numpy as np
from brax import envs
import mediapy as media
import imageio
import jax.numpy as jnp
import gym
from flax import serialization
import os

os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

def save_video(frames, filename='trajectory.mp4'):
    # Append .mp4 to ensure the correct format
    if not filename.endswith('.mp4'):
        filename += '.mp4'
    
    # Ensure the frames are in the correct format (e.g., numpy arrays, correct color channels)
    with imageio.get_writer(filename, fps=20, format='mp4') as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Video saved as {filename}")

def get_rollout(params, env, actor_model, config):
    
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    
    key = jax.random.PRNGKey(config["RNG"])
    key, key_r = jax.random.split(key)
    state = reset_fn(key_r)
    obs = state.obs
    network_params = params
    rollout = [state.pipeline_state]
    rewards = 0
    # grab a trajectory
    frames = []
    # while not done:
    for i in range(config["NUM_STEPS"]):
        key, key_a = jax.random.split(key)
        obs = obs[np.newaxis, np.newaxis, :]
        pi = actor_model.apply(network_params, obs)
        actions = pi.sample(seed=key_a)
        actions = jnp.squeeze(actions)
        state = step_fn(state, actions)
        obs = state.obs
        rewards += state.reward
        rollout.append(state.pipeline_state)
        
    print(f"Total reward: {rewards}")
    frames = env.render(trajectory=rollout, height=240, width=640)

    return frames, rewards


@hydra.main(version_base=None, config_path="config", config_name="visualization_config")
def main(config):
    
    env = envs.get_environment(config["ENV_NAME"])
    
    if config["ALG_NAME"] == "AWAC":
        from CORL.awac import GaussianPolicy
        print("awac")
        
        actor_model = GaussianPolicy(
        config["AWAC_HIDDEN_DIM"],
        action_dim=env.action_size,
        )
    '''
    if config["ALG_NAME"] == "CQL":
        from CORL.cql import TanhGaussianPolicy
        
        policy_model = TanhGaussianPolicy(
        observation_dim=observations.shape[-1],
        action_dim=actions.shape[-1],
        hidden_dims=config.hidden_dims,
        orthogonal_init=config.orthogonal_init,
        log_std_multiplier=config.policy_log_std_multiplier,
        log_std_offset=config.policy_log_std_offset,
        )
    if config["ALG_NAME"] == "DT":
        from CORL.dt import TD3Actor
        actor_model = TD3Actor(
        action_dim=action_dim,
        hidden_dims=config.hidden_dims,
        )
        
    if config["ALG_NAME"] == "IQL":
        from CORL.iql import GaussianPolicy
        
        actor_model = GaussianPolicy(
        config.hidden_dims,
        action_dim=action_dim,
        log_std_min=-5.0,
    )
        
    if config["ALG_NAME"] == "TD3BC":
        from CORL.td3bc import GaussianPolicy
    '''
    
    init_x = jnp.zeros(env.observation_size)
    init_params = actor_model.init(jax.random.PRNGKey(0), init_x)

    if config["USE_FINAL"]:
        params_path= os.path.join(config["REWARD_TYPE"], f'{config["D4RL_ENV"]}_final.msgpack')
    elif config["USE_BEST"]:
        params_path= os.path.join(config["REWARD_TYPE"], f'{config["D4RL_ENV"]}_best.msgpack')
    else:
        print("Please specify a model to load.")
        return
    if os.path.exists(params_path):
        with open(params_path, "rb") as f:
            params_bytes = f.read()
        params = serialization.from_bytes(init_params, params_bytes)
        print(f"Loaded parameters from {params_path}")
    else:
        print(f"Parameters file not found at {params_path}. Please check the path and try again.")
    
    start_time = time.time()
    frames, rewards = get_rollout(params, env, actor_model, config)
    save_video(frames, f'{config["ENV_NAME"]}_{config["ALG_NAME"]}_{config["REWARD_TYPE"]}_{rewards}_{config["RNG"]}.mp4')
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time for rendering: {total_time:.2f}")
    
    
if __name__ == "__main__":
    main()