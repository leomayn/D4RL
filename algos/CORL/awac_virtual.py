# source https://github.com/ikostrikov/jaxrl
# https://arxiv.org/abs/2006.09359
import os
import time
from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import d4rl
import distrax
import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from pydantic import BaseModel
from flax import serialization
import imageio
import matplotlib.pyplot as plt

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class AWACConfig(BaseModel):
    # GENERAL
    algo: str = "AWAC"
    project: str = "train-AWAC"
    env_name: str = "halfcheetah-medium-expert-v2"
    seed: int = 42
    eval_episodes: int = 10
    log_interval: int = 1000
    eval_interval: int = 5000
    batch_size: int = 256
    max_steps: int = int(1e6)
    n_jitted_updates: int = 8
    # DATASET
    data_size: int = int(1e6)
    normalize_state: bool = False
    # NETWORK
    actor_hidden_dims: Tuple[int, int] = (256, 256, 256, 256)
    critic_hidden_dims: Tuple[int, int] = (256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    # AWAC SPECIFIC
    beta: float = 1.0
    tau: float = 0.005
    discount: float = 0.99
    reward_type: str = 'linear'
    il_hidden_dim: int = 128
    il_network_save_path: str = 'results/IL_configs/student_network'
    il_env_name: str = 'halfcheetah-medium-expert-v2'
    weight: float = 0.5

    def __hash__(
        self,
    ):  # make config hashable to be specified as static_argnums in jax.jit.
        return hash(self.__repr__())


conf_dict = OmegaConf.from_cli()
config = AWACConfig(**conf_dict)


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    kernel_init: Callable[[Any, Sequence[int], Any], jnp.ndarray] = default_init()
    add_layer_norm: bool = False
    layer_norm_final: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, hidden_dims in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dims, kernel_init=self.kernel_init)(x)
            if self.add_layer_norm:  # Add layer norm after activation
                if self.layer_norm_final or i + 1 < len(self.hidden_dims):
                    x = nn.LayerNorm()(x)
            if (
                i + 1 < len(self.hidden_dims) or self.activate_final
            ):  # Add activation after layer norm
                x = self.activations(x)
        return x


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(
        self, observation: jnp.ndarray, action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate([observation, action], axis=-1)
        q1 = MLP((*self.hidden_dims, 1), add_layer_norm=True)(x)
        q2 = MLP((*self.hidden_dims, 1), add_layer_norm=True)(x)
        return q1, q2


class GaussianPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -20.0
    log_std_max: Optional[float] = 2.0
    final_fc_init_scale: float = 1e-3

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
        )(observations)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)

        log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        return distribution


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray


def get_dataset(
    env: gym.Env, config: AWACConfig, clip_to_eps: bool = True, eps: float = 1e-5
) -> Transition:
    dataset = d4rl.qlearning_dataset(env)

    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    imputed_next_observations = np.roll(dataset["observations"], -1, axis=0)
    same_obs = np.all(
        np.isclose(imputed_next_observations, dataset["next_observations"], atol=1e-5),
        axis=-1,
    )
    dones = 1.0 - same_obs.astype(np.float32)
    dones[-1] = 1

    dataset = Transition(
        observations=jnp.array(dataset["observations"], dtype=jnp.float32),
        actions=jnp.array(dataset["actions"], dtype=jnp.float32),
        rewards=jnp.array(dataset["rewards"], dtype=jnp.float32),
        dones=jnp.array(dones, dtype=jnp.float32),
        next_observations=jnp.array(dataset["next_observations"], dtype=jnp.float32),
    )
    # shuffle data and select the first data_size samples
    data_size = min(config.data_size, len(dataset.observations))
    rng = jax.random.PRNGKey(config.seed)
    rng, rng_permute, rng_select = jax.random.split(rng, 3)
    perm = jax.random.permutation(rng_permute, len(dataset.observations))
    dataset = jax.tree_util.tree_map(lambda x: x[perm], dataset)
    assert len(dataset.observations) >= data_size
    dataset = jax.tree_util.tree_map(lambda x: x[:data_size], dataset)
    # normalize states
    obs_mean, obs_std = 0, 1
    if config.normalize_state:
        obs_mean = dataset.observations.mean(0)
        obs_std = dataset.observations.std(0)
        dataset = dataset._replace(
            observations=(dataset.observations - obs_mean) / (obs_std + 1e-5),
            next_observations=(dataset.next_observations - obs_mean) / (obs_std + 1e-5),
        )
    return dataset, obs_mean, obs_std


def target_update(
    model: TrainState, target_model: TrainState, tau: float
) -> Tuple[TrainState, jnp.ndarray]:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)


def update_by_loss_grad(
    train_state: TrainState, loss_fn: Callable
) -> Tuple[float, Any]:
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(train_state.params)
    new_train_state = train_state.apply_gradients(grads=grad)
    return new_train_state, loss


class AWACTrainState(NamedTuple):
    rng: jax.random.PRNGKey
    critic: TrainState
    target_critic: TrainState
    actor: TrainState


class AWAC(object):
    def update_actor(
        self,
        train_state: AWACTrainState,
        batch: Transition,
        rng: jax.random.PRNGKey,
        config: AWACConfig,
    ) -> Tuple["AWACTrainState", jnp.ndarray]:
        def get_actor_loss(actor_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            dist = train_state.actor.apply_fn(actor_params, batch.observations)
            pi_actions = dist.sample(seed=rng)
            q_1, q_2 = train_state.critic.apply_fn(
                train_state.critic.params, batch.observations, pi_actions
            )
            v = jnp.minimum(q_1, q_2)

            lim = 1 - 1e-5
            actions = jnp.clip(batch.actions, -lim, lim)
            q_1, q_2 = train_state.critic.apply_fn(
                train_state.critic.params, batch.observations, actions
            )
            q = jnp.minimum(q_1, q_2)
            adv = q - v
            weights = jnp.exp(adv / config.beta)

            weights = jax.lax.stop_gradient(weights)

            log_prob = dist.log_prob(batch.actions)
            loss = -jnp.mean(log_prob * weights).mean()
            return loss

        new_actor, actor_loss = update_by_loss_grad(train_state.actor, get_actor_loss)
        return train_state._replace(actor=new_actor), actor_loss

    def update_critic(
        self,
        train_state: AWACTrainState,
        batch: Transition,
        rng: jax.random.PRNGKey,
        config: AWACConfig,
    ) -> Tuple["AWACTrainState", jnp.ndarray]:
        def get_critic_loss(
            critic_params: flax.core.FrozenDict[str, Any]
        ) -> jnp.ndarray:
            dist = train_state.actor.apply_fn(
                train_state.actor.params, batch.observations
            )
            next_actions = dist.sample(seed=rng)
            n_q_1, n_q_2 = train_state.target_critic.apply_fn(
                train_state.target_critic.params, batch.next_observations, next_actions
            )
            next_q = jnp.minimum(n_q_1, n_q_2)
            q_target = batch.rewards + config.discount * (1 - batch.dones) * next_q
            q_target = jax.lax.stop_gradient(q_target)

            q_1, q_2 = train_state.critic.apply_fn(
                critic_params, batch.observations, batch.actions
            )

            loss = jnp.mean((q_1 - q_target) ** 2 + (q_2 - q_target) ** 2)
            return loss

        new_critic, critic_loss = update_by_loss_grad(
            train_state.critic, get_critic_loss
        )
        return train_state._replace(critic=new_critic), critic_loss

    @partial(jax.jit, static_argnums=(0, 4))
    def update_n_times(
        self,
        train_state: AWACTrainState,
        dataset: Transition,
        rng: jax.random.PRNGKey,
        config: AWACConfig,
    ) -> Tuple["AWACTrainState", Dict]:
        for _ in range(config.n_jitted_updates):
            rng, batch_rng, critic_rng, actor_rng = jax.random.split(rng, 4)
            batch_indices = jax.random.randint(
                batch_rng, (config.batch_size,), 0, len(dataset.observations)
            )
            batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)

            train_state, critic_loss = self.update_critic(
                train_state, batch, critic_rng, config
            )
            new_target_critic = target_update(
                train_state.critic,
                train_state.target_critic,
                config.tau,
            )
            train_state, actor_loss = self.update_actor(
                train_state, batch, actor_rng, config
            )
        return train_state._replace(target_critic=new_target_critic), {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
        }

    @partial(jax.jit, static_argnums=(0,))
    def get_action(
        self,
        train_state: AWACTrainState,
        observations: np.ndarray,
        seed: jax.random.PRNGKey,
        temperature: float = 1.0,
        max_action: float = 1.0,  # In D4RL envs, the action space is [-1, 1]
    ) -> jnp.ndarray:
        actions = train_state.actor.apply_fn(
            train_state.actor.params, observations=observations, temperature=temperature
        ).sample(seed=seed)
        actions = jnp.clip(actions, -max_action, max_action)
        return actions


def create_train_state(
    observations: jnp.ndarray, actions: jnp.ndarray, config: AWACConfig
) -> AWACTrainState:
    rng = jax.random.PRNGKey(config.seed)
    rng, actor_rng, critic_rng, value_rng = jax.random.split(rng, 4)
    # initialize actor
    action_dim = actions.shape[-1]
    actor_model = GaussianPolicy(
        config.actor_hidden_dims,
        action_dim=action_dim,
    )
    actor = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations),
        tx=optax.adam(learning_rate=config.actor_lr),
    )
    # initialize critic
    critic_model = DoubleCritic(config.critic_hidden_dims)
    critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(learning_rate=config.critic_lr),
    )
    # initialize target critic
    target_critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(learning_rate=config.critic_lr),
    )
    return AWACTrainState(
        rng,
        critic=critic,
        target_critic=target_critic,
        actor=actor,
    )


def evaluate(
    policy_fn: Callable,
    env: gym.Env,
    num_episodes: int,
    obs_mean: float,
    obs_std: float,
) -> float:
    episode_returns = []
    for _ in range(num_episodes):
        episode_return = 0
        observation, done = env.reset(), False
        while not done:
            observation = (observation - obs_mean) / obs_std
            action = policy_fn(observations=observation)
            observation, reward, done, info = env.step(action)
            episode_return += reward
        episode_returns.append(episode_return)
    nomarlized_scores = jax.tree.map(lambda x: env.get_normalized_score(x) * 100, episode_returns)
    # return env.get_normalized_score(np.mean(episode_returns)) * 100
    return jnp.array(nomarlized_scores)


if __name__ == "__main__":
    
    print(f"Running awac_virtual with config.reward_type={config.reward_type} with seed {config.seed} on gpu{os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    wandb.init(config=config, project=config.project, name=config.env_name + "_" + config.algo + "_" + config.reward_type)
    rng = jax.random.PRNGKey(config.seed)
    env = gym.make(config.env_name)
    dataset, obs_mean, obs_std = get_dataset(env, config)
    
    # modify dataset
    
    # Beginning of IL
    from utils.networks import ScannedRNN, ActorRNN
    
    il_network_hidden = config.il_hidden_dim
    il_network_save_path = config.il_network_save_path
    
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
    params_path = os.path.join(il_network_save_path, f'{config.il_env_name}_final.msgpack')

    if os.path.exists(params_path):
        with open(params_path, "rb") as f:
            params_bytes = f.read()
        il_params = serialization.from_bytes(init_params, params_bytes)
        print(f"Loaded parameters from {params_path}")
    else:
        print(f"Parameters file not found at {params_path}. Please check the path and try again.")
        raise FileNotFoundError(f"Parameters file not found at {params_path}. Please check the path and try again.")

    # End of IL

    
    obs = dataset.observations
    action = dataset.actions
    dones = dataset.dones
    num_samples = obs.shape[0]
    
    
    obs = obs[np.newaxis, :]
    action = action[np.newaxis, :]
    dones = dones[np.newaxis, :]
    
    
    # init_hs = ScannedRNN.initialize_carry(batch_size, il_network_hidden)
    init_hs = ScannedRNN.initialize_carry(num_samples, il_network_hidden)
    
    reward, _ = generate_reward(il_model, il_params, obs, action, dones, init_hs)
    
    # reward = reward.reshape(num_samples)
    
    virtual_reward = jnp.squeeze(reward, axis=0)
    reward_mean = jnp.mean(virtual_reward)
    reward_std = jnp.std(virtual_reward)
    normalized_reward = (reward - reward_mean) / reward_std
    normalized_reward = jnp.squeeze(normalized_reward, axis=0)
    real_reward = dataset.rewards
    # create train_state
    example_batch: Transition = jax.tree_util.tree_map(lambda x: x[0], dataset)
    train_state: AWACTrainState = create_train_state(
        example_batch.observations,
        example_batch.actions,
        config,
    )
    algo = AWAC()

    num_steps = config.max_steps // config.n_jitted_updates
    best_normalized_score = -np.inf
    start = time.time()
    def linear_schedule(step, num_steps=10000):
        return min(1.0, step / num_steps)
    
    def constant_schedule(weight):
        def return_weight(step):
            return weight
        return return_weight
    
    if config.reward_type == 'virtual':
        update_alpha = constant_schedule(0.0)
    elif config.reward_type == 'real':
        update_alpha = constant_schedule(1.0)
    elif config.reward_type == 'linear':
        update_alpha = linear_schedule
    elif config.reward_type == 'mixed':
        update_alpha = constant_schedule(config.weight)
    
    for i in tqdm.tqdm(range(1, num_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # update dataset rewards
        alpha = update_alpha(i)
        dataset = dataset._replace(rewards= alpha * real_reward + (1 - alpha) * normalized_reward)
    
        rng, subkey = jax.random.split(rng)
        train_state, update_info = algo.update_n_times(
            train_state,
            dataset,
            subkey,
            config,
        )
        if (i * config.n_jitted_updates) % config.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i * config.n_jitted_updates)
            wandb.log({"training/alpha": alpha}, step=i * config.n_jitted_updates)

        if (i * config.n_jitted_updates) % config.eval_interval == 0:
            policy_fn = partial(
                algo.get_action,
                temperature=0.0,
                seed=jax.random.PRNGKey(0),
                train_state=train_state,
            )
            normalized_score = evaluate(
                policy_fn, env, config.eval_episodes, obs_mean, obs_std
            ).mean()
            # save best model based on normalized score
            if normalized_score > best_normalized_score:
                best_normalized_score = normalized_score
                if not os.path.exists(config.reward_type):
                    os.makedirs(config.reward_type)
                file_path = os.path.join(config.reward_type, f'{config.env_name}_best.msgpack')
                with open(file_path, "wb") as f:
                    f.write(serialization.to_bytes(train_state.actor.params))
                print(f"Saved the best model to {file_path}")
            
            print(i * config.n_jitted_updates, normalized_score)
            eval_metrics = {f"{config.env_name}/normalized_score": normalized_score}
            wandb.log(eval_metrics, step=i * config.n_jitted_updates)
    # final evaluation
    policy_fn = partial(
        algo.get_action,
        temperature=0.0,
        seed=jax.random.PRNGKey(0),
        train_state=train_state,
    )
    
    if not os.path.exists(config.reward_type):
        os.makedirs(config.reward_type)
    file_path = os.path.join(config.reward_type, f'{config.env_name}_final.msgpack')
    with open(file_path, "wb") as f:
        f.write(serialization.to_bytes(train_state.actor.params))
    print(f"Saved the best model to {file_path}")
    
    normalized_scores = evaluate(policy_fn, env, config.eval_episodes, obs_mean, obs_std)
    # plot a final score distribution
    plt.hist(normalized_scores)
    plt.xlabel("Normalized Score")
    plt.ylabel("Frequency")
    plt.title("Final Evaluation Score Distribution")
    plt.savefig(f"{config.env_name}_final_score_distribution.png")
    print("Final evaluation score", normalized_scores.mean())
    wandb.log({f"{config.env_name}/final_normalized_score": normalized_scores.mean()})
    wandb.finish()