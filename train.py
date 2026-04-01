import torch
import numpy as np
from mlp import mlp
from config import args
import gymnasium as gym
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical


def reward_to_go(rews, gamma=0.99):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for t in reversed(range(n)):
        rtgs[t] = rews[t] + gamma * (rtgs[t + 1] if t + 1 < n else 0)
    return rtgs


def train(env_name, hidden_size, lr, epochs, batch_size):
    env = gym.make(env_name)

    # TRY NOT TO MODIFY: seeds
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim] + hidden_size + [n_acts])

    # make value network for computing advantage.
    value_net = mlp(sizes=[obs_dim] + hidden_size + [1])

    # make function to compute action distribution.
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy).
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make policy loss function whose gradient, for the right data, is policy gradient.
    def compute_policy_loss(obs, act, weights):
        advantage = weights - value_net(obs).detach()
        # TODO: I should check without normalization.
        # NOTE: The best results with normalization!
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        logp = get_policy(obs).log_prob(act)
        return -(logp * advantage).mean()

    # make a value loss function for vanilla policy gradient.
    def compute_value_loss(obs, weights):
        # TODO: I should check value_net(obs) shape. Why we take .squeeze(-1)?
        # NOTE: We should .squeeze!
        return ((value_net(obs).squeeze(-1) - weights) ** 2).mean()

    # make policy optimizer
    policy_optimizer = Adam(logits_net.parameters(), lr=lr)

    # make value optimizer
    value_optimizer = Adam(value_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []

        # reset episode-specific variables
        obs = env.reset()[0]
        done = False
        ep_rews = []

        # Collecting experience by acting in the environment with current policy.
        while True:
            # Save obs
            batch_obs.append(obs)

            # Act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            # Save action & reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # If episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # The weight for each logprob(a|s) is R(tau).
                # NOTE: The best results with R(tau).
                batch_weights += [ep_ret] * ep_len

                # The weight for each logprob(a|s) is reward-to-go from t.
                # NOTE: With gamma the agent showed not good results.
                # batch_weights += list(reward_to_go(ep_rews))

                # Reset episode-specific variables
                obs, done, ep_rews = env.reset()[0], False, []

                # End experience loop if we have enough of it
                if len(batch_obs) >= batch_size:
                    break

        # print(torch.as_tensor(np.array(batch_obs), dtype=torch.float32).shape)
        # print(torch.as_tensor(np.array(batch_acts), dtype=torch.float32).shape)
        # print(torch.as_tensor(np.array(batch_weights), dtype=torch.float32).shape)

        # Take a single policy gradient update step
        policy_optimizer.zero_grad()
        batch_policy_loss = compute_policy_loss(
            obs=torch.as_tensor(np.array(batch_obs), dtype=torch.float32),
            act=torch.as_tensor(np.array(batch_acts), dtype=torch.int64),
            weights=torch.as_tensor(np.array(batch_weights), dtype=torch.float32),
        )
        batch_policy_loss.backward()
        policy_optimizer.step()

        # Take a several value gradient update step
        for _ in range(1):
            value_optimizer.zero_grad()
            batch_value_loss = compute_value_loss(
                obs=torch.as_tensor(np.array(batch_obs), dtype=torch.float32),
                weights=torch.as_tensor(np.array(batch_weights), dtype=torch.float32),
            )
            batch_value_loss.backward()
            value_optimizer.step()

        return batch_policy_loss, batch_value_loss, batch_rets, batch_lens

    avg_returns = []

    for i in range(epochs):
        batch_policy_loss, batch_value_loss, batch_rets, batch_lens = train_one_epoch()
        avg_batch_returns = np.mean(batch_rets)
        print(f"{avg_batch_returns:.4f}")

        avg_returns.append(avg_batch_returns)

    def plot_diag(data, title, ylabel, xlabel="Epoch"):
        plt.plot(data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    plot_diag(
        data=avg_returns,
        title="Ex10. Vanilla PG without norm, reward_to_go - with gamma, 1 training. Avg returns for epoch",
        ylabel="Return",
    )
