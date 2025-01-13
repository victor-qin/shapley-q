import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PPO(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64, lr=0.001):
        super(PPO, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def policy(self, obs):
        return torch.tanh(self.actor(obs))

    def value(self, obs):
        return self.critic(obs)

class PPOAgent:
    def __init__(self, env, hidden_dim=64, lr=0.001, clip_ratio=0.1, 
gamma=0.99, lam=0.95, entropy_coef=0.01, max_episode_length=1000):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0] if len(env.action_space.shape) >= 1 else 1
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.max_episode_length = max_episode_length
        self.ppo = PPO(self.obs_dim, self.act_dim, self.hidden_dim, 
self.lr)
        self.buffer = []

    def select_action(self, obs):
        return int(torch.round(self.ppo.policy(torch.tensor(obs))))

    def store_transition(self, obs, act, rew, next_obs, done):
        self.buffer.append((obs, act, rew, next_obs, done))

    def compute_loss(self):
        obs = torch.tensor(np.array([transition[0] for transition in self.buffer]))
        acts = torch.tensor(np.array([transition[1] for transition in self.buffer]))
        rews = torch.tensor(np.array([transition[2] for transition in self.buffer]))
        next_obs = torch.tensor(np.array([transition[3] for transition in self.buffer]))
        dones = torch.tensor(np.array([transition[4] for transition in self.buffer]))

        advantages = []
        returns = []

        for i in range(len(self.buffer)):
            gamma_sum = 0
            lam_sum = 0
            for j in range(i, len(self.buffer)):
                if j == i:
                    gamma_sum += self.gamma ** (j - i) * rews[j]
                    lam_sum += self.lam ** (j - i)
                else:
                    gamma_sum += self.gamma ** (j - i) * (rews[j] + 1.0 * dones[j])
                    lam_sum += self.lam ** (j - i)

            returns.append(gamma_sum)
            advantages.append(lam_sum)

        returns = torch.tensor(returns)
        advantages = torch.tensor(advantages)

        policy_loss = []
        value_loss = []

        for i in range(len(self.buffer)):
            pi_old = self.ppo.policy(torch.tensor(self.buffer[i][0]))
            log_pi_old = torch.log(pi_old + 1e-8).sum()
            advantages_i = advantages[i]

            ratio = (self.ppo.policy(torch.tensor(self.buffer[i][0])) / pi_old).detach().numpy()

            surr_loss = advantages_i * (log_pi_old - torch.log(ratio + 1e-8)).mean()
            clip_surr_loss = advantages_i * torch.min(
                (ratio * torch.exp(log_pi_old - torch.log(ratio + 1e-8))),
                (torch.exp(log_pi_old) * torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio))
            ).mean()

            policy_loss.append(surr_loss)
            
            value_loss.append((self.ppo.value(torch.tensor(self.buffer[i][0])) - returns[i]) ** 2)

        policy_loss = torch.stack(policy_loss).mean()
        value_loss = torch.stack(value_loss).mean()

        loss = policy_loss + 0.5 * self.lr * value_loss
        return loss

    def train(self):
        for _ in range(10):  # 10 iterations
            obs, acts, rews, next_obs, dones = zip(*self.buffer)

            self.ppo.optimizer.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            self.ppo.optimizer.step()

            self.buffer.clear()  # Clear buffer after each iteration

    def run(self):
        for episode in range(10):  # Run 10 episodes
            obs = self.env.reset()[0]
            done = False

            while not done:
                act = self.select_action(obs)
                next_obs, rew, done, _, _ = self.env.step(act)

                self.store_transition(obs, act, rew, next_obs, done)
                obs = next_obs

            self.train()

if __name__ == "__main__":
    import gym
    env = gym.make("CartPole-v1")
    agent = PPOAgent(env)
    agent.run()