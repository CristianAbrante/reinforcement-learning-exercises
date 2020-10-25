import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        self.initial_sigma = torch.Tensor([10.0])
        self.sigma = torch.Tensor([10.0])  # DONE: Implement accordingly (T1, T2)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_mean = self.fc2_mean(x)
        sigma = torch.sqrt(self.sigma)  # DONE: Is it a good idea to leave it like this?

        # DONE: Instantiate and return a normal distribution
        # with mean mu and std of sigma (T1)
        action_dist = Normal(loc=action_mean, scale=sigma)

        return action_dist

    def update_sigma_exponentially(self, episode_number):
        c = 0.0005
        self.sigma = self.initial_sigma * np.exp(-c * episode_number)


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []

    def episode_finished(self, episode_number):
        # Task 2a: update sigma of the policy exponentially decreasingly.
        self.policy.update_sigma_exponentially(episode_number + 1)

        action_probs = torch.stack(self.action_probs, dim=0) \
            .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards = [], [], []

        # DONE: Compute discounted rewards (use the discount_rewards function)
        discounted_rewards = discount_rewards(rewards, self.gamma)

        # Task 1c
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        # DONE: Compute the optimization term (T1)
        # task 1a
        baseline = 0
        # task 1b
        # baseline = 20

        weighted_probs = -action_probs * (discounted_rewards - baseline)

        # DONE: Compute the gradients of loss w.r.t. network parameters (T1)
        loss = torch.mean(weighted_probs)
        loss.backward()

        # DONE: Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # DONE: Pass state x through the policy network (T1)
        aprob = self.policy.forward(x)

        # DONE: Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        if evaluation:
            action = aprob.mean
        else:
            action = aprob.sample()

        # DONE: Calculate the log probability of the action (T1)
        act_log_prob = aprob.log_prob(action)

        return action, act_log_prob

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
