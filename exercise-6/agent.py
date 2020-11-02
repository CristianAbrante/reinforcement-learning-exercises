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
        self.hidden = 16
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)

        # DONE: Add another linear layer for the critic
        action_value_output = 1
        self.critic_layer = torch.nn.Linear(self.hidden, action_value_output)

        self.sigma = torch.nn.Parameter(torch.Tensor([10.0]))  # DONE: Implement learned variance (or copy from Ex5)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Common part
        x = self.fc1(x)
        x = F.relu(x)

        # Actor part
        action_mean = self.fc2_mean(x)
        sigma = torch.sqrt(self.sigma)  # DONE: Implement (or copy from Ex5)

        # Critic part
        # DONE: Implement
        state_value = self.critic_layer(x)

        # DONE: Instantiate and return a normal distribution
        # with mean mu and std of sigma
        # Implement or copy from Ex5
        action_dist = Normal(loc=action_mean, scale=sigma)

        # DONE: Return state value in addition to the distribution
        return action_dist, state_value


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.next_states = []
        self.done = []

    def update_policy(self, episode_number):
        # Convert buffers to Torch tensors
        action_probs = torch.stack(self.action_probs, dim=0) \
            .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        # Clear state transition buffers
        self.states, self.action_probs, self.rewards = [], [], []
        self.next_states, self.done = [], []

        # DONE: Compute state values
        state_values = torch.stack([self.policy.forward(state)[1][0] for state in states])
        next_state_values = torch.stack([self.policy.forward(state)[1][0] for state in next_states])

        # DONE: Compute critic loss (MSE)
        discounted_rewards = discount_rewards(rewards, self.gamma)

        # Normalize discounted rewards.
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        mse_loss = torch.nn.MSELoss()
        critic_loss = mse_loss(state_values, discounted_rewards)

        # Advantage estimates
        # DONE: Compute advantage estimates
        advantages = rewards + self.gamma * next_state_values - state_values

        # DONE: Calculate actor loss (very similar to PG)
        weighted_probs = -action_probs * advantages.detach()
        actor_loss = torch.mean(weighted_probs)

        # DONE: Compute the gradients of loss w.r.t. network parameters
        # Or copy from Ex5
        loss = actor_loss + critic_loss
        loss.backward()

        # DONE: Update network parameters using self.optimizer and zero gradients
        # Or copy from Ex5
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # DONE: Pass state x through the policy network
        # Or copy from Ex5
        aprob, _ = self.policy.forward(x)

        # DONE: Return mean if evaluation, else sample from the distribution
        # returned by the policy
        # Or copy from Ex5
        if evaluation:
            action = aprob.mean
        else:
            action = aprob.sample()

        # DONE: Calculate the log probability of the action
        # Or copy from Ex5
        act_log_prob = aprob.log_prob(action)

        return action, act_log_prob

    def store_outcome(self, state, next_state, action_prob, reward, done):
        # Now we need to store some more information than with PG
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
