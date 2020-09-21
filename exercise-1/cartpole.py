import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from agent import Agent, Policy
from utils import get_space_dim


# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None,
                        help="Model to be tested")
    parser.add_argument("--env", type=str, default="CartPole-v0",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=500,
                        help="Number of episodes to train for")
    parser.add_argument("--max_time_steps", "--s", type=int, default=200,
                        help="Max number of episode steps.")
    parser.add_argument("--render_training", action='store_true',
                        help="Render each frame during training. Will be slower.")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument("--reward_function", type=str, default="basic",
                        help="reward function to be used")
    parser.add_argument("--center_point", type=int, default=0,
                        help="point where you want to center the pole.")
    return parser.parse_args(args)


# Policy training function
def train(agent, env, train_episodes, args,
          early_stop=True, render=False, silent=False, train_run_id=0):
    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state
        observation = env.reset()

        # Loop until the episode is over
        while not done:
            # Get action from the agent
            action, action_probabilities = agent.get_action(observation)
            previous_observation = observation

            # Perform the action on the environment, get new state and reward
            observation, reward, done, info = env.step(action)

            reward = new_reward(observation, args, timesteps)

            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation, action_probabilities, action, reward)

            # Draw the frame, if desired
            if render:
                env.render()

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

        if not silent:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # If we managed to stay alive for 15 full episodes, assume it's learned
        # (in the default setting)
        if early_stop and np.mean(timestep_history[-15:]) == env._max_episode_steps:
            if not silent:
                print("Looks like it's learned. Finishing up early")
            break

        # Let the agent do its magic (update the policy)
        agent.episode_finished(episode_number)

    # Store the data in a Pandas dataframe for easy visualization
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id] * len(reward_history),
                         "reward": reward_history,
                         "mean_reward": average_reward_history})
    return data


# Function to test a trained policy
def test(agent, env, episodes, args, render=False):
    test_reward, test_len = 0, 0
    # Arrays to keep track of velocity
    velocity_history, timesteps_history = [], []
    max_velocity = -5  # Default value
    max_velocity_index = -1
    i = 0

    for ep in range(episodes):
        timesteps = 0
        done = False
        observation = env.reset()
        velocity_history.append([])
        timesteps_history.append([])
        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(observation, evaluation=True)
            observation, reward, done, info = env.step(action)

            reward = new_reward(observation, args, timesteps)
            print("r -> ", np.round(reward, 3), " x -> ", np.round(observation[0], 3))

            if render:
                env.render()

            velocity = observation[-1]
            velocity_history[-1].append(velocity)
            if max_velocity == -5 or velocity > max_velocity:
                max_velocity = velocity
                max_velocity_index = ep

            timesteps += 1
            timesteps_history[-1].append(timesteps)
            test_reward += reward
            test_len += 1

        print("done -> ", timesteps)

    print("Average test reward:", test_reward / episodes, "episode length:", test_len / episodes)
    data = pd.DataFrame({"timestep": timesteps_history[max_velocity_index],
                         "velocity": velocity_history[max_velocity_index]})

    # Plot max velocity
    sns.lineplot(x="timestep", y="velocity", data=data)
    plt.legend(["Velocity"])
    plt.title("Max velocity (episode %d)" % max_velocity_index)
    plt.show()
    print("Max velocity: ", max_velocity)
    print("Training finished.")


def new_reward(state, args, timesteps):
    if args.reward_function == "centered":
        return centered_reward(state, args.center_point)
    elif args.reward_function == "balanced":
        return balanced_reward(state, timesteps)
    else:
        return 1.0


def proportional_reward(x, x0=0, eps=0.1, max_reward=1.0, lower_bound=-1.0, higher_bound=+1.0):
    if (x0 - eps) <= x <= (x0 + eps):
        return 1
    elif lower_bound <= x < (x0 - eps):
        denominator = x0 - eps - lower_bound
        return (max_reward / denominator) * x - ((max_reward * lower_bound) / denominator)
    elif (x0 + eps) < x <= higher_bound:
        denominator = higher_bound - (x0 + eps)
        return (-max_reward / denominator) * x + ((max_reward * higher_bound) / denominator)
    else:
        return 0


# This function calculates the reward based on how close it is
# the point to the point given as a parameter.
def centered_reward(state, x0):
    lower_bound = -1.5
    higher_bound = +1.5
    eps = 0.1
    max_reward = 0.75

    return proportional_reward(state[0], x0, eps, max_reward, lower_bound, higher_bound)


def balanced_reward(state, timesteps):
    period = 5
    eps = 0.05
    max_reward = 0.75

    v_ideal = np.sin((2 * np.pi / period) * timesteps)
    return proportional_reward(state[1], v_ideal, eps, max_reward)


# The main function
def main(args):
    # Create a Gym environment
    env = gym.make(args.env)

    # Exercise 1
    env._max_episode_steps = args.max_time_steps

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy)

    # Print some stuff
    print("Environment:", args.env)
    print("Training device:", agent.train_device)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        training_history = train(agent, env, args.train_episodes, args, False, args.render_training)

        # Save the model
        model_file = "%s_params.ai" % args.env
        torch.save(policy.state_dict(), model_file)
        print("Model saved to", model_file)

        # Plot rewards
        sns.lineplot(x="episode", y="reward", data=training_history)
        sns.lineplot(x="episode", y="mean_reward", data=training_history)
        plt.legend(["Reward", "100-episode average"])
        plt.title("Reward history (%s)" % args.env)
        plt.show()
        print("Training finished.")
    else:
        print("Loading model from", args.test, "...")
        state_dict = torch.load(args.test)
        policy.load_state_dict(state_dict)
        print("Testing...")
        test(agent, env, args.train_episodes, args, args.render_test)


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)
