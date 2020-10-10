import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import sys

PLOTS_DIR = "plots/"
MODEL_DIR = "models/"

np.random.seed(123)

env = gym.make('LunarLander-v2')
env.seed(321)

episodes = 20000
test_episodes = 10
num_of_actions = 4

# For LunarLander, use the following values:
#         [    x     y  xdot ydot theta  thetadot cl  cr
# s_min = [ -1.2  -0.3  -2.4  -2  -6.28  -8       0   0 ]
# s_max = [  1.2   1.2   2.4   2   6.28   8       1   1 ]
discr = 16
x_min, x_max = -1.2, 1.2
y_min, y_max = -0.3, 1.2
vx_min, vx_max = -2.4, 2.4
vy_min, vy_max = -2, 2
th_min, th_max = -6.28, 6.28
av_min, av_max = -8, 8
possible_discrete_values = 2

# Parameters
gamma = 0.98
alpha = 0.1
target_eps = 0.1
a = round(target_eps * episodes / (1 - target_eps))
initial_q = 0

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
y_grid = np.linspace(y_min, y_max, discr)
vx_grid = np.linspace(vx_min, vx_max, discr)
vy_grid = np.linspace(vy_min, vy_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

q_grid = np.zeros((discr, discr, discr, discr, discr, discr, possible_discrete_values, possible_discrete_values,
                   num_of_actions)) + initial_q


# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps_type", default="fixed",
                        help="reward function to be used")
    parser.add_argument("--eps_value", type=float, default=0.0,
                        help="Max number of episode steps.")
    return parser.parse_args(args)


def get_epsilon(args, step):
    if args.eps_type == "fixed":
        return args.eps_value
    else:
        return a / (a + step)


def find_nearest(array, value):
    return np.argmin(np.abs(array - value))


def get_cell_index(state):
    x = find_nearest(x_grid, state[0])
    y = find_nearest(y_grid, state[1])
    vx = find_nearest(vx_grid, state[2])
    vy = find_nearest(vy_grid, state[3])
    th = find_nearest(th_grid, state[4])
    av = find_nearest(av_grid, state[5])
    c1 = round(state[6])
    c2 = round(state[7])
    return x, y, vx, vy, th, av, c1, c2


def get_action(state, q_values, epsilon, greedy=False):
    state_index = get_cell_index(state)
    actions = q_values[state_index]

    return np.argmax(actions) \
        if greedy or np.random.uniform(0, 1) >= epsilon \
        else np.random.randint(num_of_actions)


def update_q_value(old_state, action, new_state, reward, done, q_array):
    if not done:
        old_cell_index = get_cell_index(old_state)
        new_cell_index = get_cell_index(new_state)

        old_q_value_index = *old_cell_index, action
        old_cell_q_value = q_array[old_q_value_index]

        q_array[old_q_value_index] = old_cell_q_value + \
                                     alpha * (reward + gamma * np.max(q_array[new_cell_index]) - old_cell_q_value)


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()

    # Training loop
    ep_lengths, epl_avg = [], []
    for ep in range(episodes + test_episodes):
        test = ep > episodes
        state, done, steps = env.reset(), False, 0
        epsilon = get_epsilon(args, ep)
        while not done:
            action = get_action(state, q_grid, epsilon, greedy=test)
            new_state, reward, done, _ = env.step(action)
            if not test:
                update_q_value(state, action, new_state, reward, done, q_grid)
            else:
                env.render()
            state = new_state
            steps += 1
        ep_lengths.append(steps)
        epl_avg.append(np.mean(ep_lengths[max(0, ep - 500):]))
        if ep % 200 == 0:
            print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep - 200):])))

    # Draw plots
    plt.plot(ep_lengths)
    plt.plot(epl_avg)
    plt.legend(["Episode length", "500 episode average"])
    plt.xlabel("episodes")
    plt.ylabel("timesteps")
    plt.title("Episode lengths")
    plt.savefig(f"{PLOTS_DIR}/lunarlander-episodes-{args.eps_type}-{args.eps_value}.png")
    plt.show()
