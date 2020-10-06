import gym
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(123)

env = gym.make('CartPole-v0')
env.seed(321)

episodes = 200
test_episodes = 10
num_of_actions = 2

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# For LunarLander, use the following values:
#         [  x     y  xdot ydot theta  thetadot cl  cr
# s_min = [ -1.2  -0.3  -2.4  -2  -6.28  -8       0   0 ]
# s_max = [  1.2   1.2   2.4   2   6.28   8       1   1 ]

# Parameters
gamma = 0.98
alpha = 0.1
target_eps = 0.1
a = 0  # TODO: Set the correct value.
initial_q = 0  # T3: Set to 50

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

q_grid = np.zeros((discr, discr, discr, discr, num_of_actions)) + initial_q


def find_nearest(array, value):
    return np.argmin(np.abs(array - value))


def get_cell_index(state):
    x = find_nearest(x_grid, state[0])
    v = find_nearest(v_grid, state[1])
    th = find_nearest(th_grid, state[2])
    av = find_nearest(av_grid, state[3])
    return x, v, th, av


def get_action(state, q_values, greedy=False):
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


# Training loop
ep_lengths, epl_avg = [], []
for ep in range(episodes + test_episodes):
    test = ep > episodes
    state, done, steps = env.reset(), False, 0
    epsilon = 0.2  # T1: GLIE/constant, T3: Set to 0
    while not done:
        action = get_action(state, q_grid, greedy=test)
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

# Save the Q-value array
np.save("q_values.npy", q_grid)  # TODO: SUBMIT THIS Q_VALUES.NPY ARRAY

# Calculate the value function
values = np.zeros(q_grid.shape[:-1])  # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID
np.save("value_func.npy", values)  # TODO: SUBMIT THIS VALUE_FUNC.NPY ARRAY

# Plot the heatmap
# TODO: Plot the heatmap here using Seaborn or Matplotlib


# Choose the optimal q value for each state (based on x and y)

steps = range(discr)
optimal_q_value = np.array([
    [
        np.mean([np.max(q_grid[x, v, th, av]) for v in steps for av in steps])
        for th in steps
    ] for x in steps
])

## Here we create the labels for the ticks
# x_labels = [f"{np.round(th_grid[i], 2)}, {np.round(th_grid[i + 1], 2)}" for i in range(len(th_grid) - 1)]
x_labels = [f"{np.round(th_grid[th], 2)}" for th in range(len(th_grid))]
# y_labels = [f"{np.round(x_grid[i], 2)}, {np.round(x_grid[i + 1], 2)}" for i in range(len(x_grid) - 1)]
y_labels = [f"{np.round(x_grid[x], 2)}" for x in range(len(x_grid))]

# Creation of the heatmap plot
fig, ax = plt.subplots()
im = ax.imshow(optimal_q_value)

# Ticks created with the length of the labels
ax.set_xticks(np.arange(len(x_labels)))
ax.set_yticks(np.arange(len(y_labels)))
# And then labels shown
ax.set_xticklabels(x_labels)
ax.set_yticklabels(y_labels)
# Labels sets
ax.set_xlabel("X values")
ax.set_ylabel("θ values")

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(x_labels)):
    for j in range(len(y_labels)):
        text = ax.text(j, i, np.round(optimal_q_value[i, j], 1),
                       ha="center", va="center", color="w")

ax.set_title("Optimal Q-value function for x and θ")
fig.tight_layout()
plt.show()

# Draw plots
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode length", "500 episode average"])
plt.title("Episode lengths")
plt.show()
