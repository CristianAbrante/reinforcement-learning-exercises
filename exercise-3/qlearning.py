import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import sys

PLOTS_DIR = "plots/"
MODEL_DIR = "models/"

np.random.seed(123)

env = gym.make('CartPole-v0')
env.seed(321)

episodes = 20000
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


# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps_type", default="fixed",
                        help="reward function to be used")
    parser.add_argument("--eps_value", type=float, default=0.0,
                        help="Max number of episode steps.")
    return parser.parse_args(args)


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.4, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    # set axis values
    ax.set_xlabel("X values")
    ax.set_ylabel("θ values")

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("Optimal Q-value function for x and θ")

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


a = episodes / (1.0 - 0.01)


def get_epsilon(args, step):
    if args.eps_type == "fixed":
        return args.eps_value
    else:
        return a / (a + step)


def find_nearest(array, value):
    return np.argmin(np.abs(array - value))


def get_cell_index(state):
    x = find_nearest(x_grid, state[0])
    v = find_nearest(v_grid, state[1])
    th = find_nearest(th_grid, state[2])
    av = find_nearest(av_grid, state[3])
    return x, v, th, av


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
        epsilon = get_epsilon(args, ep)  # T1: GLIE/constant, T3: Set to 0
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

    # Save the Q-value array
    np.save(f"{MODEL_DIR}q_values-eps-{args.eps_type}-{args.eps_value}.npy",
            q_grid)  # TODO: SUBMIT THIS Q_VALUES.NPY ARRAY

    # Calculate the value function
    values = np.zeros(q_grid.shape[:-1])  # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID
    np.save(f"{MODEL_DIR}value_func-{args.eps_type}-{args.eps_value}.npy",
            values)  # TODO: SUBMIT THIS VALUE_FUNC.NPY ARRAY

    # Plot the heatmap
    # Choose the optimal q value for each state (based on x and y)

    steps = range(discr)
    optimal_q_value = np.array([
        [
            np.mean([np.max(q_grid[x, v, th, av]) for v in steps for av in steps])
            for th in steps
        ] for x in steps
    ])

    ## Labels for the ticks
    # x_labels = [f"{np.round(th_grid[i], 2)}, {np.round(th_grid[i + 1], 2)}" for i in range(len(th_grid) - 1)]
    x_labels = [f"{np.round(th_grid[th], 2)}" for th in range(len(th_grid))]
    # y_labels = [f"{np.round(x_grid[i], 2)}, {np.round(x_grid[i + 1], 2)}" for i in range(len(x_grid) - 1)]
    y_labels = [f"{np.round(x_grid[x], 2)}" for x in range(len(x_grid))]

    fig, ax = plt.subplots(figsize=(10, 10))
    im, cbar = heatmap(optimal_q_value, y_labels, x_labels, ax=ax,
                       cmap="YlGn", cbarlabel="Q-value")
    texts = annotate_heatmap(im, valfmt="{x:.1f}")

    fig.tight_layout()
    plt.savefig(f"{PLOTS_DIR}heatmap-{args.eps_type}-{args.eps_value}.png")
    plt.show()

    # Draw plots
    plt.plot(ep_lengths)
    plt.plot(epl_avg)
    plt.legend(["Episode length", "500 episode average"])
    plt.title("Episode lengths")
    plt.savefig(f"{PLOTS_DIR}episodes-{args.eps_type}-{args.eps_value}.png")
    plt.show()
