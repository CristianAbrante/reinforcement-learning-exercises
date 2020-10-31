import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import repeat
import sys
import multiprocessing as mp
#from episodic_cartpole import train
from nonep_cartpole import train


# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="ContinuousCartPole-v0",
                        help="Environment to use")
    parser.add_argument("--num_runs", type=int, default=10,
                        help="How many independent training runs to perform")
    return parser.parse_args(args)


def trainer(args):
    trainer_id, env = args
    print("Trainer id", trainer_id, "started")
    training_history = train(env, False, trainer_id)
    print("Trainer id", trainer_id, "finished")
    return training_history


# The main function
def main(args):
    # Create a pool with cpu_count() workers
    pool = mp.Pool(processes=mp.cpu_count())

    # Run the train function num_runs times
    results = pool.map(trainer, zip(range(args.num_runs),
                                    repeat(args.env, args.num_runs)))

    # Put together the results from all workers in a single dataframe
    all_results = pd.concat(results)

    # Save the dataframe to a file
    all_results.to_pickle("rewards.pkl")

    sns.set()
    figsize = (20, 12)
    plt.gcf().set_size_inches(*figsize)

    # Plot the mean learning curve, with the standard deviation
    sns.lineplot(x="episode", y="reward", data=all_results, ci="sd")

    # Plot (up to) the first 5 runs, to illustrate the variance
    n_show = min(args.num_runs, 5)
    smaller_df = all_results.loc[all_results.train_run_id < n_show]
    sns.lineplot(x="episode", y="reward", hue="train_run_id", data=smaller_df,
                 dashes=[(2,2)]*n_show, palette="Set2", style="train_run_id")
    plt.title("Training performance")
    plt.savefig("training.png")
    plt.savefig("training.pdf")
    #plt.show()


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

