# Copyright 2020 (c) Aalto University - All Rights Reserved
# ELEC-E8125 - Reinforcement Learning Course
# AALTO UNIVERSITY
#
#############################################################


import numpy as np
from time import sleep
from sailing import SailingGridworld

epsilon = 10e-4  # TODO: Use this criteria for Task 3

# Set up the environment
env = SailingGridworld(rock_penalty=-10)
value_est = np.zeros((env.w, env.h))
env.draw_values(value_est)


def value_iteration(env, gamma, termination_condition):
    value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h))
    previous_values = np.copy(value_est)
    iteration = 0
    done = False

    while not done:
        for x in range(len(env.transitions)):
            for y in range(len(env.transitions[x])):
                # variable used for storing the different values of the value iteration
                values_of_transitions = np.array([])
                for transitions_set in env.transitions[x, y]:
                    value = 0
                    # For each transition the value is calculated
                    for transtion in transitions_set:
                        next_state = transtion[0]
                        # If next state exists
                        if next_state:
                            reward = transtion[1]
                            probability = transtion[3]
                            value += probability * (reward + gamma * previous_values[next_state[0], next_state[1]])
                    values_of_transitions = np.append(values_of_transitions, value)

                # update of the current value grid
                value_est[x, y] = np.max(values_of_transitions)
                policy[x, y] = np.argmax(values_of_transitions)

        # evaluate termination condition
        done = termination_condition(value_est, previous_values, policy, iteration)

        # update necessary values
        iteration += 1
        previous_values = np.copy(value_est)

    return value_est, policy


def convergence_criteria(current_val, prev_val, _, current_iteration):
    max_change = np.max(np.abs(np.subtract(current_val, prev_val)))
    converged = max_change < epsilon
    if converged:
        print(f'algorithm converged at iteration {current_iteration}')
    return converged


if __name__ == "__main__":
    # Reset the environment
    state = env.reset()
    gamma = 0.9

    # Compute state values and the policy
    # Task 1 and 2
    max_n_iterations = 100
    max_number_of_iterations = lambda current_val, prev_val, policy, iterations: iterations >= max_n_iterations
    value_est, policy = value_iteration(env, gamma, max_number_of_iterations)
    # Show the values and the policy
    env.draw_values(value_est)
    env.draw_actions(policy)
    env.render()
    env.save_figure("report/img/final-board-with-policy-2.png")
    sleep(1)

    # Task 3
    value_est, policy = value_iteration(env, gamma, convergence_criteria)

    # Show the values and the policy
    env.draw_values(value_est)
    env.draw_actions(policy)
    env.render()
    env.save_figure("report/img/final-board-with-threshold.png")
    sleep(1)

    # Save the state values and the policy
    # fnames = "values.npy", "policy.npy"
    # np.save(fnames[0], value_est)
    # np.save(fnames[1], policy)
    # print("Saved state values and policy to", *fnames)
    #
    # # Run a single episode
    # # TODO: Run multiple episodes and compute the discounted returns (Task 4)
    # done = False
    # while not done:
    #     # Select a random action
    #     action = policy[env.state[0], env.state[1]]
    #
    #     # Step the environment
    #     state, reward, done, _ = env.step(action)
    #
    #     # Render and sleep
    #     env.render()
    #     sleep(0.05)
