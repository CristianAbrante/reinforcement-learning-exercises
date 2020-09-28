# Copyright 2020 (c) Aalto University - All Rights Reserved
# ELEC-E8125 - Reinforcement Learning Course
# AALTO UNIVERSITY
#
#############################################################


import numpy as np
from time import sleep
from sailing import SailingGridworld

np.set_printoptions(threshold=np.Inf)

epsilon = 10e-4  # TODO: Use this criteria for Task 3

# Set up the environment
env = SailingGridworld(rock_penalty=-2)
value_est = np.zeros((env.w, env.h))
env.draw_values(value_est)

if __name__ == "__main__":
    # Reset the environment
    state = env.reset()

    # Compute state values and the policy
    # TODO: Compute the value function and policy (Tasks 1, 2 and 3)
    value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h))
    gamma_val = 0.9
    number_of_iterations = 100
    previous_values = np.copy(value_est)

    for iter in range(number_of_iterations):
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
                            value += probability * (reward + gamma_val * previous_values[next_state[0], next_state[1]])
                    values_of_transitions = np.append(values_of_transitions, value)

                # update of the current value grid
                value_est[x, y] = np.max(values_of_transitions)
                policy[x, y] = np.argmax(values_of_transitions)

        # previous values are updated
        previous_values = np.copy(value_est)
        print(f"iteration {iter}")
        print(value_est)

    # Show the values and the policy
    env.draw_values(value_est)
    env.draw_actions(policy)
    env.render()
    env.save_figure("report/img/final-board.png")
    sleep(1)

    # Save the state values and the policy
    fnames = "values.npy", "policy.npy"
    np.save(fnames[0], value_est)
    np.save(fnames[1], policy)
    print("Saved state values and policy to", *fnames)

    # Run a single episode
    # TODO: Run multiple episodes and compute the discounted returns (Task 4)
    # done = False
    # while not done:
    #     # Select a random action
    #     # TODO: Use the policy to take the optimal action (Task 2)
    #     action = int(np.random.random() * 4)
    #
    #     # Step the environment
    #     state, reward, done, _ = env.step(action)
    #
    #     # Render and sleep
    #     env.render()
    #     sleep(0.5)
