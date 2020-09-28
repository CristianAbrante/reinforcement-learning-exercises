# Copyright 2020 (c) Aalto University - All Rights Reserved
# ELEC-E8125 - Reinforcement Learning Course
# AALTO UNIVERSITY
#
#############################################################


import numpy as np
from time import sleep
from sailing import SailingGridworld
from tqdm import tqdm

epsilon = 10e-4  # DONE: Use this criteria for Task 3

# Set up the environment
env = SailingGridworld(rock_penalty=-2)
value_est = np.zeros((env.w, env.h))
env.draw_values(value_est)

if __name__ == "__main__":
    # Reset the environment
    state = env.reset()

    # Compute state values and the policy
    # DONE: Compute the value function and policy (Tasks 1, 2 and 3)
    value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h))

    gamma = 0.9
    value_est_prev = np.copy(value_est)

    # Task 1 and 2
    ''' #Comment Task 3 to use this
    for i in tqdm(range(100)):
        for x in range(len(env.transitions)):
            for y in range(len(env.transitions[x])):
                vet = []
                for f in env.transitions[x, y]:
                    v = 0
                    for f_state in f:
                        if f_state[0] != None:
                            v += f_state[3] * (f_state[1] + gamma * value_est_prev[f_state[0][0], f_state[0][1]])
                    vet.append(v)
                value_est[x, y] = max(vet)
                policy[x, y] = vet.index(max(vet))
        value_est_prev = value_est
    '''
    # ''' Task 3
    not_convergence = True
    c_convergence = 0
    threshold = epsilon
    while not_convergence:
        c_convergence += 1
        delta = 0
        for x in range(len(env.transitions)):
            for y in range(len(env.transitions[x])):
                vet = []
                for f in env.transitions[x, y]:
                    v = 0
                    for f_state in f:
                        if f_state[0] != None:
                            v += f_state[3] * (f_state[1] + gamma * value_est_prev[f_state[0][0], f_state[0][1]])
                    vet.append(v)
                value_est[x, y] = max(vet)
                policy[x, y] = vet.index(max(vet))
                if abs(value_est[x, y] - value_est_prev[x, y]) > delta:
                    delta = abs(value_est[x, y] - value_est_prev[x, y])
        value_est_prev = np.copy(value_est)
        if delta < threshold:
            not_convergence = False
            # print("Converge!")
    print("Convergence at ", c_convergence, "iteration")
    # '''

    # Show the values and the policy
    env.draw_values(value_est)
    env.draw_actions(policy)
    env.render()
    sleep(1)

    # Save the state values and the policy
    fnames = "task_1_values.npy", "task_2_policy.npy"
    np.save(fnames[0], value_est)
    np.save(fnames[1], policy)
    print("Saved state values and policy to", *fnames)

    # Run a single episode
    # DONE: Run multiple episodes and compute the discounted returns (Task 4)
    disc_return = []
    wins = 0
    fails = 0

    for i in tqdm(range(1000)):
        done = False
        k = 0
        G = 0
        env.reset()
        while not done:
            # Use the policy to take the optimal action (Task 2)
            action = int(policy[env.state[0], [env.state[1]]])

            # Step the environment
            state, reward, done, _ = env.step(action)
            G += reward * gamma ** k
            k += 1
            if done:
                if env.state[0] == 14 and env.state[1] == 9:
                    wins += 1
                else:
                    fails += 1
                disc_return.append(G)

    print("Episodes ended reaching the harbour: ", wins)
    print("Episodes ended hitting a rock: ", fails)
    print("Mean of the discounted returns: ", np.mean(np.array(disc_return)))
    print("Standard deviation of the discounted returns: ", np.std(np.array(disc_return)))

    '''
    done = False
    while not done:
        # Select a random action
        # TODO: Use the policy to take the optimal action (Task 2) DONE
        action = int(policy[env.state[0], [env.state[1]]])

        # Step the environment
        state, reward, done, _ = env.step(action)

        # Render and sleep
        env.render()
        sleep(0.5)'''
