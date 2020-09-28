# Copyright 2020 (c) Aalto University - All Rights Reserved
# ELEC-E8125 - Reinforcement Learning Course
# AALTO UNIVERSITY
#
#############################################################

from matplotlib import pyplot as plt
from matplotlib.patches import Circle


class Gridworld(object):


    def __init__(self, width, height, random_prob=0.1, die_on_edge=False):

    def render(self):
        """Draw the environment on screen."""
        self.ax.patches.clear()
        self.ax.set_yticklabels([])
        self.ax.set_xticklabels([])
        circle_x = self.state[0]/self.w + 1/(2*self.w)
        circle_y = self.state[1]/self.h + 1/(2*self.h)
        circle_r = 0.4*min(1/self.w, 1/self.h)
        circ = Circle((circle_x, circle_y), circle_r)
        self.ax.add_patch(circ)
        plt.grid(True)
        self.fig.canvas.draw()
        plt.pause(0.01)
