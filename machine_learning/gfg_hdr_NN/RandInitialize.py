import numpy as np

def initialise(a, b):
    epsilon = 0.15
    c = np.random.rand(a, b + 1) * (
      # Randomly initialises values of thetas between [-epsilon, +epsilon]
      2 * epsilon) - epsilon
    return c
