# Libraries
import os
import numpy

# Class for the Hyperparameters of AI
class HyperParam():
    def __init__(self, numSteps = 1000, episodeLength = 1000, learningRate = 0.02, numDirections = 16, numBestDirections = 16, noise = 0.03, seed = 1, environmentName = ''):
        self.numSteps = numSteps
        self.episodeLength = episodeLength
        self.learningRate = learningRate
        self.numDirections = numDirections
        self.numBestDirections = numBestDirections
        # Make sure the number of best directions is less then max directions
        assert self.numBestDirections <= self.numBestDirections
        self.noise = noise
        self.seed = seed
        self.environmentName = environmentName