# Libraries
import os
import numpy as np

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

# ARS V2 State normalization
class  Nornmalizer():
    def __init__(self, numPerceptInputs):
        # Input vector
        self.n = np.zeros(numPerceptInputs)
        self.mean = np.zeros(numPerceptInputs)
        self.mean_diff = np.zeros(numPerceptInputs)
        self.variance = np.zeros(numPerceptInputs)

    def observe(self, x):
        self.n += 1.

        # Online calculation of the mean
        lastMean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        # Online calculation of the variance
        self.mean_diff += (x - lastMean) * (x - self.mean)\
        # variance
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)

    # So that states have values of 0.0 to 1.0
    def normalize(self, inputs):
        observedMean = self.mean
        # Standard deviation
        observedSTD = np.sqrt(self.var)
        return (inputs - observedMean) / observedMean

# Fleshing out our agent
class Agent():
    def __init(self, inputSize, outputSize):
        # Matrix of weights for our perceptron
        self.theta = np.zero((outputSize, inputSize))

        # V2 ARS weight perturbations (states are normalized before passed in)
        def evaluate(self, input, delta = None, direction = None):
            if direction is None:
                # Return output of the perceptron without any weights
                return self.theta.dot(input)
            elif direction == "positive":
                # Add a positive small perturbation top weight and multiple to each input
                return (self.theta + hp.noise*delta).dot(input)
            else:
                # Try the negative version of the perturbation
                return (self.theta - hp.noise*delta).dot(input)

        def samplePurterbations(self):
            # Sample deltas according to a normal or gaussian distribution.
            return [np.random.randn(*self.theta.shape) for _ in range(hp.numDirections)]

        # Implementation of the method of finite differences
        def update(self, rollout, sigma_r):
            step = np.zeros(self.theta.shape)
            for r_po, r_neg, d in rollouts:
                step += (r_pos - r_neg) * d
            self.theta += hp.learningRate / (hp.numBestDirections * sigma)
