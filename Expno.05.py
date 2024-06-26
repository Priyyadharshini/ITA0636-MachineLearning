import numpy as np
import pandas as pd

# Example DataFrame creation
np.random.seed(0)
data = np.concatenate((np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)))
df = pd.DataFrame(data, columns=['Data'])

# Initialization
def initialize_parameters(k):
    weights = np.ones(k) / k
    means = np.random.choice(df['Data'], k)
    variances = np.random.random(k)
    return weights, means, variances
# E-step
def expectation_step(df, weights, means, variances, k):
    N = df.shape[0]
    responsibilities = np.zeros((N, k))
    for i in range(k):
        responsibilities[:,i] = weights[i] * (1 / np.sqrt(2 * np.pi * variances[i])) * np.exp(
            -0.5 * ((df['Data'] - means[i]) ** 2 / variances[i]))
    responsibilities /= responsibilities.sum(axis=1)[:,np.newaxis]
    return responsibilities
# M-step
def maximization_step(df, responsibilities, k):
    Nk = responsibilities.sum(axis=0)
    weights = Nk / df.shape[0]
    means = (responsibilities * df['Data'].values[:, np.newaxis]).sum(axis=0) / Nk
    variances = (responsibilities * (df['Data'].values[:, np.newaxis] - means) ** 2).sum(axis=0) / Nk
    return weights, means, variances
# EM algorithm
def em_algorithm(df, k, iterations):
    weights, means, variances = initialize_parameters(k)
    log_likelihoods = []
    for _ in range(iterations):
        responsibilities = expectation_step(df, weights, means, variances, k)
        weights, means, variances = maximization_step(df, responsibilities, k)
    log_likelihood = np.sum(np.log(np.sum(responsibilities, axis=1)))
    log_likelihoods.append(log_likelihood)
    return weights, means, variances, log_likelihoods
# Running the EM algorithm
k = 2  # Number of Gaussian components
iterations = 100
weights, means, variances, log_likelihoods = em_algorithm(df, k, iterations)
# Output
print("Weights:", weights)
print("Means:", means)
print("Variances:", variances)
