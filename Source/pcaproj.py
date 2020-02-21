import numpy as np
# Input: number of features F
#        numpy matrix X, with n rows (samples), d columns (features)
#        numpy vector mu, with d rows, 1 column
#        numpy matrix Z, with d rows, F columns
# Output: numpy matrix P, with n rows, F columns
def run(X,mu,Z):
  X = np.copy(X)
  n, d = X.shape
  for t in range(n):
    X[t] = X[t] - mu.T
  P = np.dot(X,Z)
  return P
