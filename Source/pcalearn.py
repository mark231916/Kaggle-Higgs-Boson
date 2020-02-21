import numpy as np
import numpy.linalg as la
# Input: number of features F
#        numpy matrix X, with n rows (samples), d columns (features)
# Output: numpy vector mu, with d rows, 1 column
#         numpy matrix Z, with d rows, F columns
def run(F,X):
  X = np.copy(X)
  n, d = X.shape
  mu = np.array([np.mean(X,0)]).T
  for t in range(n):
    X[t] = X[t] - mu.T
  U, s, Vt = la.svd(X,False)
  g = s[0:F]
  for i in range(F):
    if g[i] > 0:
      g[i] = 1/g[i]
  W = Vt[0:F]
  Z = np.dot(W.T,np.diag(g))
  return (mu, Z)
