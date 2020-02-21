import numpy as np
def run(B,X_subset,y_subset,k):
    from sklearn.neighbors import KNeighborsClassifier
    n = len(X_subset)
    bs_err = np.zeros(B)
    for b in range(B):
        train_samples = list(np.random.randint(0,n,n))
        test_samples = list(set(range(n)) - set(train_samples))
        alg = KNeighborsClassifier(n_neighbors=k,algorithm='auto')
        alg.fit(X_subset[train_samples],np.ravel(y_subset[train_samples]))
        y_predict = alg.predict(X_subset[test_samples])
        bs_err[b] = np.mean(y_subset[test_samples] != np.array([y_predict]).T)
        #print b
    err = np.mean(bs_err)
    return err
