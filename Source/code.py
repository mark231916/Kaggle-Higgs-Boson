
# Import data

import numpy as np
import csv

with open('../Dataset/training.csv', 'r') as f:
    reader = csv.reader(f,delimiter=',')
    header = next(reader)
    data = list(reader)
    data = np.array(data)

# Converting original lablels to -1 and 1
s = list(np.where(data=='s'))
data[tuple(s)] = 1
b = list(np.where(data=='b'))
data[tuple(b)] = -1
data1 = data.astype(float)

# Clean data


# Select testing and training samples
data2 = data1[range(120000, 125000)] # testing samples
data1 = data1[range(125000,130000)] # training samples
#print data1.shape
X = data1[:,1:31]
y = data1[:,32]
test_X = data2[:, 1:31]
test_y = data2[:,32]
y.astype(int)
y=y.reshape(-1,1)
test_y.astype(int)
test_y.reshape(-1,1)
n,d = X.shape
pos = np.sum(y==1)
neg = np.sum(y==-1)
positive_samples = list(np.where(y==1)[0])
negative_samples = list(np.where(y==-1)[0])

# Hyperparameter tuning
train_samples = positive_samples[0:pos/2] + negative_samples[0:neg/2]
validation_samples = positive_samples[pos/2:] + negative_samples[neg/2:]

from sklearn.neighbors import KNeighborsClassifier
import bootstrapping

import pcalearn
import pcaproj
best_F_1 = 1
best_F_2 = 1
best_err_1 = 1.1
best_k_1 = 0
best_err_2 = 1.1
best_k_2 = 0
X_fold1 = X[train_samples]
X_fold2 = X[validation_samples]
plot_m = np.zeros((7,20))
for F in range(8,15):
    k_list = list(range(10,30))

    for k in k_list:
        # y_pred = np.zeros(len(X),int)
        print "Current F and k: " + str(F) + " " + str(k)
        mu_fold1, Z_fold1 = pcalearn.run(F, X_fold1)
        X_fold1_small = pcaproj.run(X_fold1, mu_fold1, Z_fold1)
        mu_fold2, Z_fold2 = pcalearn.run(F, X_fold2)
        X_fold2_small = pcaproj.run(X_fold2, mu_fold2, Z_fold2)

        B = 1
        y_pred = np.zeros(len(train_samples) + len(validation_samples), int)

        err = bootstrapping.run(B, X_fold1_small, y[train_samples], k)
        if err < best_err_1:
            best_err_1 = err
            best_k_1 = k
            best_F_1 = F
        temp = err
        
        err = bootstrapping.run(B, X_fold2_small, y[validation_samples], k)
        
        if err < best_err_2:
            best_err_2 = err
            best_k_2 = k
            best_F_2 = F
        
        plot_m[F-8][k-10] = (temp+err)/2
print "Final result is : " + str(best_k_1) + " " + str(best_k_2) + " " + str(best_err_1) + " " + str(best_err_2) + " " + str(best_F_1) + " " + str(best_F_2)

best_err = np.amin(plot_m)
best_k = np.where(plot_m==best_err)[1][0] + 10
best_F = np.where(plot_m==best_err)[0][0] + 8


# Classification error for best k and best f

mu_fold1, Z_fold1 = pcalearn.run(best_F, X_fold1)
X_fold1_small = pcaproj.run(X_fold1, mu_fold1, Z_fold1)
mu_fold2, Z_fold2 = pcalearn.run(best_F, X_fold2)
X_fold2_small = pcaproj.run(X_fold2, mu_fold1, Z_fold1)

alg = KNeighborsClassifier(n_neighbors = best_k,algorithm='auto')
alg.fit(X_fold1_small,np.ravel(y[train_samples]))
y_pred[validation_samples] = alg.predict(X_fold2_small)


X_fold1_small = pcaproj.run(X_fold1, mu_fold2, Z_fold2)
X_fold2_small = pcaproj.run(X_fold2, mu_fold2, Z_fold2)

alg = KNeighborsClassifier(n_neighbors = best_k,algorithm='auto')
alg.fit(X_fold2_small,np.ravel(y[validation_samples]))
y_pred[train_samples] = alg.predict(X_fold1_small)

err = np.mean(y != np.array([y_pred]).T)

print "Classification Error:" + str(err)


import matplotlib.pyplot as pp
x = range(10,30)
Y = range(8,15)
Z = plot_m
fig,ax = pp.subplots(1,1)
cp = ax.contourf(x, Y, Z)
fig.colorbar(cp)
ax.set_title("Error versus Hyperparameters")
ax.set_xlabel("K value")
ax.set_ylabel("F value")
pp.show()


import matplotlib.pyplot as plt
# roc curve and auc score
from sklearn.neighbors import KNeighborsClassifier

# comment from here to the end if don't want to see ROC curve.
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

alg = KNeighborsClassifier(n_neighbors = best_k,algorithm='auto')
alg.fit(X,np.ravel(y))
#probs = alg.predict_proba(test_X)
#probs = probs[:,1]
#print np.array([probs]).T.shape, test_y.shape
fpr = np.zeros(22)
tpr = np.zeros(22)
P_label = list(np.where(test_y == 1))
P_label = np.asarray(P_label)
P_label = P_label.size
N_label = list(np.where(test_y == -1))
N_label = np.asarray(N_label)
N_label = N_label.size
#print P_label,N_label
for i in range(2,30):
    #print i
    alg = KNeighborsClassifier(n_neighbors = i, algorithm='auto')
    alg.fit(X, np.ravel(y))
    probs = alg.predict(test_X)
    TP = 0
    FP = 0
    for j in range(0,len(test_y)):
        if test_y[j] == 1 and probs[j] == 1:
            TP = TP + 1
        if test_y[j] == -1 and probs[j] == 1:
            FP = FP + 1
    fpr[i-9] = FP*1.0/N_label
    tpr[i-9] = TP*1.0/P_label
    #print TP,FP
fpr[0] = 0
fpr[21] = 1
tpr[0] = 0
tpr[21] = 1
print fpr,tpr
plot_roc_curve(fpr,tpr)