# %%
# read and plot the data
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC

# read the data
lbp_features = np.load("lbp_features.npy")
num_classes = 2

def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    print(mu, sigma)
    return (data - mu) / sigma

lbp_X = lbp_features[:, :-1]
lbp_y = lbp_features[:, -1]
lbp_X = standardization(lbp_X)
Xtrain, Xtest, ytrain, ytest = train_test_split(lbp_X, lbp_y, test_size=0.2, random_state=0)

print(Xtrain.shape)
print(ytrain.shape)
# train knn model and use cross-validation to select the best k value
# define the k value range
K = range(1, 31)
scores = []
std_error = []

for k in K:
    print(k)
    model = KNeighborsClassifier(n_neighbors=k, weights='uniform')
    score = cross_val_score(model,  Xtrain, ytrain, cv=5, scoring='accuracy')
    scores.append(np.array(score).mean())
    std_error.append(np.array(score).std())
    
plt.errorbar(K, scores, yerr=std_error)
plt.xlabel('k value')
plt.ylabel('Accuracy')  
plt.show()

best_k = K[scores.index(max(scores))]
print("best_k:", best_k)
print("best_score:", scores[best_k])


# %%
