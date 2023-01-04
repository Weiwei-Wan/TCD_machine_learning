# %%
# read and plot the data
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.dummy import DummyClassifier

# read the data
data_name = "week4_data2.txt"
df = pd.read_csv(data_name, sep=",", names=["X_1", "X_2", "Y"])

# get seperate colum of data
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
y = df.iloc[:, 2]

def scatterValue(x1, x2, Y, color1, color2, marker, label1, label2, s):
    x1_pos = []
    x2_pos = []
    x1_neg = []
    x2_neg = []
    # devide the data by y value
    for i in range(0, len(Y)):
        if Y[i] == 1:
            x1_pos.append(x1[i])
            x2_pos.append(x2[i])
        elif Y[i] == -1:
            x1_neg.append(x1[i])
            x2_neg.append(x2[i])
    # draw the scatter plot
    plt.scatter(x1_pos, x2_pos, c = color1, label = label1, marker=marker, s=s)
    plt.scatter(x1_neg, x2_neg, c = color2, label = label2, marker=marker, s=s)
    plt.xlabel("X_1")
    plt.ylabel("X_2")

scatterValue(X1, X2, y, "r", "b", "+", "+1 points", " -1 points", 20)
# show the figure
plt.legend()
plt.show()

x = np.column_stack((X1, X2))
# split up our data into train and test
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

#%%
# train Logistic Regression classifier and use cross-validation to select best penalty C and polynomial features
# find the maximum order of polynomial
scores = []; std_error = []
Features = range(1, 11)
for i in Features:
    # add extra polynomial features
    poly = PolynomialFeatures(i)
    Xtrain = poly.fit_transform(xtrain)
    # define model
    model = LogisticRegression(penalty = 'l2', C = 1)
    score = cross_val_score(model, Xtrain, ytrain, cv=5, scoring='accuracy')
    scores.append(np.array(score).mean())
    std_error.append(np.array(score).std())

plt.errorbar(Features, scores, yerr=std_error)
plt.xlabel('feature order')
plt.ylabel('Accuracy')  
#plt.ylim(0.65, 0.68)
plt.show()

best_feature = Features[scores.index(max(scores))]
best_feature = 1
print("best_feature:", best_feature)
print("best_score:", scores[best_feature])

# find the best weight C
scores = []; std_error = []
C = [0.01, 0.1, 1, 10, 100, 1000]
# add extra polynomial features
poly = PolynomialFeatures(best_feature)
X = poly.fit_transform(x)
Xtrain = poly.fit_transform(xtrain)
Xtest = poly.fit_transform(xtest)
for i in C:
    # define model
    model = LogisticRegression(penalty = 'l2', C = i)
    score = cross_val_score(model,  Xtrain, ytrain, cv=5, scoring='accuracy')
    scores.append(np.array(score).mean())
    std_error.append(np.array(score).std())

plt.errorbar(np.log(C), scores, yerr=std_error)
plt.xlabel('log(C) value')
plt.ylabel('Accuracy')  
#plt.ylim(0.65, 0.68)
plt.show()

best_c = C[scores.index(max(scores))]
best_c = 0.01
print("best_c:", best_c)
print("best_score:", scores[C.index(best_c)])

# %%
# show original data and predicted data and dicision boundary of the selected Logistic Regression classifier
# Draw decision boundary
def plot_decision_boundary(pred_func, feat_num):
    # generated value and try to cover all region of the coordinate system
    x1_min, x1_max = -1.2, 1.2
    x2_min, x2_max = -1.2, 1.2
    h = 0.01
    x1_b, x2_b = np.meshgrid(np.arange(x1_min,x1_max,h), np.arange(x2_min,x2_max,h)) 
    # predict the generated value
    Z = []
    for i in range(0, len(x1_b)):
        x_tem = np.column_stack((x1_b[i], x2_b[i]))
        if feat_num > 0:
            poly_tem = PolynomialFeatures(feat_num)
            x_tem = poly_tem.fit_transform(x_tem)
        Z.append(pred_func.predict(x_tem))
    # draw the decision boundary
    plt.contourf(x1_b, x2_b, Z, alpha = 0.5)

print(poly.get_feature_names(['X_1','X_2']))

best_log_model = LogisticRegression(penalty = 'l2', C = best_c)
best_log_model.fit(Xtrain, ytrain)

# Draw decision boundary
plot_decision_boundary(best_log_model, best_feature)

scatterValue(X1, X2, y, "g", "b", "+", "+1 points", " -1 points", 20)
scatterValue(X1, X2, best_log_model.predict(X), "r", "k", ".", "predicted +1 points", "predicted -1 points", 10)

plt.legend()
plt.show()

print ('Intercept of model: ', best_log_model.intercept_[0])
print ('Coefficient of model: ', best_log_model.coef_[0])

# %%
# train knn model and use cross-validation to select the best k value
# define the k value range
K = range(1, 51)
scores = []; std_error = []
for k in K:
    model = KNeighborsClassifier(n_neighbors=k, weights='uniform')
    score = cross_val_score(model,  xtrain, ytrain, cv=5, scoring='accuracy')
    scores.append(np.array(score).mean())
    std_error.append(np.array(score).std())

plt.errorbar(K, scores, yerr=std_error)
plt.xlabel('k value')
plt.ylabel('Accuracy')  
plt.show()

best_k = K[scores.index(max(scores))]
best_k = 25
print("best_k:", best_k)
print("best_score:", scores[best_k])

# %%
# show original data and predicted data and dicision boundary of the selected knn model
best_knn_model = KNeighborsClassifier(n_neighbors=best_k, weights='uniform')
best_knn_model.fit(xtrain, ytrain)

# Draw decision boundary
plot_decision_boundary(best_knn_model, 0)
scatterValue(X1, X2, y, "g", "b", "+", "+1 points", " -1 points", 20)
scatterValue(X1, X2, best_knn_model.predict(x), "r", "k", ".", "predicted +1 points", "predicted -1 points", 10)

plt.legend()
plt.show()

# %%
# confusion matrix of 3 models
def draw_confusion_matrix(model, x_data):
    cm_log = ConfusionMatrixDisplay.from_predictions(ytest, model.predict(x_data))
    cm_log.ax_.set_title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.text(-0.1, -0.1, "Ture\nNegative", fontsize = 10, bbox = dict(facecolor = 'white', alpha = 0.5))
    plt.text(0.9, -0.1,  "False\nPositive", fontsize = 10, bbox = dict(facecolor = 'white', alpha = 0.5))
    plt.text(-0.1, 0.9,  "False\nNegative", fontsize = 10, bbox = dict(facecolor = 'white', alpha = 0.5))
    plt.text(0.9, 0.9,   "Ture\nPositive", fontsize = 10, bbox = dict(facecolor = 'white', alpha = 0.5))
    plt.show()

draw_confusion_matrix(best_log_model, Xtest)
draw_confusion_matrix(best_knn_model, xtest)

# define baseline classifer
base_model = DummyClassifier(strategy="most_frequent")
base_model.fit(xtrain, ytrain)

draw_confusion_matrix(base_model, xtest)

# %%
# ROC curve of 3 models
fpr_log, tpr_log, _ = roc_curve(ytest, best_log_model.predict_proba(Xtest)[:,1])
plt.plot(fpr_log, tpr_log, color='b', label = "Logistic Regression")
print("AUC value of Logistic Regression: ", metrics.auc(fpr_log, tpr_log))

fpr_knn, tpr_knn, _ = roc_curve(ytest, best_knn_model.predict_proba(xtest)[:,1])
plt.plot(fpr_knn, tpr_knn, color='r', label = "kNN classifier")
print("AUC value of kNN classifier: ", metrics.auc(fpr_knn, tpr_knn))

fpr_base, tpr_base, _ = roc_curve(ytest, base_model.predict_proba(xtest)[:,1])
plt.plot(fpr_base, tpr_base, color='y', label = "baseline classifier")
print("AUC value of baseline classifier: ", metrics.auc(fpr_base, tpr_base))

plt.plot([0, 1], [0, 1], color='green',linestyle='--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
#plt.xlim(-0.01, 0.3)
#plt.ylim(0.9, 1.01)
plt.legend()
plt.show()
# %%
