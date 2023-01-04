
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read the data
df = pd.read_csv("week4_data.txt", sep=",", names=["X_1", "X_2", "Y"])
print(df.head())

# get seperate colum of data
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
y = df.iloc[:, 2]

# %%
# Visualise the scatter data and use different colors to distinguish target value
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

scatterValue(X1, X2, y, "g", "b", "+", "+1 points", " -1 points", 20)

plt.legend()
#save figure
plt.savefig('./fig_a1.jpg', dpi=600)
# show the figure
plt.show()

# %%

x = np.column_stack((X1, X2))

# split up our data into train and test
from sklearn.model_selection import train_test_split
Xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

#Xtrain = vectorizer.fit_transform(xtrain)
#Xtest = vectorizer.transform(xtest)

# creat a model and train our data
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(Xtrain, ytrain)

intercept = model.intercept_[0]
coef = model.coef_[0]
print ('Intercept of model: ',intercept)
print ('Coefficient of model: ', coef)
print ('Training Accuracy of the Model: ', model.score(Xtrain, ytrain))
print ('Texting Accuracy of the Model: ', model.score(xtest, ytest))

# %%
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
        if feat_num == 2:
            Z.append(pred_func.predict(np.column_stack((x1_b[i], x2_b[i]))))
        elif feat_num == 4:
            Z.append(pred_func.predict(np.column_stack((x1_b[i], x2_b[i], x1_b[i]**2, x2_b[i]**2))))
    # draw the decision boundary
    print(Z)
    plt.contourf(x1_b, x2_b, Z, alpha = 0.5)

scatterValue(X1, X2, y, "g", "b", "+", "+1 points", " -1 points", 20)
scatterValue(X1, X2, model.predict(x), "r", "k", ".", "predicted +1 points", "predicted -1 points", 10)

plt.axis([-1.2, 1.2, -1.2, 1.2])

# Draw decision boundary
plot_decision_boundary(model,2)
a = np.linspace(-1.2, 1.2, 200)
b = -(intercept + coef[0]*a)/coef[1]
plt.plot(a, b, "k-", linewidth=1)
# show the figure
plt.legend()
#save figure
plt.savefig('./fig_a3.jpg', dpi=600)
plt.show()

# %%
from sklearn import svm
C = [0.001, 1, 100]
intercept = []
train_acu = []
test_acu = []
coef = []
for i in range(0, len(C)):
    model = svm.LinearSVC(C=C[i])
    model.fit(Xtrain, ytrain)
    # calculate the parameter values of each trained model
    intercept.append(model.intercept_[0])
    coef.append(model.coef_[0])
    train_acu.append(model.score(Xtrain, ytrain))
    test_acu.append(model.score(xtest, ytest))
    # draw disision boundary
    plot_decision_boundary(model, 2)
    a = np.linspace(-1.2, 1.2, 200)
    b = -(model.intercept_[0] + model.coef_[0][0]*a)/model.coef_[0][1]
    plt.plot(a, b, "k-", linewidth=1)
    # predict the training data
    scatterValue(X1, X2, y, "g", "b", "+", "+1 points", " -1 points", 20)
    scatterValue(X1, X2, model.predict(x), "r", "k", ".", "predicted +1 points", "predicted -1 points", 10)
    plt.axis([-1.2, 1.2, -1.2, 1.2])
    # show the figure
    plt.legend()
    #save figure
    plt.savefig('./fig_b'+ str(i+1) +'.jpg', dpi=600)
    plt.show()

# print the parameter values of each trained model
print ('Values of the penalty C: ',C)
print ('Coefficient of model: ', coef)
print ('Intercept of model: ',intercept)
print ('Training Accuracy of the Model: ', train_acu)
print ('Texting Accuracy of the Model: ', test_acu)

# %%
# create two additional features by adding the square of each 
x = np.column_stack((X1, X2, X1**2, X2**2))
Xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

model = LogisticRegression()
model.fit(Xtrain, ytrain)

# draw disision boundary
plot_decision_boundary(model, 4)

coef = model.coef_[0]
m = coef[1]/(2*(coef[3]**0.5))
n = coef[0]/(2*(coef[2]**0.5))
r = m**2 + n**2 - model.intercept_[0]
print("m=", m)
print("n=", n)
print("r=", r)

a = np.linspace((-n-r**0.5)/(coef[2]**0.5), (-n+r**0.5)/(coef[2]**0.5), 200)
b = (-m - (r-(a*(coef[2]**0.5)+n)**2)**0.5)/(coef[3]**0.5)
plt.plot(a, b, "k-", linewidth=1)

# predict the training data
scatterValue(X1, X2, y, "g", "b", "+", "+1 points", " -1 points", 20)
scatterValue(X1, X2, model.predict(x), "r", "k", ".", "predicted +1 points", "predicted -1 points", 10)
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.legend()
#save figure
plt.savefig('./fig_c.jpg', dpi=600)
plt.show()

print ('Coefficient of model: ', model.coef_[0])
print ('Intercept of model: ',model.intercept_[0])
print ('Training Accuracy of the Model: ', model.score(Xtrain, ytrain))
print ('Texting Accuracy of the Model: ', model.score(xtest, ytest))

# %%
# most reasonable baseline
def baseline():
    return len(y)*[1]

t_t = 0
for i in range(0, len(y)):
    if y[i] == 1:
        t_t += 1

print("Accuracy of baseline: ", t_t/len(y))
# %%
