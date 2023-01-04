# %%
# 1a
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# read the data
df = pd.read_csv("week3_data.txt", sep=",", names=["X_1", "X_2", "Y"])

# get seperate colum of data
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
y = df.iloc[:, 2]

#1a
# 3D scatter the data
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X1, X2, y)

# set the axes label
ax.set_xlabel('X_1')
ax.set_ylabel('X_2')
ax.set_zlabel('y') 

#save figure
plt.savefig('./fig_1a.jpg', dpi=600)

# %%
# 1b
from sklearn.linear_model import Ridge, Lasso, LassoCV, RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import cm

x = np.column_stack((X1, X2))

# add extra polynomial features to power 5
poly = PolynomialFeatures(5)
X = poly.fit_transform(x)

print(poly.get_feature_names(['X_1','X_2']))

# split up our data into train and test
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

C = [1, 10, 100, 1000, 10000]

model = []
for i in range(len(C)):
    # define model
    model.append(Lasso(alpha=1/(2*C[i])))
    # fit the lasso model
    model[i].fit(xtrain, ytrain)
    # the model score
    print("Lasso.score(xtrain, ytrain):", model[i].score(xtrain, ytrain))
    print("Lasso.score(xtest, ytest):", model[i].score(xtest, ytest))
    print("Lasso.coef_:", model[i].coef_)
    # 1c
    # draw the predicted surface
    grid = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(grid, grid)
    # predicte the target value
    Z = []
    for j in range(len(X)):
        a = np.column_stack((X[j], Y[j]))
        Z.append(model[i].predict(poly.fit_transform(a)))
    Z = np.array(Z)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    # draw the predicted surface
    ax.plot_surface(X, Y, Z, linewidth=1, alpha = 0.4, cmap=cm.autumn)
    # set the axes label
    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_zlabel('y') 
    # scatter the origine data
    ax.scatter(X1, X2, y, color = "black")
    #save figure
    plt.savefig('./fig_1c'+str(i+1)+'.jpg', dpi=600)
    plt.show()

# %%
# 1e
C = [0.001, 0.01, 0.1, 1, 10]
model = []
for i in range(len(C)):
    # define model
    model.append(Ridge(alpha=1/(2*C[i])))
    # fit the Ridge model
    model[i].fit(xtrain, ytrain)
    # the model score
    print("Ridge.score(xtrain, ytrain):", model[i].score(xtrain, ytrain))
    print("Ridge.score(xtest, ytest):", model[i].score(xtest, ytest))
    print("Ridge.coef_:", model[i].coef_)
    # 1c
    # draw the predicted surface
    grid = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(grid, grid)
    # predicte the target value
    Z = []
    for j in range(len(X)):
        a = np.column_stack((X[j], Y[j]))
        Z.append(model[i].predict(poly.fit_transform(a)))
    Z = np.array(Z)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    # draw the predicted surface
    from matplotlib import cm
    ax.plot_surface(X, Y, Z, linewidth=1, alpha = 0.4, cmap=cm.autumn)
    # set the axes label
    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_zlabel('y') 
    # scatter the origine data
    ax.scatter(X1, X2, y, color = "black")
    #save figure
    plt.savefig('./fig_1e'+str(i+1)+'.jpg', dpi=600)
    plt.show()

# %%
# 2a
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

x = np.column_stack((X1, X2))
# add extra polynomial features to power 5
poly = PolynomialFeatures(5)
X = poly.fit_transform(x)

C = [0.1, 0.5, 1, 5, 10, 25, 50, 100]

mean_error = []; std_error = []
for i in range(len(C)):
    # define model
    model= Lasso(alpha=1/(2*C[i]))
    kf = KFold(n_splits=5)
    temp = []
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])
        temp.append(mean_squared_error(y[test], ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())

plt.errorbar(C, mean_error, yerr=std_error)
plt.xlabel("C Value")
plt.ylabel("Mean square error")
plt.savefig('./fig_2a.jpg', dpi=600)
plt.show()
    
# %%
# 2c
x = np.column_stack((X1, X2))
# add extra polynomial features to power 5
poly = PolynomialFeatures(5)
X = poly.fit_transform(x)

C = [0.001, 0.01, 0.1, 0.25, 0.5, 1]

mean_error = []; std_error = []
for i in range(len(C)):
    # define model
    model= Ridge(alpha=1/(2*C[i]))
    kf = KFold(n_splits=5)
    temp = []
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])
        temp.append(mean_squared_error(y[test], ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())

plt.errorbar(C, mean_error, yerr=std_error)
plt.xlabel("C Value")
plt.ylabel("Mean square error")
plt.savefig('./fig_2c.jpg', dpi=600)
plt.show()
    
# %%

    