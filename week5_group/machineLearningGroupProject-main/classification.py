import os

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import numpy as np
import cv2
from matplotlib import pyplot as plt

num_classes = 2

def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    print(mu, sigma)
    return (data - mu) / sigma

def svm_classification(Xtrain, Xtest, ytrain, ytest):
    #可调参数，对比实验 C
    model = LinearSVC()
    model = model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    return ypred

def knn_classification(Xtrain, Xtest, ytrain, ytest):
    #可调参数，对比实验n_neighbors
    model = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    model = model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    return ypred

def compile_trainingset(positive_path, negative_path):
    train_x = []
    train_y = []
    for filename1 in os.listdir(positive_path):
        filename = os.path.join(positive_path, filename1)
        image = cv2.imread(filename)
        image = cv2.resize(image, (32, 32))
        train_x.append(image)
        train_y.append(1)
    for filename1 in os.listdir(negative_path):
        filename = os.path.join(negative_path, filename1)
        image = cv2.imread(filename)
        image = cv2.resize(image, (32, 32))
        train_x.append(image)
        train_y.append(0)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_x, train_y

def CNN(Xtrain, Xtest, ytrain, ytest):
    model = keras.Sequential()
    model.add(Conv2D(8, (3, 3), padding='same', input_shape=Xtrain.shape[1:], activation='relu'))
    model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=2, padding='same'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=2, padding='same'))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1(0.0001)))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    model.summary()

    batch_size = 128
    epochs = 100
    history = model.fit(Xtrain, ytrain, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    ypred = model.predict(Xtest)
    ypred = np.argmax(ypred, axis=1)
    return history, ypred

def show_metrics(ytest, ypred, feature_name = "", classifier_name = "", parameter_name = "", plt=None):
    label = f"{feature_name}/{classifier_name}/{parameter_name}"
    print(label)
    print(confusion_matrix(ytest, ypred))
    print(classification_report(ytest, ypred))
    if plt is not None:
        # draw roc curve
        fpr, tpr, _ = roc_curve(ytest, ypred)
        plt.plot(fpr, tpr, label=label)

def classification_all_classifiers():
    # for lbp
    lbp_features = np.load("lbp_features.npy")

    lbp_X = lbp_features[:, :-1]
    lbp_y = lbp_features[:, -1]
    lbp_X = standardization(lbp_X)
    Xtrain, Xtest, ytrain, ytest = train_test_split(lbp_X, lbp_y, test_size=0.2, random_state=0)
    print("start lbp svm")
    svm_lbp_ypred = svm_classification(Xtrain, Xtest, ytrain, ytest)
    print("start lbp knn")
    knn_lbp_ypred = knn_classification(Xtrain, Xtest, ytrain, ytest)

    # for hog
    # hog_features = np.load("hog_features.npy")
# 
    # hog_X = hog_features[:, :-1]
    # hog_y = hog_features[:, -1]
    # hog_X = standardization(hog_X)
    # Xtrain, Xtest, ytrain, ytest = train_test_split(hog_X, hog_y, test_size=0.2, random_state=0)
    # print("start hog svm")
    # svm_hog_ypred = svm_classification(Xtrain, Xtest, ytrain, ytest)
    # print("start hog knn")
    # knn_hog_ypred = knn_classification(Xtrain, Xtest, ytrain, ytest)

    # # for hist
    # hist_features = np.load("hist_features.npy")
# 
    # hist_X = hist_features[:, :-1]
    # hist_y = hist_features[:, -1]
    # hist_X = standardization(hist_X)
    # Xtrain, Xtest, ytrain, ytest = train_test_split(hist_X, hist_y, test_size=0.2, random_state=0)
    # print("start hist svm")
    # svm_hist_ypred = svm_classification(Xtrain, Xtest, ytrain, ytest)
    # print("start hist knn")
    # knn_hist_ypred = knn_classification(Xtrain, Xtest, ytrain, ytest)

    # start cnn

    # cnn_X, cnn_y = compile_trainingset("D:/BaiduNetdiskDownload/image/CMEImages/CME_polar_crop",
    #                                        "D:/BaiduNetdiskDownload/image/CMEImages/NoCME_polar_crop")
    # cnn_X = cnn_X.astype("float32") / 255.0
    #
    # cnn_xtrain, cnn_xtest, cnn_ytrain, cnn_ytest = train_test_split(cnn_X, cnn_y, test_size=0.2, random_state=0)
    #
    # cnn_ytrain_1hot = keras.utils.to_categorical(cnn_ytrain, num_classes)
    # history, cnn_ypred = CNN(cnn_xtrain, cnn_xtest, cnn_ytrain_1hot, cnn_xtest)

    plt.figure()
    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True
    # show_metrics(ytest, svm_lbp_ypred, "lbp", "svm", "", plt)
    # show_metrics(ytest, knn_lbp_ypred, "lbp", "knn", "", plt)
    # show_metrics(ytest, svm_hog_ypred, "hog", "svm", "", plt)
    # show_metrics(ytest, knn_hog_ypred, "hog", "knn", "", plt)
    # show_metrics(ytest, svm_hist_ypred, "hist", "svm", "", plt)
    # show_metrics(ytest, knn_hist_ypred, "hist", "knn", "", plt)
    # show_metrics(cnn_ytest, cnn_ypred, "cnn", "cnn", "", plt)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.legend()
    plt.show()

    plt.figure()
    plt.subplot(211)
    plt.plot(history.history['accuracy'], label=f'train accuracy ')
    plt.plot(history.history['val_accuracy'], label=f'val accuracy ')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.subplot(212)
    plt.plot(history.history['loss'], label=f'train loss ')
    plt.plot(history.history['val_loss'], label=f'val loss ')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


classification_all_classifiers()