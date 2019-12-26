import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from sklearn import metrics 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import genfromtxt, array
import fancyimpute as imp
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Conv2D,MaxPooling2D, Flatten

data = pd.read_csv('diabetes.csv')

# Import the dataset and initilize the x and y values

X = data.iloc[:, :-1].values
y = data.iloc[:, 8].values


# # Spliting the Dataset into Training test and Test test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

np.mean(y), np.mean(1-y)

# Parameter evaluation
knnclf = KNeighborsClassifier()
parameters = {'n_neighbors': range(1, 20)}
gridsearch = GridSearchCV(knnclf, parameters, cv=100, scoring = 'roc_auc')
gridsearch.fit(X, y)
print(gridsearch.best_params_)
print(gridsearch.best_score_)

# Fitting K-NN to the Training set
knnclf_model = KNeighborsClassifier(n_neighbors = 18)
knnclf_model.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knnclf_model.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knnclf_model.score(X_test, y_test)))

# # Predicting the Test set results
# y_pred = knnclf.predict(X_test)
y_pred_class_knnclf = knnclf_model.predict(X_test)
y_pred_prob_knnclf = knnclf_model.predict_proba(X_test)

print('accuracy of y predict is {:.3f}'.format(accuracy_score(y_test,y_pred_class_knnclf)))
print('roc-auc of y proba is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob_knnclf[:,1])))


def plot_roc(y_test, y_pred, model_name):

    fpr, tpr, thr = roc_curve(y_test, y_pred)
    fig, ax = plt.subplots(figsize = (8, 8))
    ax.plot(fpr, tpr, 'k-')
    ax.plot([0, 1], [0, 1], 'k--', linewidth = 0.5)  # roc curve for random model
    ax.grid(True)
    ax.set(title = 'ROC Curve for {} on diabetes problem'.format(model_name), xlim = [-0.01, 1.01], ylim = [-0.01, 1.01])

plot_roc(y_test, y_pred_prob_knnclf[:, 1], 'RF')
plt.show()


# Feature Scaling
normalizer = StandardScaler()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)

""" model ---- 1 """

# Neural networks
model = Sequential()
model.add(Dense(100, input_shape=(8,), activation="relu" ))
model.add(Dense(20, activation="relu" ))
model.add(Dense(1, activation="sigmoid" ))
print(model.summary())

model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
hist = model.fit(X_train_norm, y_train, batch_size=200, epochs=100, verbose=1, validation_data=(X_test_norm, y_test))

y_pred_class_nn = model.predict_classes(X_test_norm)
y_pred_prob_nn = model.predict(X_test_norm)

print(y_pred_class_nn[:10])
print(y_pred_prob_nn[:10])

# Print model performance and plot the roc curve
print('accuracy of y predict is {:.3f}'.format(accuracy_score(y_test,y_pred_class_nn)))
print('roc-auc of y proba is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob_nn)))

plot_roc(y_test, y_pred_prob_nn, 'NN')
plt.show()

# value loss graphs

hist.history.keys()

n = len(hist.history["loss"])
m = len(hist.history['loss'])
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(range(n), hist.history["loss"],'r', marker='.', label="Train Loss - Run 1")
ax.plot(range(n, n+m), hist.history["loss"], 'hotpink', marker='.', label="Train Loss - Run 2")

ax.plot(range(n), hist.history["val_loss"],'b', marker='.', label="Validation Loss - Run 1")
ax.plot(range(n, n+m), hist.history["val_loss"], 'LightSkyBlue', marker='.',  label="Validation Loss - Run 2")

ax.legend()
plt.show()


# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class_knnclf)

print('TP - True Negative {}'.format(cm[0,0]))
print('FP - False Positive {}'.format(cm[0,1]))
print('FN - False Negative {}'.format(cm[1,0]))
print('TP - True Positive {}'.format(cm[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))))

round(roc_auc_score(y_test,y_pred_class_knnclf),5)




# """ model ---- 2 """

# # Feature Scaling
# normalizer = StandardScaler()
# X_train = normalizer.fit_transform(X_train)
# X_test = normalizer.transform(X_test)

# # # train and test the model

# X_train.dump('X_train.dat')   # X_train.dump('X_train.dat')
# X_test.dump('X_test.dat')     # X_test.dump('X_test.dat')
# y_train.dump('y_train.dat')   # y_train.dump('y_train.dat')
# y_test.dump('y_test.dat')     # y_test.dump('y_test.dat')


# def plot_roc(y_test, y_pred, model_name):

#     fpr, tpr, thr = roc_curve(y_test, y_pred)
#     fig, ax = plt.subplots(figsize = (8, 8))
#     ax.plot(fpr, tpr, 'k-')
#     ax.plot([0, 1], [0, 1], 'k--', linewidth = 0.5)  # roc curve for random model
#     ax.grid(True)
#     ax.set(title = 'ROC Curve for {} on diabetes problem'.format(model_name), xlim = [-0.01, 1.01], ylim = [-0.01, 1.01])

# plot_roc(y_test, y_pred_prob_knnclf[:, 1], 'RF')
# plt.show()

# def train_model():
#     X_train = np.load('X_train.dat', allow_pickle = True)
#     y_train = np.load('y_train.dat', allow_pickle = True)

#     model = Sequential()
#     model.add ( Dense(400, input_shape=(8,), activation = "relu" ) )  # z1 = (W1 * x) + b1 and in input shape at last " , " represents 1-D
#     model.add ( Dense(100, activation = "relu" ) )
#     model.add ( Dense(1, activation = "softmax") )
#     print (model.summary())

#     model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
#     hist = model.fit(X_train, y_train, batch_size=200, epochs=100, verbose=1, validation_data=(X_test, y_test))

#     model.save('model.h5')


# def predict_on_test_data():
#     X_test = np.load('X_test.dat', allow_pickle = True) # allow_pickel = true when numpy verson is greater 1.0v
#     y_test = np.load('y_test.dat', allow_pickle = True)

#     # y_test = to_categorical(y_test)
#     print('X_test.shape:',X_test.shape,'y_test.shape', y_test.shape)

#     model = load_model('model.h5') # nn model in keras save in h55
#     y_pred = list(model.predict_classes(X_test))
#     print('y_pred:',len(y_pred))

#     acc = 0
#     for el1, el2 in zip(y_test, y_pred):
#         if el1 == el2:
#             acc += 1
#     print(acc)


# if __name__ == "__main__":
#     train_model()
#     predict_on_test_data()