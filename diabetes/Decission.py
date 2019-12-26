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
treeclf = DecisionTreeClassifier(random_state = 42)
parameters = {'max_depth': [6, 7, 8, 9], 'min_samples_split': [2, 3, 4, 5], 'max_features': [1, 2, 3, 4] }
gridsearch = GridSearchCV(treeclf, parameters, cv=100, scoring = 'roc_auc')
gridsearch.fit(X,y)
print(gridsearch.best_params_)
print(gridsearch.best_score_)


# Adjusting development threshold
treeclf_model = DecisionTreeClassifier(max_depth = 6, max_features = 4, min_samples_split = 5, random_state = 42 )

X_train,X_test,y_train,y_test = train_test_split(X, y, random_state = 42)

treeclf_model.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(treeclf_model.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(treeclf_model.score(X_test, y_test)))


# # Predicting the Test set results
y_pred_class_treeclf = treeclf_model.predict(X_test)
y_pred_prob_treeclf = treeclf_model.predict_proba(X_test)

print('accuracy of y predict is {:.3f}'.format(accuracy_score(y_test,y_pred_class_treeclf)))
print('roc-auc of y proba is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob_treeclf[:,1])))


def plot_roc(y_test, y_pred, model_name):

    fpr, tpr, thr = roc_curve(y_test, y_pred)
    fig, ax = plt.subplots(figsize = (8, 8))
    ax.plot(fpr, tpr, 'k-')
    ax.plot([0, 1], [0, 1], 'k--', linewidth = 0.5)  # roc curve for random model
    ax.grid(True)
    ax.set(title = 'ROC Curve for {} on diabetes problem'.format(model_name), xlim = [-0.01, 1.01], ylim = [-0.01, 1.01])

plot_roc(y_test, y_pred_prob_treeclf[:, 1], 'RF')
plt.show()


# Feature Scaling
normalizer = StandardScaler()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)


# Neural networks
model = Sequential()
model.add(Dense(500, input_shape=(8,), activation="relu" ))
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
cm = confusion_matrix(y_test, y_pred_class_treeclf)

print('TP - True Negative {}'.format(cm[0,0]))
print('FP - False Positive {}'.format(cm[0,1]))
print('FN - False Negative {}'.format(cm[1,0]))
print('TP - True Positive {}'.format(cm[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))))

round(roc_auc_score(y_test,y_pred_class_treeclf),5)
