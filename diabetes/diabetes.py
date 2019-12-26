import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics 
from numpy import genfromtxt, array
import fancyimpute as imp
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Conv2D,MaxPooling2D, Flatten







#read and print the file and show info
data = pd.read_csv('diabetes.csv')

#print dimensions healthy and unhealth peoples
print(data.head(5))
print(data.info())
print("dimension of diabetes data: {}".format(data.shape))
print(data.groupby('Outcome').size())

""" graphs"""

corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()
print(data.corr())



# plot the count and scatter plot graph
sns.countplot(data['Outcome'], label = "Count")
plt.show()

# graph between glucose vs glucose, insulin etc like one vs all
g = sns.PairGrid(data, vars = ['Glucose', 'Insulin', 'BMI', 'Age', 'SkinThickness', 'BloodPressure', 'Pregnancies', 'DiabetesPedigreeFunction'], hue="Outcome", size=2.4)
g.map_diag(plt.hist)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.add_legend()
plt.show()


columns = ['Glucose', 'Age', 'BloodPressure', 'Insulin','BMI','SkinThickness' ,'Pregnancies', 'DiabetesPedigreeFunction']
n_cols = 2
n_rows = 4
idx = 0

for i in range(n_rows):
    fg,ax = plt.subplots(nrows = 1, ncols = n_cols, sharey = True, figsize = (8, 2.4))
    for j in range(n_cols):
        sns.violinplot(x = data.Outcome, y = data[columns[idx]], ax = ax[j]) 
        idx += 1
        if idx >= 8:
            break

sns.heatmap(data.corr(),cmap = 'YlGnBu') # YlGnBu means colors

data.plot.scatter(x = 'Pregnancies', y = 'Glucose')
data.plot.scatter(x = 'BloodPressure', y = 'SkinThickness')
data.plot.scatter(x = 'BMI', y = 'DiabetesPedigreeFunction')

data = data.drop(data.index[data.Pregnancies >= 11.5], axis = 0)
data = data.drop(data.index[data.Glucose >= 185], axis = 0)
data = data.drop(data.index[data.BloodPressure >= 92], axis = 0)
data = data.drop(data.index[data.BloodPressure <= 37], axis = 0)
data = data.drop(data.index[data.BMI >= 45], axis = 0)
data = data.drop(data.index[data.DiabetesPedigreeFunction >= 1.2], axis = 0)
plt.show()


# Import the dataset and initilize the x and y values

X = data.iloc[:, :-1].values
y = data.iloc[:, 8].values

# # Spliting the Dataset into Training test and Test test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""------------- Classifiers (KNN, Gausian, Decission tress, Randam Forest, Confuse Matrix ETC.) ---------------- """

"""------- KNN --------"""

# Parameter evaluation
knnclf = KNeighborsClassifier()
parameters = {'n_neighbors': range(1, 20)}
gridsearch = GridSearchCV(knnclf, parameters, cv=100, scoring = 'roc_auc')
gridsearch.fit(X, y)
print(gridsearch.best_params_)
print(gridsearch.best_score_)

# Fitting K-NN to the Training set
knnClassifier = KNeighborsClassifier(n_neighbors = 18)
knnClassifier.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knnClassifier.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knnClassifier.score(X_test, y_test)))

# Predicting the Test set results
y_pred = knnClassifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print('TP - True Negative {}'.format(cm[0,0]))
print('FP - False Positive {}'.format(cm[0,1]))
print('FN - False Negative {}'.format(cm[1,0]))
print('TP - True Positive {}'.format(cm[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))))

round(roc_auc_score(y_test,y_pred),5)


"""----- Decission tress Classifier ----- """

# Parameter evaluation
treeclf = DecisionTreeClassifier(random_state = 42)
parameters = {'max_depth': [6, 7, 8, 9], 'min_samples_split': [2, 3, 4, 5], 'max_features': [1, 2, 3, 4] }
gridsearch = GridSearchCV(treeclf, parameters, cv=100, scoring = 'roc_auc')
gridsearch.fit(X,y)
print(gridsearch.best_params_)
print(gridsearch.best_score_)

# Adjusting development threshold
tree = DecisionTreeClassifier(max_depth = 6, max_features = 4, min_samples_split = 5, random_state = 42 )
X_train,X_test,y_train,y_test = train_test_split(X, y, random_state = 42)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# Predicting the Test set results
y_pred = tree.predict(X_test) 

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print('TP - True Negative {}'.format(cm[0,0]))
print('FP - False Positive {}'.format(cm[0,1]))
print('FN - False Negative {}'.format(cm[1,0]))
print('TP - True Positive {}'.format(cm[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))))

round(roc_auc_score(y_test,y_pred),5)


"""  logistic regression  """

# Parameter evaluation
logclf = LogisticRegression(random_state = 42)
parameters = {'C': [1, 4, 10], 'penalty': ['l1', 'l2']}
gridsearch = GridSearchCV(logclf, parameters, cv = 100, scoring = 'roc_auc')
gridsearch.fit(X, y)
print(gridsearch.best_params_)
print(gridsearch.best_score_)

# Adjusting development threshold
logreg_classifier = LogisticRegression(C = 1, penalty = 'l1')
X_train,X_test,y_train, y_test = train_test_split(X, y, random_state=42)
logreg_classifier.fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg_classifier.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg_classifier.score(X_test, y_test)))

# Predicting the Test set results
y_pred = logreg_classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print('TP - True Negative %s' %cm[0,0])
print('FP - False Positive %s' %cm[0,1])
print('FN - False Negative %s' %cm[1,0])
print('TP - True Positive %s' %cm[1,1])
print('Accuracy Rate: %s' %np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm)))
print('Misclassification Rate: %s' %np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm)))

round(roc_auc_score(y_test,y_pred),5)


"""  Randam forest Classifier  """

# Parameter evaluation
rfclf = RandomForestClassifier(random_state = 42)
parameters={'n_estimators': [50, 100],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth' : [4,5,6,7],
            'criterion' :['gini', 'entropy'] }
gridsearch = GridSearchCV(rfclf, parameters, cv = 50, scoring = 'roc_auc', n_jobs = -1)
gridsearch.fit(X, y)
print(gridsearch.best_params_)
print(gridsearch.best_score_)


rf = RandomForestClassifier(n_estimators = 100, criterion = 'gini', max_depth = 6, max_features = 'auto', random_state = 0)
rf.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))

y_pred = rf.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(f'TP - True Negative {cm[0,0]}')
print(f'FP - False Positive {cm[0,1]}')
print(f'FN - False Negative {cm[1,0]}')
print(f'TP - True Positive {cm[1,1]}')
print(f'Accuracy Rate: {np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))}')
print(f'Misclassification Rate: {np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))}')

round(roc_auc_score(y_test,y_pred),5)


"""----- SVM CLASS ------"""

#svm with grid search
svm = SVC(random_state = 42)
parameters = {'kernel':('linear', 'rbf'), 'C':(1,0.25,0.5,0.75),
              'gamma': (1,2,3,'auto'),'decision_function_shape':('ovo','ovr'),
              'shrinking':(True,False)}

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    svm = GridSearchCV(SVC(), parameters, cv=5, scoring='%s_macro' % score)
    svm.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(svm.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = svm.cv_results_['mean_test_score']
    stds = svm.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, svm.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, svm.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

svm_model = SVC(kernel='rbf', C=100, gamma = 0.0001, random_state=42)
svm_model.fit(X_train, y_train)
spred = svm_model.predict(X_test)
print ('Accuracy with SVM {0}'.format(accuracy_score(spred, y_test) * 100))


# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print('TP - True Negative {}'.format(cm[0,0]))
print('FP - False Positive {}'.format(cm[0,1]))
print('FN - False Negative {}'.format(cm[1,0]))
print('TP - True Positive {}'.format(cm[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))))

round(roc_auc_score(y_test,y_pred),5)



"""----- GRADIENT BOOST ----- """

# Parameter evaluation with GSC validation
gbe = GradientBoostingClassifier(random_state=42)
parameters={'learning_rate': [0.05, 0.1, 0.5],
            'max_features': [0.5, 1],
            'max_depth': [3, 4, 5]
}
gridsearch=GridSearchCV(gbe, parameters, cv=100, scoring='roc_auc')
gridsearch.fit(X, y)
print(gridsearch.best_params_)
print(gridsearch.best_score_)

# Adjusting development threshold
gbi = GradientBoostingClassifier(learning_rate=0.05, max_depth=3, max_features=0.5, random_state=42)
X_train,X_test,y_train, y_test = train_test_split(X, y, random_state=42)
gbi.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbi.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbi.score(X_test, y_test)))

# Storing the prediction
y_pred = gbi.predict_proba(X_test)[:,1]

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred.round())

print('TP - True Negative {}'.format(cm[0,0]))
print('FP - False Positive {}'.format(cm[0,1]))
print('FN - False Negative {}'.format(cm[1,0]))
print('TP - True Positive {}'.format(cm[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))))


# Plotting the predictions
plt.hist(y_pred,bins=10)
plt.xlim(0,1)
plt.xlabel("Predicted Proababilities")
plt.ylabel("Frequency")

round(roc_auc_score(y_test,y_pred),5)



""" ----- Gaussian classifier ----- """

classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred=classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print('TP - True Negative {}'.format(cm[0,0]))
print('FP - False Positive {}'.format(cm[0,1]))
print('FN - False Negative {}'.format(cm[1,0]))
print('TP - True Positive {}'.format(cm[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))))

round(roc_auc_score(y_test,y_pred),5)

