import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn import metrics

df=pd.read_csv('Data/Real-Data/Real_Combine.csv')
print(df.head())

# # Check for null values
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
plt.show()

 ## function is used to remove rows and columns with Null/NaN values. By default, this function returns a new DataFrame and the source DataFrame remains unchanged. 
df = df.dropna()

X = df.iloc[:, :-1] ## independent features
y = df.iloc[:, -1]  ## dependent features

# # checking null values
print('\n null valuse in X:', X.isnull())
print('\n null valuse in y:', y.isnull())

sns.pairplot(df)
plt.show()

# # correlation matrix with heatmap
print('\n correlation:', df.corr())

corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (20,20))
# # plot the graph
g = sns.heatmap(df[top_corr_features].corr(), annot = True, cmap = "RdYlGn")
plt.show()

print('\n corrmat index:', corrmat.index)

# # Feature Importance
"""Feature importance is an inbuilt class that comes with 
   Tree Based Regressor, we will be using Extra Tree Regressor 
   for extracting the top 10 features for the dataset. """

model = ExtraTreesRegressor()
model.fit(X, y)
print('\n Head of X:')
print(X.head())
print('\n feature importance:', model.feature_importances_)

# # plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

# # Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)

print("coefficient of determation r^2 on train set: {}".format(regressor.score(X_train, y_train)))
print("coefficient of determation r^2 on test set: {}".format(regressor.score(X_test, y_test)))

score = cross_val_score(regressor, X, y, cv = 5)
print('\n mean of the score:', score.mean())

# # Model Evaulation
prediction = regressor.predict(X_test)

sns.distplot(y_test-prediction)
plt.show()

plt.scatter(y_test, prediction)
plt.show()

# # Hyperparameter Tuning
# # Randomized Search CV
# Number of trees in random forest
n_estimators = [int(X) for X in np.linspace(start = 100, stop = 1200, num = 12)]
print('\n estimatrs:', n_estimators)

# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
# Method of selecting samples for training each tree
# bootstrap = [True, False]


# Create the random grid
random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
               'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}

print("\n randam grid:", random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring = 'neg_mean_squared_error',
                                n_iter = 10, cv = 5, verbose = 2, random_state = 42, n_jobs = 1)
rf_random.fit(X_train,y_train)

print('\n random best params:', rf_random.best_params_)
print('\n random best score:', rf_random.best_score_)

sns.distplot(y_test-prediction)
plt.show()
plt.scatter(y_test,prediction)
plt.show()

# # Regression Evaluation Metrics
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)


