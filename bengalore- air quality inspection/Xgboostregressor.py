import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn import metrics

df = pd.read_csv('Data/Real-Data/Real_Combine.csv')
print('shape of the data:', df.shape())

# # Check for null values
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
plt.show()

df = df.dropna()

X = df.iloc[:, :-1] ## independent features
y = df.iloc[:, -1] ## dependent features

# # Check null values
print('\n null values in X:', X.isnull())
print('\n null values in y:', y.isnull())

sns.pairplot(df)
plt.show()

# # correlation matrix with heatmap
print('\n correlation:', df.corr())
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (20, 20))
# #plot heat map
g = sns.heatmap(df[top_corr_features].corr(), annot = True, cmap = 'RdYlGn')
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
feat_importance = pd.deries(model.feature_importances_, index = X.columns)
feat_importance.nlargest(5).plt(kind = 'barh')
plt.show()

# # Linear Regression
sns.distplot(y)
plt.show()

# # split the data and do train and test on the splitted data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

Regressor = xgb.XGDRegressor()
Regressor.fit(X_train, y_train)

print("coefficient of determination R^2 on train set: {}".format(Regressor.score(X_train, y_train)))
print("coefficient of determination R^2 on test set: {}".format(Regressor.score(X_test, y_test)))

score = cross_val_score(Regressor, X, y, cv = 5)
print('\n mean of the score:', score.mean())

# # Model Evaulation
prediction = Regressor.predict(X_test)

sns.distplot(y_test-prediction)
plt.show()

plt.scatter(y_test, prediction)
plt.show()

# # Hyperparameter Tuning
xgb.XGDRegressor()

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)

 #Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Various learning rate parameters
learning_rate = ['0.05','0.1', '0.2','0.3','0.5','0.6']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
#Subssample parameter values
subsample=[0.7,0.6,0.8]
# Minimum child weight parameters
min_child_weight=[3,4,5,6,7]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'learning_rate': learning_rate,
               'max_depth': max_depth,
               'subsample': subsample,
               'min_child_weight': min_child_weight}

print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
regressor=xgb.XGBRegressor()

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
xg_random = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)

xg_random.best_params_
xg_random.best_score_

predictions=xg_random.predict(X_test)
sns.distplot(y_test-prediction)
plt.show()
plt.scatter(y_test,prediction)
plt.show()

# # Regression Evaluation Metrics
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

# open a file, where you ant to store the data
file = open('Xgb_boost.pkl', 'wb')

# dump information to that file
pickle.dump(regressor, file)







