import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn import metrics

df=pd.read_csv('Data/Real-Data/Real_Combine.csv')
print(df.shape())

# # Check for null values
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()

# # function is used to remove rows and columns with Null/NaN values. By default, this function returns a new DataFrame and the source DataFrame remains unchanged. 
df = df.dropna()

X=df.iloc[:,:-1] ## independent features
y=df.iloc[:,-1] ## dependent features

# # check null values
print('\n null values in X:', X.isnull())
print('\n null values in y:', y.isnull())

# # pair plot
sns.pairplot(df)
plt.show()

# # correlation matrix with heatmap
print('\n correlation:', df.corr())

# # get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (20, 20))
# # plot heat map
g = sns.heatmap(df[top_corr_features].corr(),annot = True, cmap = "RdYlGn")
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
feat_importances = pd.series(model.feature_importances_, index = X.columns)
feat_importances.nlargest(5).plot(kind = 'barh')
plt.show()

# # K Nearest Neighbor Regression
sns.distplot(y)
plt.show()

# # split the data and do train and test on the splitted data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

Regressor = KNeighborsRegressor(n_neighbors = 1)
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
accuracy_rate = []

## will take some time
for i in range(1, 40):
    knn = KNeighborsRegressor(n_neighbors = i)
    score = cross_val_score(knn, X, y, cv = 10, scoring = "neg_mean_squared_error")
    accuracy_rate.append(score.mean())

plt.figure(figsize = (10, 6))
plt.plot(range(1,40), accuracy_rate, color = 'blue', linestyle = 'dashed', marker = 'o', markerfacecolor = 'red', markersize = 10)
# plt.plot(range(1,40), accuracy_rate, color = 'blue', linestyle = 'dashed', marker = 'o', markerfacecolor = 'red', markersize = 10)
plt.title('Accuracy Rate vs K Values')
plt.xlabel('k')
plt.ylabel('Accuracy Rate')
plt.show()

# # First a quick Comparision to our orginal K=1
knn = KNeighborsRegressor(n_neighbors = 1)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

sns.distplot(y_test-prediction)
plt.show()
plt.scatter(y_test,prediction)
plt.show()

# # Regression Evaluation Metrics
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

# open a file, where you ant to store the data
file = open('K_nearest_neighbour_model.pkl', 'wb')

# dump information to that file
pickle.dump(knn, file)

