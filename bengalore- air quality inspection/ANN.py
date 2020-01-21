import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle 

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


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

model = Sequential()
# The Input Layer :
model.add(Dense(128, kernel_initializer = 'normal', input_dim = X_train.shape[1], activation = 'relu'))

# The Hidden Layers :
model.add(Dense(256, kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense(256, kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense(256, kernel_initializer = 'normal', activation = 'relu'))

# The Output Layer :
model.add(Dense(1, kernel_initializer = 'normal', activation = 'linear'))

# Compile the network :
model.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics = ['mean_absolute_error'])
model.summary()

# Fitting the ANN to the Training set
model_history = model.fit(X_train, y_train, validation_split = 0.33, batch_size = 10, nb_epoch = 100)

#### Model Evaluation
prediction = model.predict(X_test)

sns.distplot(y_test.values.reshape(-1,1)-prediction)
plt.show()

plt.scatter(y_test,prediction)
plt.show()

# # Regression Evaluation Metrics
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

# open a file, where you ant to store the data
file = open('neural.pkl', 'wb')

# dump information to that file
pickle.dump(model, file)



