import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import pickle 

df=pd.read_csv('Data/Real-Data/Real_Combine.csv')

print(df.isnull())
print(df.head())

# Check for null values
# heat map is used to see null vallues iin this we can see null values in a yellow line
df=df.dropna() # # it will drop all the null values
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis') ## Cbar gives the density of the colour
plt.show()

X=df.iloc[:,:-1] # # independent features
y=df.iloc[:,-1] # # dependent features

## check null values
print(X.isnull())
print(y.isnull())

sns.pairplot(df)
plt.show()
print(df.corr())


#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
##plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()
print(df.corr())
corrmat.index

##Feature Importance
model = ExtraTreesRegressor()
model.fit(X,y)

print(X.head())
sns.distplot(y)
plt.show()

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()
print(model.feature_importances_)

## train and split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


## linear regression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
regressor.coef_
regressor.intercept_
print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_train, y_train)))
print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_test, y_test)))

## cross validation
score=cross_val_score(regressor,X,y,cv=5)
score.mean()

##model evaulation
coeff_df = pd.DataFrame(regressor.coef_,X.columns,columns=['Coefficient'])
coeff_df

prediction=regressor.predict(X_test)
sns.distplot(y_test-prediction)
plt.scatter(y_test,prediction)


## regression evaulation metrics
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

# open a file, where you ant to store the data
file = open('regression_model.pkl', 'wb')
# dump information to that file
pickle.dump(regressor, file)


