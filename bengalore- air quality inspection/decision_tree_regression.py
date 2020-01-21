import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from IPython.display import Image  
from sklearn.externals.six import StringIO  
# import pydotplus
import os
from datetime import datetime
from sklearn import metrics
import pickle 

df=pd.read_csv('Data/Real-Data/Real_Combine.csv')
print(df.head())

## Check for null values
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()

df=df.dropna()

X=df.iloc[:,:-1] ## independent features
y=df.iloc[:,-1] ## dependent features

## check null values
print(X.isnull())
print(y.isnull())

sns.pairplot(df)
plt.show()
print(df.corr())

##Correlation Matrix with Heatmap
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()
corrmat.index

##Feature Importance
model = ExtraTreesRegressor()
model.fit(X,y)

print(X.head())
print(model.feature_importances_)

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

##Decision Tree Regressor
sns.distplot(y)
plt.show()

##Train Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


dtree=DecisionTreeRegressor(criterion="mse")
dtree.fit(X_train,y_train)

print("Coefficient of determination R^2 <-- on train set: {}".format(dtree.score(X_train, y_train)))
print("Coefficient of determination R^2 <-- on test set: {}".format(dtree.score(X_test, y_test)))

score=cross_val_score(dtree,X,y,cv=5)
print(score.mean())

##Tree Visualization
features = list(df.columns[:-1])
features

##os
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"
dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

##Model Evaluation
prediction=dtree.predict(X_test)
sns.distplot(y_test-prediction)

plt.scatter(y_test,prediction)

##Hyperparameter Tuning DEcision Tree Regressor
DecisionTreeRegressor()
## Hyper Parameter Optimization

params={"splitter"    : ["best","random"] , "max_depth" : [ 3, 4, 5, 6, 8, 10, 12, 15], "min_samples_leaf" : [ 1,2,3,4,5 ],
"min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],"max_features" : ["auto","log2","sqrt",None ], "max_leaf_nodes":[None,10,20,30,40,50,60,70] }

## Hyperparameter optimization using GridSearchCV
random_search=GridSearchCV(dtree,param_grid=params,scoring='neg_mean_squared_error',n_jobs=-1,cv=10,verbose=3)

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X,y)
timer(start_time) # timing ends here for "start_time" variable 

random_search.best_params_
random_search.best_score_
predictions=random_search.predict(X_test)
sns.distplot(y_test-predictions)
plt.show()

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# open a file, where you ant to store the data
file = open('decision_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(random_search, file)