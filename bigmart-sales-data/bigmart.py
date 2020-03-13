import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

# #print dimensions
print("\ntrain - head and test - head is : ")
print(train.head(), test.head())

print("\ntrain - info and test - info is :")
print(train.info(), test.info())

print(((train.isnull() | train.isna()).sum() * 100 / train.index.size).round(2))

# # Distribution of item outlet sales
plt.style.use('fivethirtyeight')
plt.figure(figsize=(12, 7))
sns.distplot(train.Item_Outlet_Sales, bins=25)
plt.ticklabel_format(style='plain', axis='x', scilimits=(0, 1))
plt.xlabel("Item_Outlet_Sales")
plt.ylabel("Number of Sales")
plt.title("Item_Outlet_Sales Distribution")
# plt.show()

# # Distribution of fat content
sns.countplot(train.Item_Fat_Content)
# plt.show() 

# # Distribution of outlet size
sns.countplot(train.Item_Type)
plt.xticks(rotation=90)
sns.countplot(train.Outlet_Size)
# plt.show()

# # Distribution of outlet location type
sns.countplot(train.Outlet_Location_Type)
# plt.show()

# # Distribution of outlet type
sns.countplot(train.Outlet_Type)
plt.xticks(rotation=90)
# plt.show() 

# #Exploratory Data Analysis
train['source']='train'
test['source']='test'

data = pd.concat([train, test], ignore_index = True)
print("dimension of train data: {}".format(train.shape))
print("dimension of test data: {}".format(test.shape))
print("dimension of data: {}".format(data.shape))

print("\ndimension of data head: ")
print(train.head())

# # On seeing the data information, we see that there are many null values, thus we have to remove the 0 values
# # Check for duplicates
Uniqueid = len(set(train.Item_Identifier))
Totalid =train.shape[0]
Duplicate = Totalid - Uniqueid
print("\nThere are " + str(Uniqueid) + " duplicate IDs for " + str(Totalid) + " total entries")

# # Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
# # Exclude ID cols and source
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
# # Print frequency of categories
for col in categorical_columns:
    print ('\nFrequency of Categories for varible %s'%col)
    print (data[col].value_counts())


# # data cleaning
# #about null values
temp_df = train.isnull().sum().reset_index()
temp_df['Percentage'] = (temp_df[0]/len(train))*100
temp_df.columns = ['Column Name', 'Number of null values', 'Null values in percentage']
print(f"\nThe length of dataset is \t {len(train)}")

print("\ntemp null values :\n")
print(temp_df)

# # do the corrections in fat content column
def convert(x):
    if x in ['low fat', 'LF']: 
        return 'Low Fat'
    elif x=='reg':
        return 'Regular'
    else:
        return x

train['Item_Fat_Content'] = train['Item_Fat_Content'].apply(convert)
test['Item_Fat_Content'] = train['Item_Fat_Content'].apply(convert)

print("\nNow Unique values in this column in Train Set are:", train['Item_Fat_Content'].unique())
print("Now Unique values in this column in Test Set are:", test['Item_Fat_Content'].unique())

# # Create table with identifiers and its weights and ignores the NaN values
item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')

def impute_weight(cols):  # Filling null values with mean
    Weight = cols[0]
    Identifier = cols[1]

    if pd.isnull(Weight):
        return item_avg_weight['Item_Weight'][item_avg_weight.index == Identifier]
    else:
        return Weight


print ('\nOrignal missing: %d' % sum(data['Item_Weight'].isnull()))
print ("Mean of size :", data['Item_Weight'].mean())

data['Item_Weight'] = data[['Item_Weight', 'Item_Identifier']].apply(impute_weight, axis=1).astype(float)
print ('\nFinal missing: %d' % sum(data['Item_Weight'].isnull()))
print ("Mean of size :",data['Item_Weight'].mean())

# # Determing the mode for each
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=lambda x: x.mode())

def impute_size_mode(cols):  # Same as above but using mode
    Size = cols[0]
    Type = cols[1]
    if pd.isnull(Size):
        return outlet_size_mode.loc['Outlet_Size'][outlet_size_mode.columns == Type][0]
    else:
        return Size

# # Impute data and check #missing values before and after imputation to confirm
print ('\nOrignal missing: %d' % sum(data['Outlet_Size'].isnull()))
data['Outlet_Size'] = data[['Outlet_Size', 'Outlet_Type']].apply(impute_size_mode, axis=1)
print ('\nFinal missing: %d' % sum(data['Outlet_Size'].isnull()))

# #calculating mean value to replace the null values
avg_mean = data.pivot_table(index = 'Item_Identifier', values = 'Item_Weight')
print(avg_mean)

# # Visibility
visibility_item_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')

def impute_visibility_mean(cols):
    visibility = cols[0]
    item = cols[1]
    if visibility == 0:
        return visibility_item_avg['Item_Visibility'][visibility_item_avg.index == item]
    else:
        return visibility

print ('\nOriginal zeros: %d' % sum(data['Item_Visibility'] == 0))

data['Item_Visibility'] = data[['Item_Visibility', 'Item_Identifier']].apply(impute_visibility_mean, axis=1).astype(float)
print ('Final zeros: %d' % sum(data['Item_Visibility'] == 0))

# Remember the data is from 2013
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']

print("\ndata describe :")
print(data['Outlet_Years'].describe())

# # Get the first two characters of ID
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])

# # Rename them to more intuitive categories
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'})
data['Item_Type_Combined'].value_counts()
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})

print("\nvalue counts of Item_Fat_Content Edible:")
print(data['Item_Fat_Content'].value_counts())

# # make a new column depicting the years of operation of a store
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()

# Mark non-consumables as separate category in low_fat:
data.loc[data['Item_Type_Combined'] == "Non-Consumable", 'Item_Fat_Content'] = "Non-Edible"

print("\nvalue counts of Item_Fat_Content Non-Edible:")
print(data['Item_Fat_Content'].value_counts())

def func(x):
    return x['Item_Visibility'] / visibility_item_avg['Item_Visibility'][visibility_item_avg.index == x['Item_Identifier']][0]

data['Item_Visibility_MeanRatio'] = data.apply(func, axis=1).astype(float)

print("\nitem visibility describe:")
print(data['Item_Visibility_MeanRatio'].describe())

print('\nSkewness: %f' %train['Item_Outlet_Sales'].skew(), "highly skewed")
print('Kurtsis: %f' %data['Item_Outlet_Sales'].kurt())

# #imputing Missing values by label encoder and one hot encoder
from sklearn.preprocessing import LabelEncoder

le  = LabelEncoder()
encode = le.fit_transform(train['Item_Identifier'])
print("\nitem identifier encode:", encode)
print("mean of Item_Weight:", train['Item_Weight'].fillna(encode.mean(),inplace = True))
print("Item_Weight sum :", train.Item_Weight.isna().sum())

print("\noutlet size of fill :", train['Outlet_Size'].fillna('Small',inplace  = True))
print("sum of outlet size :", train['Outlet_Size'].isna().sum())

train['Item_Visibility'].plot(kind = 'hist',bins = 200)
# # plt.show()

# #replace the Zero values
encode = train[train['Item_Visibility']!=0]['Item_Visibility'].mean()
train['Item_Visibility'] = train['Item_Visibility'].replace(0.00,encode)

train['Item_Visibility'].plot(kind = 'hist',bins = 200)
# # plt.show()

# # #Feature engineering and scaling
food = ["Breads", "Breakfast", "Dairy", "Fruits and Vegetables", "Meat", "Seafood"]
non_food = ["Baking Goods", "Canned", "Frozen Foods", "Hard Drinks", "Health and Hygiene", "Household", "Soft Drinks"]

# #crate new features
item_list =[] 
for i in train['Item_Type']:
    if i in food:
        item_list.append('food')
    elif (i in non_food):
        item_list.append('non_food')
    else:
        item_list.append('not_sure')
        
train['Item_Type_new'] = item_list
train['Item_Category'] =train['Item_Identifier'].replace({'^DR[A-Z]*[0-9]*':'DR','^FD[A-Z]*[0-9]*':'FD','^NC[A-Z]*[0-9]*':'NC'},regex = True)

data = pd.crosstab(train['Item_Type'],train['Item_Category'])
print("\nNew Features:")
print(data)

def clusters(x):
    if x<69:
        return '1st'
    elif x in range(69,136):
        return '2nd'
    elif x in range(136,203):
        return '3rd'
    else:
        return '4th'
train['Item_MRP_Clusters'] = train['Item_MRP'].astype('int').apply(clusters)

print("\nitem MRP clusters Head:")
print(train.head())


""" ---------------- Grident Decient ------------------- """

from numpy import *

X = train.drop(columns=['Item_Outlet_Sales']).values
y = train['Item_Outlet_Sales'].values

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():
    points = genfromtxt("train.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

if __name__ == '__main__':
    run()



""" ----------------------------- Regression ----------------------------- """

# #Exploratory Data Analysis
train['source']='train'
test['source']='test'

data = pd.concat([train, test], ignore_index = True)

# # Exporting the data back
# #Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

# #Drop unnecessary columns:
X_train = train.drop(['Item_Outlet_Sales', 'Outlet_Identifier','Item_Identifier'], axis=1)
y_train = train.Item_Outlet_Sales

X_test = test.drop(['Outlet_Identifier','Item_Identifier'], axis=1)


""" Linear Regression """

# # Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pandas import Series, DataFrame

regressor = LinearRegression()

train['Item_Weight'].fillna((train['Item_Weight'].mean()), inplace=True)

X = train.loc[:,['Outlet_Establishment_Year', 'Item_MRP', 'Item_Weight']]
X_train, X_test, y_train, y_test = train_test_split(X,train.Item_Outlet_Sales)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print("\npredict of y :")
print( y_pred[0:5])

print("\ntest of y :")
print( y_test[0:5])

print("\nActual and predicated values :")
print(pd.DataFrame({'Actual': y_test, 'Predicted': y_test}))

mse = np.mean((y_pred - y_test)**2)
print("mean square error is :", mse)

## calculating coefficients
coeff = DataFrame(X_train.columns)
coeff['Coefficient Estimate'] = Series(regressor.coef_)

print("\ncoeff of X-train columns:")
print(coeff['Coefficient Estimate'])

# print("\nregressor score :", regressor.score(X_test, y_test))



""" ----------------------------- Neural Networks ----------------------------------- """

#lets replace NaN with Medium

train.drop(axis=1,columns=['Item_Identifier','Outlet_Identifier'],inplace = True)

train['Item_Visibility'].replace(to_replace=0, value= train['Item_Visibility'].quantile(q=.10),inplace = True)

train['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'},inplace = True)

train['Item_Weight'].fillna(train['Item_Weight'].mean(), inplace = True)
train['Outlet_Size'].fillna(value='Medium',inplace = True)

train['Outlet_Establishment_Year']=(2013-train['Outlet_Establishment_Year'])

dummy = pd.get_dummies(train, columns=['Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type'],drop_first=True)
train = dummy

# # drop the columns
X = train.drop(columns=['Item_Outlet_Sales']).values
y = train['Item_Outlet_Sales'].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))
# print(X,y)

from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout

#splitting data into train and test
from sklearn.model_selection import train_test_split
# # split the data and do train and test on the splitted data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

model = Sequential()
# The Input Layer :
model.add(Dense(128, kernel_initializer = 'normal', input_dim = X_train.shape[1], activation = 'relu'))

# The Hidden Layers
model.add(Dense(256, kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense(256, kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense(256, kernel_initializer = 'normal', activation = 'relu'))

# The Output Layer
model.add(Dense(1, kernel_initializer = 'normal', activation = 'linear'))

# Compile the network
model.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics = ['mean_absolute_error'])
model.summary()

# # Fitting the ANN to the Training set
model_history = model.fit(X_train, y_train, epochs = 10, batch_size = 64, validation_split = 0.2)

# # Model Evaluation
prediction = model.predict(X_test)

print("\nprediction is:")
print(prediction)

plt.scatter(y_test,prediction)
plt.show()

# # Regression Evaluation Metrics
from sklearn import metrics

print('\nMAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

