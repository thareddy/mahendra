import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint


#read and print the file and show info
data = pd.read_csv('diabetes.csv')

#print dimensions healthy and unhealth peoples
print(data.head())
print(data.describe())
print("dimension of diabetes data: {}".format(data.shape))



# let's remove the 0-entries for these fields
zero_fields = [ 'Pregnancies','Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction','Age']

def check_zero_entries(data, fields):
    # List number of 0-entries in each of the given fields
    for field in fields:
        print('field %s: num 0-entries: %d' % (field, len(data.loc[ data[field] == 0, field ])))

check_zero_entries(data, zero_fields)


# First - split into Train/Test

features = list(data.columns.values)
features.remove('Outcome')
print(features)
X = data[features]
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

print('X tarin data:', X_train.shape)
print('X test data: ' ,X_test.shape)


# lets fix the 0-entry for a field in the dataset with its mean value
def impute_zero_field(data, field):
    nonzero_vals = data.loc[data[field] != 0, field]
    avg = np.sum(nonzero_vals) / len(nonzero_vals)
    k = len(data.loc[ data[field] == 0, field])   # num of 0-entries
    data.loc[ data[field] == 0, field ] = avg
    print('Field: %s; fixed %d entries with value: %.3f' % (field, k, avg))

# Fix it for Train dataset
for field in zero_fields:
    impute_zero_field(X_train, field)

# double check for the Train dataset
check_zero_entries(X_train, zero_fields)

# Fix for Test dataset
for field in zero_fields:
    impute_zero_field(X_test, field)

# double check for the Test dataset
check_zero_entries(X_test, zero_fields)

# Feature Scaling
normalizer = StandardScaler()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)


## Create our model
model = Sequential()

# 1st layer: 12 nodes, RELU
model.add(Dense(500, input_shape=(8,), activation='relu'))
# 2nd layer: 8 nodes, RELU
model.add(Dense(8, activation = 'relu'))
# output layer: dim=1, activation sigmoid
model.add(Dense(1, activation = 'sigmoid' ))
print (model.summary())

# Compile the model     # since we are predicting 0/1
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# checkpoint: store the best model
ckpt_model = 'pima-weights.best.hdf5'
checkpoint = ModelCheckpoint(ckpt_model, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
callbacks_list = [checkpoint]

print('Starting training...')
# train the model, store the results for plotting
history = model.fit(X_train, y_train, batch_size=200, epochs=100, verbose=1, validation_data=(X_test, y_test))


# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

# Model Losss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# print final accuracy
scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



