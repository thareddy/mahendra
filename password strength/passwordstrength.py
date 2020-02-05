import pandas as pd
import numpy as np
import random

data = pd.read_csv('data.csv',',', error_bad_lines = False)
print(data.head())
print("\n Null:", data[data['password'].isnull()])
print(data.dropna(inplace = True))

passwords_tuple = np.array(data)
print("\n Tuple Password :", passwords_tuple)

# # shuffling randomly for robustness
random.shuffle(passwords_tuple)

y=[labels[1] for labels in passwords_tuple]
X=[labels[0] for labels in passwords_tuple]

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
sns.countplot(x='strength',data=data,palette='RdBu_r')
plt.show()


def word_divide_char(inputs):
    characters=[]
    for i in inputs:
        characters.append(i)
    return characters

# # Tfidf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(tokenizer = word_divide_char)
X = vectorizer.fit_transform(X)

print("\n shape of X is :", X.shape)
print("\n vocabulary is:", vectorizer.vocabulary_)

print("\ndata ot 0 :", data.iloc[0,0])


feature_names = vectorizer.get_feature_names() 
#get tfidf vector for first document
first_document_vector=X[0]
 
#print the scores
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
print(df.sort_values(by=["tfidf"],ascending=False))

# # logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# # splitting 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

log_class = LogisticRegression(penalty = 'l2', multi_class = 'ovr')
log_class.fit(X_train, y_train)

print(log_class.score(X_test, y_test))

# # Multinomial
clf = LogisticRegression(random_state = 0, multi_class = 'multinomial', solver = 'newton-cg')
# # training
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

X_predict=np.array(["%@123abcd"])
X_predict=vectorizer.transform(X_predict)
y_pred=log_class.predict(X_predict)
print(y_pred)

