# -*- coding: utf-8 -*-
"""Diabetics_prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14pKfsdIDGPIZLdT5qSHF1-906xiIu77I

### Importing the dependencies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetics_dataset = pd.read_csv('/diabetes2.csv')
diabetics_dataset.head()

diabetics_dataset.tail()

diabetics_dataset.shape

diabetics_dataset.info()

diabetics_dataset.describe()

diabetics_dataset['Outcome'].value_counts()

"""**1 - > Diabetics**

**0 - > Non-diabetics**
"""

sns.countplot('Outcome', data=diabetics_dataset)

diabetics_dataset.groupby('Outcome').mean()

plt.figure(figsize=(8,8))
sns.heatmap(diabetics_dataset.corr(), cbar=True, square=True, fmt = ' .1f', annot=True, annot_kws={'size':10}, cmap='Blues' )

# Seperating the data into feature and target
x = diabetics_dataset.drop('Outcome', axis = 1)
y = diabetics_dataset['Outcome']

# splitting the data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify = y)

print(x.shape, x_train.shape, x_test.shape)

"""### `Traning the Model"""

model= svm.SVC(kernel='linear')

model.fit(x_train, y_train)

"""### Model Evaluation"""

x_train_prediction = model.predict(x_train)

x_train_accuracy = accuracy_score(x_train_prediction, y_train)
x_train_accuracy

x_test_prediction = model.predict(x_test)

x_test_accuracy = accuracy_score(x_test_prediction, y_test)
x_test_accuracy

"""### Building a predictive System"""

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to a numpy array
input_data_asarray = np.asarray(input_data)

# reshape the array as we are predicting for one instance
reshape_numpy_array = input_data_asarray.reshape(1,-1)

# making prediction
prediction = model.predict(reshape_numpy_array )
print(prediction)

if (prediction[0] == 0):
  print('Non diabetic')
else:
  print('Diabetic')

"""### Saving the trained model"""

import pickle

filename = 'trained_diabetic_model.sav'
pickle.dump(model, open(filename, 'wb'))

#loading the model
loaded_model = pickle.load(open('trained_diabetic_model.sav', 'rb'))

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to a numpy array
input_data_asarray = np.asarray(input_data)

# reshape the array as we are predicting for one instance
reshape_numpy_array = input_data_asarray.reshape(1,-1)

# making prediction
prediction = loaded_model.predict(reshape_numpy_array )
print(prediction)

if (prediction[0] == 0):
  print('Non diabetic')
else:
  print('Diabetic')

