import numpy as np
import pandas as pd                                        #for importing dataset
import sklearn.model_selection as ms                       #for splitting the dataset into the Training set and Testing set
from sklearn.metrics import confusion_matrix as cm         #for calculating accuracy

from sklearn.preprocessing import Imputer as imp                                       # for filling missing values in dataset
from sklearn.preprocessing import LabelEncoder as l_enc, OneHotEncoder as oh_enc       #for encoding categorical data and independent variables
from sklearn.preprocessing import StandardScaler as ss                                 #for feature scaling of data

# Importing the dataset
data = pd.read_csv('images.csv')
m, fnum = data.shape                #no. of training examples and features
X = data.iloc[:, :-1].values        #Feature matrix
y = data.iloc[:, fnum - 1].values   #Target variable vector


#Will use Imputer only if columns have missing values
"""
miss = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
miss.fit(X[])
X[] = miss.transform(X[])
"""

#Will use encoding only if needed
"""
# Encoding categorical data
# Encoding the Independent Variable
l_enc_X = l_enc()
X[] = l_enc_X.fit_transform(X[])
oh_enc_X = oh_enc(categorical_features = [0])
X = oh_enc_X.fit_transform(X).toarray()

# Encoding the Target Variable, if categorical
l_enc_y = l_enc()
y = l_enc_y.fit_transform(y)
"""


# Splitting the dataset into the Training set and Testing set
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# We will create our Deep Learning NN model using keras here

# Predicting the Test set results
y_pred = model.predict(X_test)

# Making the Confusion Matrix

mat = cm(y_test, y_pred)
