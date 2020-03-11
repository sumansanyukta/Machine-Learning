#!/usr/bin/env python
# coding: utf-8

# In[23]:


#Import libraries panda, numpy, scikit
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer #Loading breast cancer data from the sklearn datasets
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()
#Data set classify cancer into teo categories Malignant=0 and Benign=1
breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
X = X[['mean area', 'mean compactness']] #Extracting feature--> mean area and mean compactness
y = pd.Categorical.from_codes(breast_cancer.target, breast_cancer.target_names)
y = pd.get_dummies(y, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')#Create instance of ofKNeighbourClassifier
knn.fit(X_train, y_train)#Fit the model for KNN
y_pred = knn.predict(X_test)#Test data from the test data set using our newly trained model


# In[25]:


sns.scatterplot(
    x='mean area',
    y='mean compactness',
    hue='benign',
    data=X_test.join(y_test, how='outer')
)


# In[26]:


plt.scatter(
    X_test['mean area'],
    X_test['mean compactness'],
    c=y_pred,
    cmap='coolwarm',
    alpha=0.7
)


# In[27]:


confusion_matrix(y_test, y_pred) #Compute confusion matrx to check the accuracy of the model


# In[36]:


accuracy=(121/143)*100 #Here 121 is the total number of data classified correctly
print (accuracy)

