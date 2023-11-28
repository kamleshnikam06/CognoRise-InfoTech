#!/usr/bin/env python
# coding: utf-8

# # Iris Flower Classification

# ## Import the required packages

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization library
import matplotlib.pyplot as plt # for creating static, animated, and interactive visualizations


# ## Load the Dataset

# In[2]:


df = pd.read_csv("D:\CognoRise InfoTech\IRIS.csv")


# In[3]:


iris = df.copy()
iris.head()


# In[4]:


iris.info()
iris.species.value_counts()


# In[5]:


iris.describe().T


# ## **Visualization**

# In[6]:


plt.figure(figsize=(8,6));
sns.pairplot(iris,kind='reg',hue ='species',palette="husl" );


# In[7]:


plt.figure(figsize=(8,6));
sns.scatterplot(x=iris.sepal_length,y=iris.sepal_width,hue=iris.species).set_title("Sepal length and Sepal width distribution of three flowers");


# In[8]:


plt.figure(figsize=(8,6));
cmap = sns.cubehelix_palette(dark=.5, light=.9, as_cmap=True)
ax = sns.scatterplot(x="petal_length", y="petal_width",hue="species",size="species",sizes=(20,200),legend="full",data=iris);


# ## Creating ML classify Models

# In[9]:


#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
iris['species'] = lb_make.fit_transform(iris['species'])
iris.sample(3)


# In[10]:


# # PCA ===> if data consist of too many parameters/variables(columns) then we need to use PCA; in this data it is not necessary
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 2,whiten = True) #whitten = normalize
# pca.fit(iris)
# x_pca = pca.transform(iris)


# In[11]:


# Importing metrics for evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[12]:


y = iris.species
X = iris.drop('species',axis = 1)


# In[13]:


#Train and Test split,cross_val,k-fold
from sklearn.model_selection import KFold,train_test_split,cross_val_score
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# ## KNN Model

# In[14]:


from sklearn.neighbors import KNeighborsClassifier


# In[15]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)


# In[16]:


y_pred = knn.predict(X_test)


# ## Summary of the predictions

# In[17]:


# Summary of the predictions made by the KNN
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# ## Accuracy score

# In[18]:


from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[ ]:




