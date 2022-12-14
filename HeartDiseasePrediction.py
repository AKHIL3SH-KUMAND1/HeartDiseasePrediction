#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


# In[8]:


# loading the data to a pandas data frame
heart_data = pd.read_csv('heart_disease_data.csv')


# In[9]:


# print first 5 rows of data
heart_data.head()
heart_data.tail()


# In[10]:



# no.of rows and columns
heart_data.shape


# In[11]:



#getting info on dataset
heart_data.info()


# In[12]:


# checking for missing values
heart_data.isnull().sum()


# In[13]:


#statistical data about the data
heart_data.describe()


# In[14]:


#checking the distribution of target variable
heart_data['target'].value_counts()
     


# In[15]:


X = heart_data.drop(columns='target',axis=1)
y=heart_data['target']


# In[16]:


X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.1,stratify=y,random_state=3)


# In[17]:


print(X_train.shape,X_test.shape)


# In[18]:


model=LogisticRegression()


# In[19]:


#training the model
model.fit(X_train,y_train)


# In[20]:


# testing data
accuracy_score(model.predict(X_test),y_test)


# In[21]:


# training data
accuracy_score(model.predict(X_train),y_train)


# In[24]:


inp=[]
print("Enter the following details ")
inp.append(int(input("age")))
inp.append(int(input("sex")))
inp.append(int(input("chest pain type (4 values)")))
inp.append(int(input("resting blood pressure")))
inp.append(int(input("serum cholestoral in mg/dl")))
inp.append(int(input("fasting blood sugar > 120 mg/dl")))
inp.append(int(input("resting electrocardiographic results (values 0,1,2)")))
inp.append(int(input("maximum heart rate achieved")))
inp.append(int(input("exercise induced angina")))
inp.append(float(input("oldpeak = ST depression induced by exercise relative to rest")))
inp.append(int(input("the slope of the peak exercise ST segment")))
inp.append(int(input("number of major vessels (0-3) colored by flourosopy")))
inp.append(int(input("thal: 0 = normal; 1 = fixed defect; 2 = reversable defect")))
input_np = np.asarray(inp)
input_r=input_np.reshape(1,-1)
print(model.predict(input_r)[0])


# In[4]:





# In[24]:


input_np = np.asarray(input())


# In[ ]:




