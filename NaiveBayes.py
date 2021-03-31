
# coding: utf-8

# In[1]:


#import libraries 
import random # to randomize the training dataset
import math # self-explanatory
import pprint # to pretty print our initial non-panda dataset
import pandas as pd # to list and do vector manipulation


# In[2]:


df = pd.read_csv("DataTraining.csv")
print df

# In[3]:


df


# In[4]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[5]:


smt = le.fit_transform(df['Semester'])
print smt

# In[6]:


gender = le.fit_transform(df['Gender'])
print gender

# In[7]:


jp = le.fit_transform(df['JenisPresensi'])
print jp

# In[8]:


label = le.fit_transform(df['Minat'])
print label

# In[9]:


fs = zip(smt, gender, jp)
print fs

# In[10]:


from sklearn.naive_bayes import GaussianNB


# In[18]:


#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(fs, label)

#Predict Output
predicted = model.predict([[0,1,2],[0,1,1],[3,1,2]])
predicted
