#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#importing the dataset
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Successfully Data Imported")


# In[3]:


#showing the top of the dataset
data.head(15)


# In[4]:


# Plotting the distribution of scores
data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours Vs Score')
plt.xlabel('Study Hour')
plt.ylabel('Student Score')
plt.show()


# In[5]:


# Retriving the attributes & labels from the dataset using iloc

X = data.iloc[:, :-1].values
Y = data.iloc[:, 1].values


# In[6]:


# Using Scikit-Learn's built-in train_test_split() method

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state = 0)


# In[7]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

print("Training Completed")


# In[8]:


# Plotting the regression line
line = regressor.coef_*X + regressor.intercept_

plt.scatter(X,Y)
#plt.xlabel('Study Hour')
#plt.ylabel('Student Score')
plt.plot(X,line)
plt.show()


# In[9]:


print(X_test) # Testing Data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[10]:


# Actual Data Vs Predicted Data
df = pd.DataFrame({'Actual':Y_test, 'Predicted':y_pred})
df


# In[11]:


# Predicted score if a student studies for 9.25 hrs/day

hours = np.array(9.25)
hours = hours.reshape(-1, 1)
z_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(z_pred[0]))


# In[12]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))


# In[ ]:




