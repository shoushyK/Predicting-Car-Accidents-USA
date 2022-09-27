#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot  as plt


# In[35]:


from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

#import hdbscan
import folium
import re
import seaborn as sns


# In[3]:


df=pd.read_csv('/Users/macbookpro/Downloads/US_Accidents_Dec21_updated.csv')


# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


df.tail()


# In[7]:


df.info()


# In[8]:


df.describe().T


# In[9]:


df.isnull().sum()


# In[10]:


percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'percent_missing': percent_missing})


# In[11]:


missing_value_df.sort_values('percent_missing', ascending=False,inplace=True) #inplace = True REMOVES empty data


# In[12]:


missing_value_df[missing_value_df['percent_missing'] > 0]


# In[13]:


df.isnull()


# In[14]:


remove = 'Number'
df.drop(remove, inplace =True, axis =1)


# In[15]:


df.isnull().sum()


# In[16]:


percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
missing_value_df.sort_values('percent_missing', ascending=False,inplace=True) #inplace = True REMOVES empty data
missing_value_df[missing_value_df['percent_missing'] > 0]


# In[17]:


df.duplicated()


# In[18]:


df.select_dtypes(include= 'bool').columns.tolist() 


# In[19]:


df.describe()

Conclusions from df.describe:
1. Visibility=9 miles is the average, which is excellent visibility -> eliminate visibility as a factor
2. In 75% percentile, there was no reported precipitation -> eliminate rain as a factor
3. Distance (Lenght of road extent affected) is on average = 0.7 miles
# In[20]:


df['City'].nunique()


# In[21]:


State = df.State.value_counts().reset_index()
State.columns = ['State','Accidents']
State.head()


# In[22]:


State['Percentage'] = round(State['Accidents'] * 100 / State['Accidents'].sum() , 2)
State.head()


# In[30]:


plt.figure(figsize=(18,9))
graph = plt.bar(State.State.head(10),State.Percentage.head(10),)
plt.title('Percentage of accidents occured across the top 10 States',ha='center',weight='bold')
plt.xlabel("State",ha='center',weight='bold')
plt.ylabel("Percentage of accidents",ha='center',weight='bold')
 
plt.show()


# In[24]:


Acc_Severity = df.Severity.value_counts().reset_index()
Acc_Severity.columns = ['Severity','Accidents']
Acc_Severity['Percentage'] = round(Acc_Severity['Accidents'] * 100 /Acc_Severity['Accidents'].sum() , 2)
Acc_Severity.head()

There is further analysis with date & time as well as other columns
# In[25]:


X=np.array(df[['Start_Lat','Start_Lng']],dtype='float64')
plt.scatter(X[:,0],X[:,1],alpha=0.2,s=5)
plt.grid(True)


# In[26]:


#test_df = df.loc[(df['Roundabout'] == True) | (df['Station'] == True)]
#test_df.info()


# In[31]:


sample_df = df.sample(int(0.1 * len(df)))


# In[37]:


sns.scatterplot(x=sample_df.Start_Lng, y=sample_df.Start_Lat, size=0.0001)


# In[ ]:




