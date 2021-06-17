#!/usr/bin/env python
# coding: utf-8

# # Shivesh Upadhyay

# # Data pre- Processsing 
#    Here we look at data and make it ready for model applying

# In[27]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[28]:


dataset = pd.read_csv('Iris.csv')


# In[29]:


dataset.head()


# ### Here we see the columns that are suitable for our model are 1,2,3,4

# In[30]:


X = dataset.iloc[:, [1,2,3,4]].values


# In[31]:


print(X)


# # Determining number of cluster by Elbow method

# In[32]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# ### So number of cluster will be lebow bend in graph that is 3

# # Training model and getting clusters

# In[33]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)


# # Visualising clusters
#    Here we visualise cluster by plotting various 2d plots taking 2 columns at a time

# In[26]:


for i in range(4):
    for j in range(i+1,4):
        if (i != j):
            plt.scatter(X[y_kmeans == 0, i], X[y_kmeans == 0, j], s = 100, c = 'red', label = 'Cluster 1')
            plt.scatter(X[y_kmeans == 1, i], X[y_kmeans == 1, j], s = 100, c = 'blue', label = 'Cluster 2')
            plt.scatter(X[y_kmeans == 2, i], X[y_kmeans == 2, j], s = 100, c = 'green', label = 'Cluster 3')
            plt.scatter(kmeans.cluster_centers_[:, i], kmeans.cluster_centers_[:, j], s = 300, c = 'yellow', label = 'Centroids')
            plt.title('Clusters')
            plt.xlabel(dataset.columns[i+1])
            plt.ylabel(dataset.columns[j+1])
            plt.legend()
            plt.show()


# In[ ]:




