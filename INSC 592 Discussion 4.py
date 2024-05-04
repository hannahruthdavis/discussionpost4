#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd 
import matplotlib.pyplot as plt


# In[8]:


# Load the Iris Dataset
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])


# In[9]:


# Scatter plot of the Iris dataset
_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)


# In[10]:


iris = load_iris()
import numpy as np
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns= iris['feature_names'] + ['target']).astype({'target': int}) \
       .assign(species=lambda x: x['target'].map(dict(enumerate(iris['target_names']))))
cols = ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']

df.describe()


# In[11]:


# Descriptive analysis for 'petal length (cm)' and 'petal width (cm)'
petal_df = df[['petal length (cm)', 'petal width (cm)']]
petal_desc_stats = petal_df.describe()

# Print descriptive statistics
print("Descriptive Statistics for Petal Length and Petal Width:")
print(petal_desc_stats)


# In[12]:


# Histograms for Petal Length and Petal Width by Species

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))

df_setosa = df[df['species'] == 'setosa']
df_versicolor = df[df['species'] == 'versicolor']
df_virginica = df[df['species'] == 'virginica']

plt.subplot(2, 3, 1)
plt.hist(df_setosa['petal length (cm)'], bins=20, color='lightblue', edgecolor='black')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.title('Histogram of Petal Length for Setosa')

plt.subplot(2, 3, 2)
plt.hist(df_versicolor['petal length (cm)'], bins=20, color='lightgreen', edgecolor='black')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.title('Histogram of Petal Length for Versicolor')

plt.subplot(2, 3, 3)
plt.hist(df_virginica['petal length (cm)'], bins=20, color='lightpink', edgecolor='black')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.title('Histogram of Petal Length for Virginica')

plt.subplot(2, 3, 4)
plt.hist(df_setosa['petal width (cm)'], bins=20, color='lightyellow', edgecolor='black')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Frequency')
plt.title('Histogram of Petal Width for Setosa')

plt.subplot(2, 3, 5)
plt.hist(df_versicolor['petal width (cm)'], bins=20, color='purple', edgecolor='black')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Frequency')
plt.title('Histogram of Petal Width for Versicolor')

plt.subplot(2, 3, 6)
plt.hist(df_virginica['petal width (cm)'], bins=20, color='orange', edgecolor='black')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Frequency')
plt.title('Histogram of Petal Width for Virginica')

plt.tight_layout()
plt.show()


# In[ ]:




