#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Importing necessary liabraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# In[2]: Read the file by using pandas


df1 = pd.read_csv('24824_33185_compressed_housing.csv.zip')


# In[3]: copy the data and save it in df variable


df = df1.copy(deep = True)


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]: Plotting the bar plot for all the columns in data 


fig = plt.figure(figsize=(25, 15))
cols = 5
rows = np.ceil(float(df.shape[1]) / cols)

for i, column in enumerate(df.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if df.dtypes[column] == np.object:
        df[column].value_counts().plot(kind="bar", axes=ax)
    else:
        df[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.5, wspace=0.2)


# In[7]: Plot heatmap for Correlation 


plt.figure(figsize=(11,9))
sns.heatmap(df.corr(), annot=True, fmt='.1g',cmap='viridis')


# In[8]: Filling the Na values by the median of total_bedrooms


df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())


# In[9]: Check the null values after filling.


df.isnull().sum()


# In[10]: Check the values of ocean_proximity column 


df['ocean_proximity'].value_counts()


# In[ ]:

# In[11]:


df.columns


# In[ ]:





# In[12]:Plotting scatter of median_income and median_house_value


df.plot.scatter(x='median_income',y='median_house_value',alpha=0.1)


# In[13]:


df.plot.scatter(x='housing_median_age',y='population')


# In[14]:


df = df[df['population']<20000]


# In[15]: Creating some features 


df['rooms_per_bedrooms'] = df['total_rooms'] / df['total_bedrooms']
df['rooms_per_households'] = df['total_rooms'] / df['households']
df['households_per_pop'] = df['households'] / df['population']


# In[16]:


df.corr()['median_house_value'].sort_values(ascending=False)


# In[17]:


c = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
        'ocean_proximity', 'rooms_per_bedrooms',
       'households_per_pop','rooms_per_households','median_house_value']
df = df.reindex(columns=c,)
df.head()


# In[ ]:


# In[18]:


df = pd.get_dummies(df, columns=['ocean_proximity'])


# In[19]:


df.columns


# In[20]:


scaler = MinMaxScaler()

col = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'rooms_per_bedrooms', 'households_per_pop', 'rooms_per_households',
        'ocean_proximity_<1H OCEAN',
       'ocean_proximity_INLAND', 'ocean_proximity_ISLAND',
       'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN','median_house_value']

values = df[col].values

scale_df = scaler.fit_transform(values)


# In[21]:


df_t = pd.DataFrame(scale_df, columns=col,index = df.index)


# In[22]:


df_t.head()#.describe()


# In[23]:


data = df_t.copy()


# In[24]:


X = data.drop(['median_house_value'],axis=1)
y = data['median_house_value']


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


# In[27]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[28]:


lin_reg = LinearRegression()


# In[29]:


lin_reg.fit(X_train, y_train)


# In[30]:


lin_pred = lin_reg.predict(X_test)


# In[31]:


mse = mean_squared_error(lin_pred, y_test)
rmse = np.sqrt(mse)
print('RMSE :',rmse)


# In[32]:


score = lin_reg.score(X_train, y_train)
print('score:',score*100)


# In[33]:


pd.DataFrame(zip(X.columns,lin_reg.coef_))


# In[34]:


pd.DataFrame(zip(y_test,lin_pred),columns=['True','Predicted'])


# In[35]:


data2 = data.copy()


# In[36]:


data2.columns


# In[41]:


from sklearn.ensemble import RandomForestRegressor


# In[64]:


forest_reg = RandomForestRegressor(max_depth=None,random_state=42)
forest_reg.fit(X_train, y_train)
forest_pred = forest_reg.predict(X_test)

mse = mean_squared_error(forest_pred, y_test)
rmse = np.sqrt(mse)
print('RMSE:',rmse)


# In[65]:

score = forest_reg.score(X_train, y_train)
print('score:',score*100)

Result = pd.DataFrame(zip(y_test, forest_pred), columns=['Actual', 'Predicted'],)
print(Result)
