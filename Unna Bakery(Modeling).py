#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg


# In[2]:


df =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales_Cleaned.xlsx')


# In[3]:


df.drop(columns=df.columns[0], axis=1, inplace=True)


# In[4]:


df


# In[5]:


df_copy=df.copy()


# In[6]:


df_copy['Month']= df['Posting Date'].dt.month_name()
df_copy['Day'] = df['Posting Date'].dt.day_name()


# In[7]:


df_copy.columns


# In[8]:


df_copy.isnull().sum()


# In[9]:


df_copy.nunique()


# In[10]:


df_copy['Flavor'].unique()


# In[11]:


df_copy.describe()


# In[12]:


df_copy.info()


# In[13]:


df_copy.drop(columns=['Posting Date', 'Quantity', 'Group Name'], inplace=True)


# In[14]:


df_copy.rename(columns = {'Total Sales $':'Sales', 'Customer/Vendor Name':'Vendor','Warehouse Name':'Warehouse','Item/Service Description':'Size'}, inplace = True)


# In[15]:


df_copy


# In[16]:


df_copy['Size']=df_copy['Size'].str.split("(").str[1]


# In[17]:



df_copy['Size'] = [x[0:-1] for x in df_copy['Size']]


# In[18]:


df_copy['Size']


# ## Relationship Analysis

# In[19]:


corr= df_copy.corr()


# In[20]:


sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)


# In[21]:


sns.pairplot(df_copy)


# In[22]:


sns.relplot(x='State', y='Sales', hue='Size', data=df_copy)


# In[23]:


sns.relplot(x='State', y='Sales', hue='Flavor', data=df_copy)


# In[24]:


sns.relplot(x='City', y='Sales', hue='Flavor', data=df_copy)


# In[25]:


sns.distplot(df_copy['Sales'], bins=5)


# In[26]:


sns.catplot(x='Sales',kind='box', data=df_copy)


# In[27]:


df_copy['Sales'].describe()


# ## 3-Modelling

# ## 1- Detecting & Removing Outliers

# In[28]:


df_copy.describe()


# In[29]:


df_copy.columns


# In[30]:


df_copy.info()


# 
# fig, axs = plt.subplots(4,figsize=(6,18))
# x = df_copy[['Vendor','Sales','City','Size','State','Warehouse','Flavor','Month','Day']]
# for i,column in enumerate(x):
#     sns.boxplot(df_copy[column], ax=axs[i])

# In[31]:


f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(df_copy['Sales'])
plt.title('Total Sales Before Removing outliers')


# ## Using IQR (Inter Quartile Range)

# In[32]:


Q1 = np.percentile(df_copy['Sales'], 25,
                   interpolation = 'midpoint')
 
Q3 = np.percentile(df_copy['Sales'], 75,
                   interpolation = 'midpoint')
IQR = Q3 - Q1
 
print("Old Shape: ", df_copy.shape)
 
# Upper bound
upper = np.where(df_copy['Sales'] >= (Q3+1.5*IQR))
# Lower bound
lower = np.where(df_copy['Sales'] <= (Q1-1.5*IQR))
print("upper bound: ", upper)
print("lower bound: ", lower)

df_copy.drop(upper[0], inplace = True)
df_copy.drop(lower[0], inplace = True)
 
print("New Shape: ", df_copy.shape)


# In[33]:


f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(df_copy['Sales'])
plt.title('Total Sales After Removing outliers using IQR')


# In[34]:


df_copy.describe()


# plt.figure(figsize=(15,15))
# sns.scatterplot(x='Year', y='Total Sales $', hue='Flavor', data=df_copy, s=150)
# plt.xlabel('Year', fontsize=25)
# plt.ylabel('Sales $', fontsize=25)
# plt.xticks(np.arange(df_copy['Year'].min(),df_copy['Year'].max()+1, 1.0))
# plt.xticks(fontsize=20) 
# plt.yticks(fontsize=20)
# plt.legend(loc=2, prop={'size': 20})
# 

# In[35]:


plt.figure(figsize=(15,15))
sns.scatterplot(x='Month', y='Sales', hue='Flavor', data=df_copy, s=150)
plt.xlabel('Month', fontsize=25)
plt.ylabel('Sales $', fontsize=25)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)
plt.legend(loc=2, prop={'size': 20})


# In[36]:


plt.figure(figsize=(15,15))
sns.scatterplot(x='Day', y='Sales', hue='Flavor', data=df_copy, s=150)
plt.xlabel('Day', fontsize=25)
plt.ylabel('Sales $', fontsize=25)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)
plt.legend(loc=2, prop={'size': 20})


# In[37]:


plt.figure(figsize=(15,15))
sns.scatterplot(x='State', y='Sales', hue='Flavor', data=df_copy, s=150)
plt.xlabel('State', fontsize=25)
plt.ylabel('Sales $', fontsize=25)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)
plt.legend(prop={'size': 20})


# In[38]:


plt.figure(figsize=(20,15))
sns.scatterplot(x='Warehouse', y='Sales', hue='Flavor', data=df_copy, s=150)
plt.xlabel('Warehouse Name', fontsize=25)
plt.ylabel('Sales $', fontsize=25)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)
plt.legend(prop={'size': 20})


# In[39]:


plt.figure(figsize=(20,15))
sns.scatterplot(x='Flavor', y='Sales',data=df_copy, s=150)
plt.xlabel('Flavor', fontsize=25)
plt.ylabel('Sales $', fontsize=25)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)
plt.xticks(rotation=90)


# In[40]:


Warehouse_Sales=df_copy.groupby(df_copy['Warehouse']).agg({'Sales':'sum'}).reset_index()


# In[41]:


Vendor_Sales=df_copy.groupby(df_copy['Vendor']).agg({'Sales':'sum'}).reset_index()


# In[42]:


City_Sales=df_copy.groupby(df_copy['City']).agg({'Sales':'sum'}).reset_index()


# In[43]:


weekday_Sales=df_copy.groupby(df_copy['Day']).agg({'Sales':'sum'}).reset_index()


# In[44]:


Month_Sales=df_copy.groupby(df_copy['Month']).agg({'Sales':'sum'}).reset_index()


# In[71]:


State_Sales=df_copy.groupby(df_copy['State']).agg({'Sales':'sum'}).reset_index()


# In[72]:


Size_Sales=df_copy.groupby(df_copy['Size']).agg({'Sales':'sum'}).reset_index()


# In[46]:


plt.figure(figsize=(25,15))
sns.violinplot(x='State', y='Sales', hue='Flavor', data=df_copy, bw=14.5)
plt.xlabel('State', fontsize=25)
plt.ylabel('Sales', fontsize=25)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)


# In[47]:


plt.figure(figsize=(20,15))
sns.boxplot(x='Flavor', y='Sales', data=df_copy)
plt.xlabel('Flavor', fontsize=25)
plt.ylabel('Sales in $', fontsize=25)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)
plt.xticks(rotation=90)


# In[48]:


plt.figure(figsize=(20,15))
sns.boxplot(x='Size', y='Sales', data=df_copy)
plt.xlabel('Item', fontsize=25)
plt.ylabel('Sales in $', fontsize=25)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)
plt.xticks(rotation=90)


# In[49]:


plt.figure(figsize=(15,15))
sns.boxplot(x='State', y='Sales', data=df_copy)
plt.xlabel('State', fontsize=25)
plt.ylabel('Sales in $', fontsize=25)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)
plt.xticks(rotation=90)


# In[50]:


plt.figure(figsize=(10,20))
sns.boxplot(x='Warehouse', y='Sales', data=df_copy)
plt.xlabel('Warehouse', fontsize=25)
plt.ylabel('Sales in $', fontsize=25)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)
plt.xticks(rotation=90)


# In[51]:


df_copy['Size'].unique()


# In[52]:


plt.figure(figsize=(20,15))
sns.barplot(x=State_Sales['State'], y=State_Sales['Sales'])
plt.xlabel('State', fontsize=25)
plt.ylabel('Sales', fontsize=25)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)


# In[53]:


df_copy[df_copy['State']=='DE']


# In[54]:


plt.figure(figsize = (20,5))
sns.scatterplot(x=df_copy['Flavor'], y=df_copy['Sales'], s=100)
plt.xticks(rotation=90)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)
plt.xlabel('Flavor',fontsize=25)
plt.ylabel('Total Sales $', fontsize=25)


# In[63]:


plt.figure(figsize=(20,10))
sns.lineplot(data=weekday_Sales, x='Day', y='Sales', palette=['orange'], linewidth=2.5)
#sns.lineplot(data=Total_Sales, palette=['orange'], linewidth=2.5)
plt.title('Total Sales on weekday and weekends',fontsize=20)
plt.legend(['Total Sales $'])
plt.xlabel('Date', fontsize=20)
plt.ylabel('Total Sales $', fontsize=20)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20) 
plt.legend(loc=2,prop={'size': 20})


# In[65]:


plt.figure(figsize=(20,10))
sns.lineplot(data=Month_Sales, x='Month', y='Sales', palette=['orange'], linewidth=2.5)
#sns.lineplot(data=Total_Sales, palette=['orange'], linewidth=2.5)
plt.title('Total Sales in Months',fontsize=20)
plt.legend(['Total Sales $'])
plt.xlabel('Date', fontsize=20)
plt.ylabel('Total Sales $', fontsize=20)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20) 
plt.legend(loc=2,prop={'size': 20})


# In[56]:


df_copy['Vendor'].unique()


# In[67]:


plt.figure(figsize=(15,15))
sns.scatterplot(x='City', y='Sales', hue='Flavor', data=df_copy, s=150)
plt.xlabel('City', fontsize=25)
plt.ylabel('Sales $', fontsize=25)
plt.xticks(fontsize=15) 
plt.yticks(fontsize=20)
plt.legend(loc=2, prop={'size': 20})
plt.xticks(rotation=90)


# In[58]:


City_Sales=df_copy.groupby(['City']).agg({'Sales':'sum'}).reset_index()


# In[69]:


plt.figure(figsize=(15,15))
sns.scatterplot(x='Size', y='Sales', hue='Flavor', data=df_copy, s=150)
plt.xlabel('Size', fontsize=25)
plt.ylabel('Sales $', fontsize=25)
plt.xticks(fontsize=15) 
plt.yticks(fontsize=20)
plt.legend(loc=2, prop={'size': 20})
plt.xticks(rotation=90)


# In[59]:


plt.figure(figsize=(20,15))
sns.barplot(x=City_Sales['City'], y=City_Sales['Sales'])
plt.xlabel('City', fontsize=25)
plt.ylabel('Sales', fontsize=25)
plt.xticks(fontsize=14) 
plt.yticks(fontsize=20)
plt.xticks(rotation=90)


# In[73]:


plt.figure(figsize=(20,15))
sns.barplot(x=Size_Sales['Size'], y=Size_Sales['Sales'])
plt.xlabel('State', fontsize=25)
plt.ylabel('Sales', fontsize=25)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)


# In[74]:


City_Sales.describe()


# In[75]:


f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(City_Sales['Sales'])


# In[76]:


df_copy.nunique()


# In[ ]:


df_copy.shape


# ## Categorical Data Transformstion:

# In[ ]:


#from sklearn.preprocessing import OrdinalEncoder


# In[107]:


df_copy


# In[108]:


df_copy.describe()


# In[109]:


df_copy.nunique()


# In[110]:


y=df_copy.iloc[:,2]


# In[111]:


y


# In[112]:


df_copy.info()


# In[113]:


df_copy


# In[114]:


X=df_copy.iloc[:,[0,1,3,4,5,6,7,8]]


# In[115]:


X
##


# In[116]:


X.nunique()


# In[117]:


X


# In[118]:


encoded_x =None
for i in [1,3,4,5,6,7]:
    label_encoder = LabelEncoder()
    feature = label_encoder.fit_transform(X.iloc[:,i])
    feature = feature.reshape(X.shape[0], 1)
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    feature = onehot_encoder.fit_transform(feature)
    if encoded_x is None:
        encoded_x = feature
    else:
        encoded_x = np.concatenate((encoded_x, feature), axis=1)

encoded_feature1=label_encoder.fit_transform(X.iloc[:,0])
encoded_feature2=label_encoder.fit_transform(X.iloc[:,2])
encoded_feature1=encoded_feature1.reshape(X.shape[0],1)
encoded_feature2=encoded_feature2.reshape(X.shape[0],1)
encoded_x=np.concatenate((encoded_feature1,encoded_feature2,encoded_x), axis=1)


# In[119]:


encoded_x


# ## Split Data Into Train and Test

# In[120]:


X_train, X_test, y_train, y_test = train_test_split (encoded_x, y, test_size=0.3, random_state=123)


# ## Fit the Model to the training Set

# In[121]:


from xgboost import XGBRegressor
from xgboost import cv
from xgboost import DMatrix
data_dmatrix = DMatrix(data=encoded_x,label=y)

params = {"objective":"reg:squarederror",'colsample_bytree': 0.25,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}
cv_results = cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)


# In[122]:


cv_results.head()


# In[123]:


print((cv_results["test-rmse-mean"]).tail(1))


# In[124]:


from xgboost import XGBRegressor
params = {'n_estimators':1000,
          'max_depth':10,
          'min_samples_split': 1200,
          'learning_rate': 0.01,
         'min_samples_leaf':60,
         'random_state':10,
         'subsample':0.5,
         }
xg_reg = XGBRegressor(**params)
#xg_reg = XGBRegressor()
xg_reg.fit(X_train, y_train)
print (xg_reg)
from sklearn.metrics import mean_squared_error
y_pred=xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: %f" % (rmse))
from sklearn.metrics import mean_absolute_error
MAE= np.sqrt(mean_absolute_error(y_test, y_pred))
print('MAE: %f' % (MAE))
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))



 #plot for residual erro#
 
## setting plot style
plt.style.use('fivethirtyeight')
 
## plotting residual errors in training data
plt.scatter(xg_reg.predict(X_train), xg_reg.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')
 
## plotting residual errors in test data
plt.scatter(xg_reg.predict(X_test), xg_reg.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
 
## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
 ## plotting legend
plt.legend(loc = 'upper right')
 
## plot title
plt.title("Residual errors")
 
## method call for showing the plot
plt.show()


# In[136]:



plt.plot(y_test, y_pred,'ro')
plt.plot(y_test, y_test,'b-')
plt.show()




plt.scatter(xg_reg.predict(X_train), xg_reg.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')
 
## plotting residual errors in test data
plt.scatter(xg_reg.predict(X_test), xg_reg.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
 
    
## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
 ## plotting legend
plt.legend(loc = 'upper right')
 
## plot title
plt.title("Residual errors")
## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)


# ## Two-Way Anova

# In[129]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
from sklearn.metrics import accuracy_score
warnings.filterwarnings( "ignore" )

formula = 'Sales ~ C(Month) + C(Day) + C(Month):C(Day)'
model = ols(formula, df_copy).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print(aov_table)


# In[130]:


df_copy


# In[131]:


1.041595e-02 < 0.05


# In[132]:


1.046303e-22 < 0.05


# In[133]:


2.941737e-16 < 0.05


# The p value obtained from ANOVA analysis for Sales, months, days and interaction are statistically significant (p<0.05). We conclude that the months significantly affects the sales outcome, days significantly affects the sales outcome, and interaction of both month and days significantly affect the sales outcome.

# ## B- Linear Regression Model

# In[140]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics

print(X)
# create linear regression object
reg = linear_model.LinearRegression()
 
# train the model using the training sets
reg.fit(X_train, y_train)
y_pred2= reg.predict(X_test)
 
# regression coefficients
print('Coefficients: ', reg.coef_)
 
# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))
print('Mean Squared Error: {}'.format(np.square(np.subtract(y_test,y_pred2)).mean()))
 
# plot for residual error
 
## setting plot style
plt.style.use('fivethirtyeight')
 
## plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')
 
## plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
 
## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
 
## plotting legend
plt.legend(loc = 'upper right')
 
## plot title
plt.title("Residual errors")
 
## method call for showing the plot
plt.show()

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred2))

print('Accuracy:',reg.score(X_train, y_train)*100)
#sns.scatterplot(y_pred2, y_test)


plt.plot(y_test, y_pred2,'ro')
plt.plot(y_test, y_test,'b-')
plt.show()


residual_train = y_train - (reg.predict(X_train))
fig, ax = plt.subplots(figsize=(15,5))
plt.xlabel('Error in Predicted Value', fontsize = 18)
plt.ylabel('Predicted Value', fontsize = 18)
plt.title("Residual plot (for Train Data)", fontsize = 20)
_ = ax.scatter(residual_train,reg.predict(X_train))



residual_test = y_test - y_pred2
fig, ax = plt.subplots(figsize=(15,5))
plt.xlabel('Error in Predicted Value', fontsize = 18)
plt.ylabel('Predicted Value', fontsize = 18)
plt.title("Comparison of Predicted Values VS Error (for Test Data)", fontsize = 20)
_ = ax.scatter(residual_test, y_pred2)


# In[ ]:




