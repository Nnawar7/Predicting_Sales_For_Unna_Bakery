#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


Feb_18 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='Feb 2018')
March_18 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='March 2018')
April_18 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='April 2018')
May_2018 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='May 2018')
June_2018 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='June 2018')
July_2018 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='July 2018')
Aug_2018 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='August 2018')
Sep_2018 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='September 2018')
Oct_2018 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='October 2018')


# In[4]:


df3=pd.concat([Feb_18, March_18,April_18, May_2018,June_2018,July_2018,Aug_2018, Sep_2018, Oct_2018], ignore_index=True)


# In[5]:


df3.head(10)


# In[6]:


df3=df3.rename(columns=({'Ship Date': 'Posting Date', 'Customer Name' :'Customer/Vendor Name','itemcode':'Item No.','ItemName':'Item/Service Description'}))


# In[7]:


df3.info()


# In[8]:


df3 = df3.drop(columns=['Customer code', 'Brand', 'UPC'])


# In[9]:


df3


# In[10]:


df3.dropna(subset=['Item No.'])


# In[11]:


df3.info()


# In[12]:


df3['Item No.'].isna().sum()


# In[13]:


df3=df3.dropna(subset=['Item No.'])


# In[14]:


df3['Item No.'].isna().sum()


# In[15]:


df3.info()


# In[16]:


df3.loc[:,'Item No.']=df3.loc[:,'Item No.'].map(lambda x: int(x))


# In[17]:


def unit_price(row):
    L1=188401, 188302, 188303, 188102, 188202, 188201, 188301, 188105, 188101, 188104, 188107, 188106, 188204, 188503, 188501, 188502, 188504
    L2= [52.0, 20.66, 20.66, 17.13, 13.6, 13.6, 20.66, 20.85, 20.85, 17.13, 20.85, 20.15, 13.6, 18.0, 20.1, 20.1, 20.1]
    d = {k:v for k,v in zip(L1,L2)}
    row['Price Per Unit']=row['Item No.'].map(lambda x : d.get(x))
    return row
    


# In[18]:


df3=unit_price(df3)


# In[19]:


df3


# In[20]:


df3['Total Sales $']=df3['Price Per Unit'] * df3['Quantity']


# In[21]:


df3


# In[22]:


df3.info()


# In[23]:


df3=df3.dropna()


# In[24]:


df3


# In[25]:


df3=df3.drop(columns='Price Per Unit')


# In[26]:


df3


# In[27]:


df3=df3.drop(columns=['Address On File', 'Item No.'])


# In[28]:


df3


# In[28]:


df3.to_excel(r'F:\Projects\Unna Bakery\Sales_Feb18-Oct18.xlsx', encoding='utf8') 


# In[29]:


df3.info()


# In[ ]:




