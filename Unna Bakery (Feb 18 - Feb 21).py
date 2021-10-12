#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Exploring The Data Set

# In[2]:


Apr_20 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='April 2020')
May_20 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='May 2020')
Jun_20 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='June 2020')
Jul_20 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='July 2020')
Aug_20 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='Aug 2020')
Sept_20 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='Sept 2020')
Oct_20 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='Oct 2020')
Nov_20 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='Nov 2020')
Dec_20 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='Dec 2020')
Jan_21 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='Jan 2021')
Feb_21 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='Feb 2021')


# In[3]:


df2=pd.concat([Apr_20, May_20, Jun_20, Jul_20,Aug_20, Sept_20, Oct_20, Nov_20, Dec_20, Jan_21, Feb_21], ignore_index=True)


# In[4]:


df2.head(10)


# In[5]:


df2.tail()


# In[6]:


df2=df2.rename(columns=({'Ship-to Street': 'Street', 'Ship-to City' :'City','Ship-to State':'State','Ship-to Zip Code':'Zip Code'}))


# In[7]:


df2.info()


# In[10]:


Nov_18 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='November 2018')
Dec_18 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='December 2018')
Jan_19 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='January 2019')
Feb_19 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='February 2019')
Mar_19 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='March 2019')
Apr_19 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='April 2019')
May_19 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='May 2019')
Jun_19 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='June 2019')
Jul_19=pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='July 2019')
Aug_19=pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='August 2019')
Sep_19=pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='Sept 2019')
Oct_19=pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='Oct 2019')
Nov_19=pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='Nov 2019')
Dec_19=pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='Dec 2019')
Jan_20=pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='Jan 2020')
Feb_20=pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='Feb 2020')
Mar_20=pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales.xlsx', sheet_name='March 2020')


# In[11]:


df1=pd.concat([Nov_18,Dec_18,Jan_19,Feb_19,Mar_19,Apr_19,May_19,Jun_19,Jul_19,Aug_19,Sep_19,Oct_19,Nov_19,Dec_19,Dec_19,Jan_20,Feb_20,Mar_20], ignore_index=True)


# In[12]:


df1.head(10)


# In[13]:


df1.tail(10)


# In[14]:


df=pd.concat([df1,df2],ignore_index=True)


# In[13]:


df.info()


# In[15]:


df.shape


# In[16]:


df.info()


# In[17]:


df[df['Street'].isnull()]


# In[18]:


df=df.dropna(subset=['Document Number'])


# In[20]:


df.describe()


# In[21]:


df.head()


# In[22]:


df.loc[:,'Document Number']=df.loc[:,'Document Number'].map(lambda x: int(x))


# In[23]:


df.loc[:,'Item No.']=df.loc[:,'Item No.'].map(lambda x: int(x))


# In[24]:


df.info()


# In[25]:


len(df.index)


# In[26]:


df['Item No.'].unique()


# In[27]:


grouped_df= df.loc[:, ['Item No.','Quantity','Total Sales $']]


# In[28]:


grouped_df = grouped_df[grouped_df.Quantity==1] #Selecting unit price


# In[29]:


grouped_df=grouped_df.drop_duplicates(subset=['Item No.'])


# In[30]:


x= zip(grouped_df['Item No.'], grouped_df['Total Sales $'])
 


# In[31]:


x=dict(x)


# In[32]:


x


# In[33]:


df['Credit Hold'].unique()


# In[34]:


df['Credit Hold'].replace('n', 'N', inplace=True)


# ## Dropping unwanted Features and concatinating the last dataframe

# In[35]:


df['Group Name.1']


# In[36]:


df = df.drop(columns=['Group Name.1'])


# In[37]:


#Vendor Code, Ship-to-Street,Ship-to Zip Code,  Item_No., Sales Employee Name, Group Name(Snacks/Speciality)

df = df.drop(columns=['Credit Hold','Customer/Vendor Code', 'Street', 'Zip Code','Sales Employee Name','Document Number','Item No.'])


# In[38]:


df.info()


# In[40]:


Feb18 =pd.read_excel(r'F:\ML Projects\Unna Bakery\Sales_Feb18-Oct18.xlsx')


# In[41]:


Feb18.drop(columns=Feb18.columns[0], inplace=True)


# In[42]:


Feb18.info()


# In[43]:


df=pd.concat([Feb18,df])


# In[44]:


df.info()


# ## Mode Imputation

# In[45]:


df['City'] = df['City'].fillna(df['City'].mode()[0])


# In[46]:


df['State'] = df['State'].fillna(df['State'].mode()[0])


# In[47]:


df['Warehouse Name'] = df['Warehouse Name'].fillna(df['Warehouse Name'].mode()[0])


# In[48]:


df['Group Name'] = df['Group Name'].fillna(df['Group Name'].mode()[0])


# In[49]:


df.info()


# ## Adding a new Feature "Flavor"

# In[50]:


df['Flavor'] = df['Item/Service Description']


# In[51]:


df['Flavor'].value_counts()


# In[52]:


df.loc[df['Flavor'].str.contains('Lemon Lime'), 'Flavor'] = "Lemon Lime"
df.loc[df['Flavor'].str.contains('Coconut'), 'Flavor'] = "Coconut"
df.loc[df['Flavor'].str.contains('Brown'), 'Flavor'] = "Brown Butter"
df.loc[df['Flavor'].str.contains('Raspberry'), 'Flavor'] = "Raspberry"
df.loc[df['Flavor'].str.contains('Cardamom'), 'Flavor'] = "Cardamom"
df.loc[df['Flavor'].str.contains('Farmer'), 'Flavor'] = "Butter & Almonds"
df.loc[df['Flavor'].str.contains('Ginger Snap & Vanilla Dream'), 'Flavor'] = "Ginger & Vanilla"
df.loc[df['Flavor'].str.contains('Vanilla Dream'), 'Flavor'] = "Vanilla"
df.loc[df['Flavor'].str.contains('Vanilla Cookie'), 'Flavor'] = "Vanilla"
df.loc[df['Flavor'].str.contains('Ginger Snap'), 'Flavor'] = "Ginger"


# In[53]:


df['Flavor'].value_counts()


# In[54]:


df.Flavor.value_counts()


# In[55]:


df


# In[53]:


df.to_excel(r'F:\Projects\Unna Bakery\Sales_Cleaned.xlsx', encoding='utf8') 


# ## Data Visualization

# ## 1- How Did the Total Sales grow over years?

# In[56]:


Total_Sales=df.groupby(['Posting Date']).agg({'Total Sales $':'sum'})


# In[57]:


Avg_Sales=df.groupby(['Posting Date']).agg({'Total Sales $':'mean'})


# In[58]:


Avg_Sales.sort_values(by='Total Sales $', ascending=False)


# In[60]:


Total_Sales.sort_values(by='Total Sales $', ascending=False)


# In[61]:


Total_Sales.head(10)


# In[62]:


Total_Sales=Total_Sales.reset_index()


# In[63]:


Total_Sales


# In[64]:


plt.figure(figsize=(20,10))
sns.lineplot(data=Total_Sales, x='Posting Date', y='Total Sales $', palette=['orange'], linewidth=2.5)
#sns.lineplot(data=Total_Sales, palette=['orange'], linewidth=2.5)
plt.title('Total Sales From February 2018 till February 2021',fontsize=20)
plt.legend(['Total Sales $'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Total Sales $', fontsize=18)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20) 


# In[65]:


df['Total Sales $'].plot(figsize=(10,6))

df['Total Sales $'].rolling(window=20).mean().plot()


# In[66]:


df.hist()


# In[69]:


df_copy=df.copy()

df_copy['Year']= df['Posting Date'].apply(lambda x: (x.year))
df_copy['Month']= df['Posting Date'].apply(lambda x: (x.month))
df_copy['Day']= df['Posting Date'].apply(lambda x: (x.day))


# In[70]:


df_copy


# In[71]:


plt.figure(figsize=(20,15))
sns.boxplot(x='Year', y='Total Sales $', data=df_copy)
plt.xlabel('Year', fontsize=25)
plt.ylabel('Sales in dollars', fontsize=25)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)


# In[72]:


plt.figure(figsize=(20,15))
sns.boxplot(x='Month', y='Total Sales $', data=df_copy)
plt.xlabel('Month', fontsize=25)
plt.ylabel('Sales', fontsize=25)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)


# In[73]:


df[df['Total Sales $'] > 300]


# In[74]:


plt.figure(figsize=(20,15))
sns.boxplot(x='Day', y='Total Sales $', data=df_copy)
plt.xlabel('Day', fontsize=25)
plt.ylabel('Sales', fontsize=25)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)


# In[75]:


Year_Sales=df.groupby(df['Posting Date'].dt.strftime('%y')).agg({'Total Sales $':'sum'}).reset_index()


# In[72]:


#df['Posting Date'].apply(lambda x: (x.year))


# In[76]:


plt.figure(figsize=(20,15))
sns.barplot(x=Year_Sales['Posting Date'], y=Year_Sales['Total Sales $'])
plt.xlabel('Year', fontsize=25)
plt.ylabel('Sales', fontsize=25)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)


# plt.figure(figsize=(20,15))
# sns.swarmplot(x='Year', y='Total Sales $', data=df_copy)
# plt.xlabel('Year', fontsize=25)
# plt.ylabel('Sales', fontsize=25)
# plt.xticks(fontsize=20) 
# plt.yticks(fontsize=20)

# plt.figure(figsize=(20,15))
# sns.swarmplot(x='Month', y='Total Sales $', data=df_copy)
# plt.xlabel('Month', fontsize=25)
# plt.ylabel('Sales', fontsize=25)
# plt.xticks(fontsize=20) 
# plt.yticks(fontsize=20)

# In[77]:


plt.pie(Year_Sales['Total Sales $'], labels=Year_Sales['Posting Date'],autopct='%1.1f%%', textprops={'fontsize': 14})


# In[78]:


Month_Sales=df.groupby(df['Posting Date'].dt.strftime('%m')).agg({'Total Sales $':'sum'}).reset_index()


# In[79]:


Month_Sales


# In[80]:


Day_Sales=df.groupby(df['Posting Date'].dt.strftime('%d')).agg({'Total Sales $':'sum'}).reset_index()


# In[81]:


plt.figure(figsize=(20,15))
sns.barplot(x=Month_Sales['Posting Date'], y=Month_Sales['Total Sales $'])
plt.xlabel('Month', fontsize=25)
plt.ylabel('Sales', fontsize=25)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)


# In[82]:


plt.figure(figsize=(20,15))
sns.barplot(x=Day_Sales['Posting Date'], y=Day_Sales['Total Sales $'])
plt.xlabel('Day', fontsize=25)
plt.ylabel('Sales', fontsize=25)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)


# In[83]:


plt.figure(figsize=(20,15))
sns.barplot(x=Month_Sales['Posting Date'], y=Month_Sales['Total Sales $'])
plt.xlabel('Month', fontsize=25)
plt.ylabel('Sales', fontsize=25)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)


# In[84]:


plt.pie(Month_Sales['Total Sales $'], labels=Month_Sales['Posting Date'],autopct='%1.1f%%', textprops={'fontsize': 14})


# In[85]:


plt.figure(figsize=(20,15))
sns.scatterplot(x='Year', y='Total Sales $', data=df_copy, s=100)
plt.xlabel('Year', fontsize=25)
plt.ylabel('Sales', fontsize=25)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)


# ## Higher Sales were achieved in 2020 than 2019

# ## November and December Have the highest Sales
# ## june, July, August Have the lowest Sales

# ## Best Selling Flavors & Products

# In[86]:


df['Flavor'].value_counts()


# In[87]:


plt.figure(figsize = (20,10))
plt.hist(df['Flavor'])


# In[88]:


df['Item/Service Description'].value_counts()


# In[89]:


plt.figure(figsize = (20,10))
plt.hist(df['Item/Service Description'])
plt.xticks(rotation=90)                                                               


# In[90]:



Flavor_Quantity = df.groupby(['Flavor']).agg({"Quantity":"sum"}).sort_values(by='Quantity', ascending=False).reset_index()
Flavor_Sales = df.groupby(['Flavor']).agg({"Total Sales $":"sum"}).sort_values(by='Total Sales $', ascending=False).reset_index()


# In[91]:


Flavor_Quantity


# In[92]:


Flavor_Sales


# In[93]:


sns.barplot(x=Flavor_Quantity['Flavor'], y=Flavor_Quantity['Quantity'])
plt.xticks(rotation=95)


# In[94]:


sns.barplot(x=Flavor_Sales['Flavor'], y=Flavor_Sales['Total Sales $'])
plt.xticks(rotation=95)


# In[95]:


plt.figure(figsize = (20,10))
sns.scatterplot(x=df['Quantity'], y=df['Total Sales $'], hue=df['Flavor'])


# In[96]:


plt.figure(figsize = (20,5))
sns.scatterplot(x=Flavor_Sales['Flavor'], y=Flavor_Sales['Total Sales $'])
plt.xticks(rotation=90)


# In[97]:


Product_Sales = df.groupby(['Item/Service Description']).agg({"Total Sales $":"sum"}).sort_values(by='Total Sales $', ascending=False).reset_index()


# In[98]:


plt.figure(figsize=(15,6))
sns.barplot(x=Product_Sales['Item/Service Description'], y=Product_Sales['Total Sales $'])
plt.xticks(rotation=95)


# In[99]:


plt.figure(figsize=(15,10))

sns.scatterplot(x=df['Quantity'], y=df['Total Sales $'], hue=df['Flavor'])


# In[100]:


Total_Sales=Total_Sales.reset_index()


# In[100]:


Cust_Sales =df.groupby(['Customer/Vendor Name']).agg({"Total Sales $":"sum"}).sort_values(by='Total Sales $', ascending=False).reset_index()


# In[101]:


Cust_Sales


# plt.figure(figsize=(20,6))
# sns.barplot(x=Cust_Sales['Customer/Vendor Name'], y=Cust_Sales['Total Sales $'])  
# plt.xticks(rotation=90)  

# In[102]:



plt.figure(figsize=(20,6))
sns.catplot(x='State', kind="count", palette="ch:.25", data=df)


# In[103]:


City_Sales =df.groupby(['City']).agg({"Total Sales $":"sum"}).sort_values(by='Total Sales $', ascending=False).reset_index()


# In[104]:


City_Sales


# In[105]:


plt.figure(figsize=(45,15))
categories = df['City'].value_counts().index
counts = df['City'].value_counts().values
plt.bar(categories, counts, width=0.5)
plt.xticks(rotation=90)  
plt.xticks(fontsize= 22)


# In[106]:


plt.figure(figsize = (20,5))
sns.scatterplot(x=City_Sales['City'], y=City_Sales['Total Sales $'])
plt.xticks(rotation=90)


# In[107]:


State_Sales= df.groupby(['State']).agg({"Total Sales $":"sum"}).sort_values(by='Total Sales $', ascending=False).reset_index()


# In[108]:


State_Sales


# In[109]:


sns.barplot(x=State_Sales['State'], y=State_Sales['Total Sales $'])  


# In[110]:


plt.figure(figsize=(20,10))
sns.barplot(x=City_Sales['City'], y=City_Sales['Total Sales $'])  
plt.xticks(rotation=90)


# In[111]:


Warehouse_Sales= df.groupby(['Warehouse Name']).agg({"Total Sales $":"sum"}).sort_values(by='Total Sales $', ascending=False).reset_index()


# In[112]:


Warehouse_Sales


# In[113]:


df['Warehouse Name'].unique()


# In[114]:


plt.figure(figsize=(25,15))
categories = Warehouse_Sales['Warehouse Name']
counts = Warehouse_Sales['Total Sales $']
plt.bar(categories, counts, width=0.5)  
plt.xlabel('Warehouse Name')
plt.ylabel('Total Sales in $ per City')
plt.xticks(fontsize=24)


# In[115]:


df


# In[116]:


df.info()


# In[117]:


plt.pie(Year_Sales['Total Sales $'], labels=Year_Sales['Posting Date'],autopct='%1.1f%%', textprops={'fontsize': 14})


# In[118]:


plt.pie(Month_Sales['Total Sales $'], labels=Month_Sales['Posting Date'],autopct='%1.1f%%', textprops={'fontsize': 14})


# In[ ]:




