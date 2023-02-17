#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import all necessary packages
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import os

#set directory
os.chdir("C:/Users/Sofia/Downloads")

#import datasets
data = pd.read_csv('HW1_instacart.csv')
data_2 = pd.read_csv('HW1_products.csv')

#merge datasets to use product names instead of IDs
df = pd.merge(data, data_2, how="left", on=["product_id"])[['order_id','product_name']]

df.head(5)


# In[3]:


#Group dataset by order ID
grouped_df = df.groupby('order_id')['product_name'].apply(list)
orders_list = grouped_df.tolist()
orders_list[:2]


# In[4]:


#Group dataset by order ID (MINE)
#df = df.groupby('order_id')['product_name'].apply(lambda x: ','.join(x.astype(str))).reset_index()

#orders_dict = list(df["product_name"].apply(lambda x:x.split(",") ))
#orders_dict[:2]


# In[5]:


#Drop any order that has less than 2 products
item_database = []
for record in orders_list:
    if len(record) > 1:
        item_database.append(record)

item_database[:2]


# In[6]:


#convert dataset to binary format
te = TransactionEncoder()
te_ary = te.fit(item_database).transform(item_database)

df = pd.DataFrame(te_ary, columns=te.columns_)


# In[7]:


#Find the frequent itemsets with highest support
freq_items = apriori(df, min_support = 0.01,use_colnames = True)

freq_items.head


# In[8]:


#sort frequent itemsets to get top 3 rules
freq_items_sort = freq_items.sort_values(by = 'support', ascending = False)
freq_items_sort.head(3)


# In[9]:


#find top 3 rules by support
rules = association_rules(freq_items, metric = "confidence", min_threshold=0.1)

rules_support = rules.sort_values(by = 'support', ascending=False)
rules_support.head(9)


# In[11]:


#find top 3 rules by lift
rules_lift = rules.sort_values(by = 'lift', ascending=False)
rules_lift.head(50)

