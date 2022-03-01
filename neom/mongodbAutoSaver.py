#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pymongo
import pandas as pd
from pymongo import MongoClient
from pymongo.cursor import CursorType

myClient = pymongo.MongoClient("mongodb://localhost:27017/")
cursor   = myClient["dbname"]["collection name"].find({"boardId": "alphabetKey"})
df       = pd.DataFrame(list(cursor))

columnNames = ["field1", "field2", "...", "fieldN"]

df.loc[df['field1'] != True, 'field1'] = False
df['field2'] = pd.to_numeric(df['field2'], errors='coerce').fillna(0)

dfCopy = df.copy(deep=True)
# dfCopy.to_csv('./pastDF.csv') # copy version save to a directory which includes this python file.

db               = myClient["dbname"]
backupCollection = db["backupCollection name"]
backupCollection.drop()
backupCollection.insert_many(dfCopy.to_dict('records'))

# to check whether this code is proceeded successfully.
# dfCheck = pd.DataFrame(list(backupCollection.find()))

