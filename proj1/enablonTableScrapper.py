#!/usr/bin/env python
# coding: utf-8

# In[54]:


import re
import requests
import time
import pandas as pd
from tqdm import tqdm
from requests.auth import HTTPBasicAuth

savePath = ""
URL      = "https://www99.enablon.com/~~~"
userName = "username"
password = "password"

res = requests.get(URL, auth=HTTPBasicAuth(userName, password))

pattern    = '"name":"[a-zA-Z0-9_]*"'
tableNames = re.compile(pattern).findall(res.text)
tableNames = [tableName.split(":")[1].strip('"')+"/" for tableName in tableNames]
tableNames = tqdm(tableNames)
for tableName in tableNames:
    time.sleep(0.1)
    tableNames.set_description(f'Processing {tableName}')
    
    res = requests.get(URL+tableName, auth=HTTPBasicAuth(userName, password))
    tableInfo = res.json()['value']
    dfDict, headers = {}, []
    for row in tableInfo:
        for key, value in row.items():
            if key not in headers:
                headers.append(key)
                dfDict[key] = [value]
            else:
                dfDict[key].append(value)
    
    df = pd.DataFrame(dfDict)
    df.to_csv(savePath + '/' + tableName[:-1] + '.csv')
    
print("tables were extracted successfully.")

