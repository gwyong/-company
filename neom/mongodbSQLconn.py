import pymssql, pymysql
import pandas as pd

server   = "servername"
database = "dbname"
username = "id"
password = "password"
portnum  = "portnumber" # int type

## MSSQL
conn   =  pymssql.connect(server, username, password, database, charset='utf8')
cursor = conn.cursor()

## MYSQL
# conn   = pymysql.connect(host=server, port=portnum, user=username, password=password, db=database, charset='utf8')
# curs = conn.cursor(pymysql.cursors.DictCursor)

df = pd.read_csv("./dataframe.csv").drop(["dropcolumn"], axis=1)
df = df.fillna("None") # filling in missing values

## change header names to avoid specific rules of SQL table
newHeaders = ['_'.join(key.split(' ')) for key in df.keys()]
newHeaders = [key.replace('(',"") for key in newHeaders]
newHeaders = [key.replace(')',"") for key in newHeaders]
newHeaders = [key.replace('.',"") for key in newHeaders]
newHeaders = [key.replace('/',"") for key in newHeaders]
newHeaders = [key.replace('-',"_") for key in newHeaders]
newHeaders = [key.replace('__',"_") for key in newHeaders]

headerDic = {}
for i in range(len(df.keys())):
    headerDic[df.keys()[i]] = newHeaders[i]
df = df.rename(columns=headerDic)

## create table
query = "CREATE TABLE dbo.tablename("
for key in df.keys():
    query += (key+' varchar(max) NOT NULL, ')
query = query[:-2]+');'

cursor.execute(query)

## converting dataframe to table in SQL
makeStringQuery = ["%s"]*len(df.keys())
makeStringQuery = ', '.join(makeStringQuery)
makeStringQuery = "(" + makeStringQuery + ");"
query = "INSERT INTO dbo.tablename VALUES " + makeStringQuery
data  = tuple(map(tuple, df.values))

cursor.executemany(query, data)
cursor.close()
conn.close()