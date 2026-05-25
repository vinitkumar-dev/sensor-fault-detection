import pandas as pd
import json
from pymongo.mongo_client import MongoClient

url = 'mongodb+srv://pwskills:aBWLq7IA53wjxMpc@cluster0.o3qp0iv.mongodb.net/?appName=Cluster0'

# create a new client and connect to server
client = MongoClient(url)

DATABASE_NAME = 'pwskills'
COLLECTION_NAME ='waferfault'

df = pd.read_csv('notebooks\wafer_23012020_041211.csv')



df.drop('Unnamed: 0',axis=1,inplace=True)

json_record = list(json.loads(df.T.to_json()).values())

#is used to convert a Pandas DataFrame into a list of JSON-like records (row-wise dictionaries).

db = client[DATABASE_NAME] # db create
coll_create= db[COLLECTION_NAME] # table like

coll_create.insert_many(json_record)