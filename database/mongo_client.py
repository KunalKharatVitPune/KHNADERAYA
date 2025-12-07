
# Previous Name: analysis/mongodb_client.py
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pprint import pprint

uri = "mongodb+srv://kunalkharat2004:tyJiTGkoVPgsE7DQ@cluster0.78goccb.mongodb.net/dcrm?appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
    
    db = client['dcrm']
    
    print(f"\nCollections in '{db.name}':")
    collections = db.list_collection_names()
    for col_name in collections:
        print(f"- {col_name}")
        
    print("\nSample Documents:")
    for col_name in collections:
        print(f"\n--- Collection: {col_name} ---")
        doc = db[col_name].find_one()
        pprint(doc)
        
except Exception as e:
    print(e)
