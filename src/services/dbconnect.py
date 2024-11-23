from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Temp
username = dev_user
password = adQavSiPuYsXSyV1
DB_NAME = "reco_ai"

client = None
db_connect_tries = 0
URI = f"mongodb+srv://{username}:{password}@cluster0.r9atf7b.mongodb.net/{DB_NAME}"

def get_database_session():
    try:
        global db_connect_tries
        global client
        if client is None:  
            client = MongoClient(uri, connect=False)
        db = client.get_database(DB_NAME)
        return db
    except:
        db_connect_tries += 1
        if db_connect_tries==3:
            print("Unable to connect to DB!")
        else:
            get_database_session()
