from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("MONGODB_DB")]
collection = db[os.getenv("MONGODB_COLLECTION")]

print("Total feature rows:", collection.count_documents({}))

latest = collection.find_one(sort=[("timestamp", -1)])
print("Latest row:\n", latest)
