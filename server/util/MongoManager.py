from pymongo import MongoClient
import datetime

client = MongoClient(['mongodb+srv://cluster0.hetsy.mongodb.net'], 27017, username='sumin', password='AjOUqvMAUryqMMBe')
db = client['test-database']
collection = db['test-collection']
post = {"author": "Mike",
        "text": "My first blog post!",
        "tags": ["mongodb", "python", "pymongo"],
        "date": datetime.datetime.utcnow()}

posts = db.posts
post_id = posts.insert_one(post).inserted_id

print(post_id)
