from pymongo import MongoClient
import datetime

client = MongoClient(['mongodb+srv://cluster0.hetsy.mongodb.net'], 27017, username='sumin', password='AjOUqvMAUryqMMBe')

db = client['GARIGO']
authCollection = db['auth']


def getUserById(id):
    user = db.auths.find_ond({
        "id": id,
    })
    return user is not None


def getUser(id, pw):
    user = db.auths.find_ond({
        "id": id,
        "pw": pw,
    })

    return user is not None


def createUser(id, pw):
    user = {
        "id": id,
        "pw": pw
    }

    _id = db.auths.insert_one(user).inserted_id
    return _id
