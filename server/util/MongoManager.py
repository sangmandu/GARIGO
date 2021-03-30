import collections

from pymongo import MongoClient
import datetime
client = MongoClient(['mongodb+srv://cluster0.kvjf8.mongodb.net'], 27017, username='kijoo', password='BBQqeWXzmi5EOU0Y')

db = client['GARIGO']
authCollection = db['auth']


def getUserById(id):
    user = db.auths.find_one({
        "id": id,
    })
    return user is not None


def getUser(id, pw):
    user = db.auths.find_one({
        "id": id,
        "pw": pw,
    })

    return user


def createUser(id, pw):
    user = {
        "id": id,
        "pw": pw
    }

    _id = db.auths.insert_one(user).inserted_id
    return _id


def createMedia(pid, name, index, fileUid):
    info = {
        "pid": pid,
        "name": name,
        "index": index,
        "fileUid": fileUid
    }

    _id = db.medias.insert_one(info).inserted_id
    return _id

def getImageByPid(pid):
    images = db.medias.find({
        "pid": pid
    })

    imageDict = collections.defaultdict(list)
    for image in images:
        imageDict[image['name']].append(image)

    return imageDict

