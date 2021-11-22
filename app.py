from types import prepare_class
from flask import *
import pyrebase
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from urllib.request import urlopen
from signetmodel import predict_score

config={
    'apiKey':'AIzaSyAd9wxtpZNk6J_gKEmBPggo6ZRhRCCe8fM',
    'authDomain':'forgery-60388.firebaseapp.com',
    'databaseURL':'https://forgery-60388-default-rtdb.asia-southeast1.firebasedatabase.app/',
    'storageBucket': "gs://forgery-60388.appspot.com",
}

firebase=pyrebase.initialize_app(config)
auth=firebase.auth()
db=firebase.database()
storage=firebase.storage()

url=db.child('Users').child('rT0LC0DrjcgrYlXvM4kKzR9RfwC3').child('Signatures').get()

x=url.val()

imagelist=[]
for key,value in x.items():
    imagelist.append(value)

first=urlopen(imagelist[0])
with open('image1.jpg','wb') as f:
    f.write(first.read())
    f.close()
second=urlopen(imagelist[1])
with open('image2.jpg','wb') as f:
    f.write(second.read())
    f.close()

print(predict_score('./image1.jpg','./image2.jpg'))