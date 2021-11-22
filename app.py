from signetmodel import predict_score
from urllib.request import urlopen
from urllib import request
from flask import Flask
import pyrebase
import warnings
warnings.filterwarnings('ignore')   
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = {
    'apiKey': 'AIzaSyAd9wxtpZNk6J_gKEmBPggo6ZRhRCCe8fM',
    'authDomain': 'forgery-60388.firebaseapp.com',
    'databaseURL': 'https://forgery-60388-default-rtdb.asia-southeast1.firebasedatabase.app/',
    'storageBucket': "gs://forgery-60388.appspot.com",
}


app = Flask(__name__)

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
db = firebase.database()
storage = firebase.storage()


@app.route('/compile', methods=['GET'])
def complie():
    uid = request.args.get('uid')
    url = db.child('Users').child(
        uid).child('Signatures').get()

    x = url.val()

    imagelist = []
    for key, value in x.items():
        imagelist.append(value)

    first = urlopen(imagelist[0])
    with open('image1.jpg', 'wb') as f:
        f.write(first.read())
        f.close()
    second = urlopen(imagelist[1])
    with open('image2.jpg', 'wb') as f:
        f.write(second.read())
        f.close()

    result = predict_score('./image1.jpg', './image2.jpg')
    return result

if __name__ == "__main__":
    app.run(debug=True)