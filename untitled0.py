import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from signetmodel import predict_score

result=predict_score('./genuine.png','./Untitled.png')
print(result)
