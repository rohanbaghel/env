import cv2
import numpy as np
from keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D,ZeroPadding2D,BatchNormalization,MaxPooling2D,Dropout,Lambda, Flatten, Dense,Input

input_shape=(155, 220, 1)
threshold=0.4000009313225178
weights='./Assets/signet-bhsig260-020.h5'
        
def euclidean_distance(vects):
        x, y = vects
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def create_base_network_signet(input_shape): 
    seq = Sequential()
    seq.add(Conv2D(96, kernel_size=(11, 11), activation='relu', name='conv1_1', strides=4, input_shape= input_shape, 
                            kernel_initializer='glorot_uniform'))
    seq.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))    
    seq.add(ZeroPadding2D((2, 2)))
        
    seq.add(Conv2D(256, kernel_size=(5, 5), activation='relu', name='conv2_1', strides=1, kernel_initializer='glorot_uniform'))
    seq.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(Dropout(0.3))
    seq.add(ZeroPadding2D((1, 1)))
        
    seq.add(Conv2D(384, kernel_size=(3, 3), activation='relu', name='conv3_1', strides=1, kernel_initializer='glorot_uniform'))
    seq.add(ZeroPadding2D((1, 1)))
        
    seq.add(Conv2D(256, kernel_size=(3, 3), activation='relu', name='conv3_2', strides=1, kernel_initializer='glorot_uniform'))   
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(Dropout(0.3))
    seq.add(Flatten(name='flatten'))
    seq.add(Dense(1024, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform'))
    seq.add(Dropout(0.5))
        
    seq.add(Dense(128, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform'))
        
    return seq

    

def predict_score(sample_image,input_image):
    base_network = create_base_network_signet(input_shape)

    input_a = Input(shape=(input_shape))
    input_b = Input(shape=(input_shape))

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model(inputs=[input_a, input_b], outputs=distance)
    model.load_weights(weights)
    img1 = cv2.imread(str(sample_image)) 
    img2 = cv2.imread(str(input_image))
    img1 = cv2.resize(img1, (220, 155))
    img2 = cv2.resize(img2, (220, 155))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, img1 = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY)
    ret, img2 = cv2.threshold(img2, 120, 255, cv2.THRESH_BINARY)
    img1 = np.array(img1, dtype = np.float64)
    img2 = np.array(img2, dtype = np.float64)
    img1 /= 255
    img2 /= 255
    img1 = img1[..., np.newaxis]
    img2 = img2[..., np.newaxis]
    pairs=[np.zeros((1, 155, 220, 1)) for i in range(2)]
    pairs[0][0, :, :, :]=img1
    pairs[1][0, :, :, :]=img2
    
    result = model.predict([pairs[0], pairs[1]])
    diff = result[0][0]
    print("Difference Score = ", diff)
    if diff > threshold:
        return 'Forged'
    else:
        return 'Genuine'
