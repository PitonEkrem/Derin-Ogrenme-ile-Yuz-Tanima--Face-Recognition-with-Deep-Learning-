import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.python.keras.layers import Dense,Dropout,Softmax,Flatten,Activation
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import BatchNormalization
import os
import matplotlib as plt
#kırpılmış train ve test resimlerinizin yolunu seçin...
#burada daha önce train edilmiş vgg face modeli upload ediyoruz.Bu işlem train süremizi kısaltıcak.
#kirpilmiş resimlerimizi traine hazırlıyoruz olusan datayı kaydediyoruz ayrıca datasetimizdeki isimleride txt olarak kaydettik.
train_dir='C:/Users/bjk_m/Desktop/hababam_sinif/trainingkirp/'
test_dir='C:/Users/bjk_m/Desktop/hababam_sinif/testkirp/'
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))
model.load_weights('vgg_face_weights.h5')
vgg_face=Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)

x_train=[]
y_train=[]
person_folders=os.listdir(train_dir)
person_rep=dict()
for i,person in enumerate(person_folders):

    person_rep[i]=person
    image_names=os.listdir(train_dir+person+'/')
    for image_name in image_names:
        img=load_img(train_dir+person+'/'+image_name,target_size=(224,224))
        img=img_to_array(img)
        img=np.expand_dims(img,axis=0)
        img=preprocess_input(img)
        img_encode=vgg_face(img)
        x_train.append(np.squeeze(K.eval(img_encode)).tolist())
        y_train.append(i)

x_train=np.array(x_train)
y_train=np.array(y_train)

x_train=[]
y_train=[]
person_folders=os.listdir(test_dir)

for i,person in enumerate(person_folders):
    image_names=os.listdir(test_dir+person+'/')
    for image_name in image_names:
        img=load_img(test_dir+person+'/'+image_name,target_size=(224,224))
        img=img_to_array(img)
        img=np.expand_dims(img,axis=0)
        img=preprocess_input(img)
        img_encode=vgg_face(img)
        x_train.append(np.squeeze(K.eval(img_encode)).tolist())
        y_train.append(i)

x_test=np.array(x_train)
y_test=np.array(y_train)
f = open("isimler.txt", "x")
with open('isimler.txt', 'w') as f:
 
   
   f.write(str(person_rep))
   f.close()

# Save test and train data for later use
np.save('train_data',x_train)
np.save('train_labels',y_train)
np.save('test_data',x_test)
np.save('test_labels',y_test)
