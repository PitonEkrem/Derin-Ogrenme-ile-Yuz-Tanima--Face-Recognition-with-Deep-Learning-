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
import cv2
import matplotlib.pyplot as plt
import dlib
import glob

testresim_dir='C:/Users/bjk_m/Desktop/hababam_sinif/testler/'
sonuc_dir='C:/Users/bjk_m/Desktop/hababam_sinif/sonuclar/'
classifier_model=tf.keras.models.load_model('face_classifier_model.h5')
dnnFaceDetector=dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
#testler ve sonuçlar adında 2 klasaör oluşturun.Testler klasörüne test etmek istediğiniz resimleri kaydedin.
#person_rep=txt olarak kaydettiğimiz isimleri kopyalayıp yapıştırın.
#sonuclarınız sonuclar klasörüne kaydedilecektir.
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

person_rep={0: 'Feridun Savli', 1: 'Adile Nasit', 2: 'Cem Gurdap', 3: 'Ahmet Ariman', 4: 'Kemal Sunal', 5: 'Cengiz Nezir', 6: 'Halit Akcatepe', 7: 'Sener Sen', 8: 'Munir Ozkul', 9: 'Tarik Akan', 10: 'Mete Akyol', 11: 'Mikail Akyol'}

for img_name in os.listdir(testresim_dir):
  if img_name=='crop_img.jpg':
    continue
  # Load Image
  img=cv2.imread(testresim_dir+img_name)
  gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Detect Faces
  rects=dnnFaceDetector(gray,1)
  left,top,right,bottom=0,0,0,0
  for (i,rect) in enumerate(rects):
    # Extract Each Face
    left=rect.rect.left() #x1
    top=rect.rect.top() #y1
    right=rect.rect.right() #x2
    bottom=rect.rect.bottom() #y2
    width=right-left
    height=bottom-top
    img_crop=img[top:top+height,left:left+width]
    cv2.imwrite(testresim_dir+'/crop_img.jpg',img_crop)
    
    # Get Embeddings
    crop_img=load_img(testresim_dir+'/crop_img.jpg',target_size=(224,224))
    crop_img=img_to_array(crop_img)
    crop_img=np.expand_dims(crop_img,axis=0)
    crop_img=preprocess_input(crop_img)
    img_encode=vgg_face(crop_img)

    # Make Predictions
    embed=K.eval(img_encode)
    person=classifier_model.predict(embed)
    name=person_rep[np.argmax(person)]
    os.remove(testresim_dir+'/crop_img.jpg')
    cv2.rectangle(img,(left,top),(right,bottom),(0,255,0), 2)
    img=cv2.putText(img,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA)
    img=cv2.putText(img,str(np.max(person)),(right,bottom+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
  # Save images with bounding box,name and accuracy 
  cv2.imwrite(sonuc_dir+img_name,img)
