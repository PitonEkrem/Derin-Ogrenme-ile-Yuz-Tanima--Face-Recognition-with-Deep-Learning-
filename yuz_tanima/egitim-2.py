from cv2 import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
import dlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K
# Load saved data
x_train=np.load('train_data.npy')
y_train=np.load('train_labels.npy')
x_test=np.load('test_data.npy')
y_test=np.load('test_labels.npy')
print('data load basarılı...')
# burada sınıflandırıcmızı olusturuyoruz kac adet isim var ise dense units=12 kısmını değiştirin.
#örnek 18 adet isim var ise dense units=12 yerine 18 olacak
#modeli 100 tur eğitip oluşturduğumuz modeli ve basarı grafiğini kaydettik.
classifier_model=Sequential()
classifier_model.add(Dense(units=100,input_dim=x_train.shape[1],kernel_initializer='glorot_uniform'))
classifier_model.add(BatchNormalization())
classifier_model.add(Activation('tanh'))
classifier_model.add(Dropout(0.3))
classifier_model.add(Dense(units=10,kernel_initializer='glorot_uniform'))
classifier_model.add(BatchNormalization())
classifier_model.add(Activation('tanh'))
classifier_model.add(Dropout(0.2))
classifier_model.add(Dense(units=12,kernel_initializer='he_uniform'))
classifier_model.add(Activation('softmax'))
classifier_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer='nadam',metrics=['accuracy'])
tf.keras.models.save_model(classifier_model,'face_classifier_model.h5')

history = classifier_model.fit(x_train, y_train, epochs=100, 
                    validation_data=(x_test, y_test))
plt.plot(history.history['accuracy'], label='basari')
plt.plot(history.history['val_accuracy'], label = 'val_basari')
plt.xlabel('Tur')
plt.ylabel('Basari')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig('grafik.jpg')
print('kaydedildi...')


