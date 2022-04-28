import os
import cv2
import numpy as np
import dlib
class Image():
    def __init__(self, name, file_name, dir_path):
        self.name = name
        self.file_name = file_name
        self.dir_path = dir_path


#Burada input_dir=datasetiniz bulunduğu konumu giriniz. output için ise kırpılmıs resimleri nereye kaydetmek
#istiyorsanız orayı seçin sizin klasör oluşturmanıza gerek yoktur.Datasetinizde cok fazla resim varsa veya
#imwrite hatası alıyorsanız face detect için haarcascade kullanmanızı tavsiye ediyorum.
#bu kod sizin datasetinizdeki yüzleri tespit edip kırpar hem dataset hemde test klasörünüzdeki resimler için uygulayın..
input_dir = 'C:/Users/bjk_m/Desktop/hababam_sinif/testler'
output_dir ='C:/Users/bjk_m/Desktop/hababam_sinif/kirpilmistestler'
if os.path.exists(output_dir)!=1:
        os.makedirs(output_dir)

classes = []
for name in os.listdir(input_dir):
        class_dir = input_dir + '\\' + name
        faces = os.listdir(class_dir)
        classes.append(Image(name,faces,class_dir))


dnnFaceDetector=dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
for name in classes:
    
    name_dir_write = output_dir + '\\' + name.name
    if os.path.exists(name_dir_write) != 1:
        os.makedirs(name_dir_write)
    
    for face_file in name.file_name:
        
        file_path_read = name.dir_path+'\\'+face_file
        img = cv2.imread(file_path_read)
       
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects=dnnFaceDetector(gray,1)
        left,top,right,bottom=0,0,0,0
    for (i,rect) in enumerate(rects):
     left=rect.rect.left() #x1
     top=rect.rect.top() #y1
    right=rect.rect.right() #x2
    bottom=rect.rect.bottom() #y2
    width=right-left
    height=bottom-top
    img_crop=img[top:top+height,left:left+width]
    resized_face = cv2.resize(img_crop, (224, 224))
    resized_face_path = name_dir_write+'\\'+face_file
    cv2.imwrite(resized_face_path, resized_face)
       
        