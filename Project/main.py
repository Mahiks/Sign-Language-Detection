from tensorflow.keras.utils import load_img, img_to_array 
import os,numpy as np
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import cv2
import pyttsx3
model = keras.models.load_model('sign_detection/')
model_vgg = VGG16(weights='imagenet', include_top=False)
video_object = cv2.VideoCapture(0)
label_ref = {0: 'nothing',
 1: 'space',
 2: 'del',
 3: 'hi',
 4: 'hello',
 5: 'how',
 6: 'you',
 7: 'morning',
 8: 'evening',
 9: 'afternoon',
 10: 'night',
 11: 'water',
 12: 'fine',
 13: 'person',
 14: 'dog',
 15: 'cat',
 16: 'walk',
 17: 'run',
 18: 'stand',
 19: 'sit',
 20: 'eat',
 21: 'drink',
 22: 'thanks',
 23: 'welcome',
 24: 'stop',
 25: 'come',
 26: 'go',
 27: 'good',
 28: 'bad'}

while True:
 ret,frame = video_object.read() 
 cv2.imshow('Frames',frame) 
 image_array = cv2.resize(frame, (224,224), interpolation = cv2.INTER_AREA)
 image_array = img_to_array(image_array)
 test = preprocess_input(image_array)
 test = np.expand_dims(test,axis=0)
 test_predict = model_vgg.predict(test)
 test_predict = test_predict.reshape(test_predict.shape[0],25088)
 pred = model.predict(test_predict)
 pred_tmp = np.argmax(pred[0])
 print("Result",label_ref[pred_tmp])
 if cv2.waitKey(10) & 0xFF == ord('q'):
      break
 # initialisation
 engine = pyttsx3.init() 
 # testing
 engine.say(label_ref[pred_tmp])
 engine.runAndWait() 
 
 	
 