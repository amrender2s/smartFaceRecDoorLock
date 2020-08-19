#LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1 thonny &
#LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1.2.0 python3

# Imported the necessary libraries.

import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np 
import os
import sys
from os import listdir
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
# import imutils
import RPi.GPIO as GPIO
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Mail section import
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Email you want to send the update from (only works with gmail)
fromEmail = '*********@gmail.com'
fromEmailPassword = '**********'

# Email you want to send the update to
toEmail = '*************@gmail.com'


def sendEmail(image):
    msgRoot = MIMEMultipart('related')
    msgRoot['Subject'] = 'Security Update'
    msgRoot['From'] = fromEmail
    msgRoot['To'] = toEmail
    msgRoot.preamble = 'Raspberry pi security camera update'

    msgAlternative = MIMEMultipart('alternative')
    msgRoot.attach(msgAlternative)
    msgText = MIMEText('Smart security cam found unknown person')
    msgAlternative.attach(msgText)

    text = MIMEText("Smart security cam found unknown persons/")
    msgRoot.attach(text)
    msgImage = MIMEImage(image)
    msgImage.add_header('Content-ID', '<image1>')
    msgRoot.attach(msgImage)

    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    smtp.ehlo()
    smtp.starttls()
    smtp.ehlo()
    smtp.login(fromEmail, fromEmailPassword)
    smtp.sendmail(fromEmail, toEmail, msgRoot.as_string())
    smtp.quit()



# Setup the used raspberry pins.

relay_pin = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(relay_pin, GPIO.OUT)
GPIO.output(relay_pin, 0) 

# Load the image trained model.

classifier=load_model('model/12_facenet_keras1_6-2-20.h5')


# Here we got the dictionary of every individual.

train_data_dir = 'faces/train'
imageSize=160
batchSize=1000

train_datagen=ImageDataGenerator( rescale=1./255,
                                  rotation_range=45,
                                  width_shift_range=0.3,
                                  height_shift_range=0.3,
                                  horizontal_flip=True,
                                  fill_mode='nearest'
                                ) 
train_generator=train_datagen.flow_from_directory(train_data_dir,
                                                  target_size=(imageSize,imageSize),
                                                  color_mode='rgb',
                                                  batch_size=batchSize,
                                                  class_mode='categorical',
                                                  shuffle=True
                                                  )


class_labels = train_generator.class_indices
person_dict = {v: k for k, v in class_labels.items()}
classes = list(person_dict.values())
print(person_dict)



#Setup the raspberry pi camera and face cascade.

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
camera.hflip
rawCapture = PiRGBArray(camera, size=(640, 480))
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX


# It preprocess the image according the trained model.

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(160, 160))
    img = img_to_array(img)
    img = img / 255.
    img = img.reshape(1,160,160,3) 
    return img

people_pictures = "./persons/"

all_people_faces = dict()

# It predict the pre-existing people faces and stores in the all_people_faces dictionary.

for file in listdir(people_pictures):
    person_face, extension = file.split(".")
    all_people_faces[person_face] = classifier.predict(preprocess_image('persons/%s.png' % (person_face)))[0,:]


# This function find the cosine similarity between all_people_faces dictionary and face captured by camera.

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

# It capture image by camera extract the face out of it. And then predict the face by classifier and then check the similarity with the above cosine function. And if the similary is more or equals then 90 percent then detect face and open the lock.

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = frame.array
    faces = faceCascade.detectMultiScale(frame, scaleFactor = 1.5, minNeighbors = 5)
    for (x, y, w, h) in faces:
        sub_face = frame[y:y+h, x:x+w]
        sub_face = cv2.resize(sub_face, (160, 160), interpolation = cv2.INTER_LINEAR)
        sub_face = sub_face / 255.
        sub_face = sub_face.reshape(1,160,160,3) 

        captured_representation = classifier.predict(sub_face)[0,:]
        
        person_name = person_dict[captured_representation.argmax()]
        person_name = person_name.lower()
        print(person_name)
        
        representation = all_people_faces[person_name]
        similarity = findCosineSimilarity(representation, captured_representation)
        similarity= 1-similarity    
        if similarity >= 0.90:
            GPIO.output(relay_pin, 1)
            #print('true')
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, person_name, (x, y), font, 2, (0, 0 ,255), 2,cv2.LINE_AA)            

#        elif similarity < 0.90:
#            GPIO.output(relay_pin, 0)
            
        else:
            sendEmail(sub_face)
            continue

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

    rawCapture.truncate(0)

    if key == ord("q"):
                GPIO.output(relay_pin, 0)
                break
#It close all open camera by pressing q button.                
cv2.destroyAllWindows()



