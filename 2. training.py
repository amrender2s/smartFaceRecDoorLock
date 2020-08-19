#import all necessary libraries.
import pandas as pd
import numpy as np
import os
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import math



# Here we load the pre-trained facenet model.
file_path="model/facenet_keras.h5"
base_model=load_model(file_path)


# Here we print out the summary (all layers) of loaded facenet model and count of total numbers of layers.
print('Pretrained Model Summary')
print(base_model.summary())
print('No. of layers ', len(base_model.layers))


# By default every layer has training enabled, thats why we are disableing the training of pre-trained model.
for layer in base_model.layers:
    layer.trainable = False
    
    
    
# Here we removed the last 4 layers as following:-

#AvgPool (GlobalAveragePooling2D (None, 1792)         0           Block8_6_ScaleSum[0][0]          
#__________________________________________________________________________________________________
#Dropout (Dropout)               (None, 1792)         0           AvgPool[0][0]                    
#__________________________________________________________________________________________________
#Bottleneck (Dense)              (None, 128)          229376      Dropout[0][0]                    
#__________________________________________________________________________________________________
#Bottleneck_BatchNorm (BatchNorm (None, 128)          384         Bottleneck[0][0]    
x=base_model.layers[-5].output


# Then we added 6 new layers as following:-
x=GlobalAveragePooling2D()(x)
x=Dense(1024, activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024, activation='relu')(x) #dense layer 2
x=Dense(512,  activation='relu')(x) #dense layer 3
x=Dropout(0.25)(x)
preds=Dense(4, activation='softmax')(x) #final layer with softmax activation

# Here we create the object of model, inputs=base_model.input is used to set the input parameter from facenet model and outputs=preds is used so that model will use it for output (It will give output as per given the of image classes).
model=Model(inputs=base_model.input,outputs=preds)

# Again we have printed the summary of new model, so that we can check the added layers.
print(model.summary())
print('No. of layers ', len(model.layers))



# Let's print and check all layers whether the only added layers are trainable or not.
for (i,layer) in enumerate(model.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)
    


# Directory of training and validation dataset.
train_data_dir = 'facedata/train'
validation_data_dir = 'facedata/val'


#The batch size defines the number of samples that will be propagated through the network. For instance, let's say you have 1050 training samples and you want to set up a batch_size equal to 100.

#The batch size is a number of samples processed before the model is updated. The number of epochs is the number of complete passes through the training dataset. The size of a batch must be more than or equal to one and less than or equal to the number of samples in the training dataset.
batchSize=1000

# Required input image size for our model (our model only accepts 160x160x3 colour images).
imageSize=160



# Here we have done data augumentation of training and validation dataset. From line 82 to 101.
train_datagen=ImageDataGenerator( rescale=1./255,
                                  rotation_range=45,
                                  width_shift_range=0.3,
                                  height_shift_range=0.3,
                                  horizontal_flip=True,
                                  fill_mode='nearest'
                                ) 
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(train_data_dir,
                                                  target_size=(imageSize,imageSize),
                                                  color_mode='rgb',
                                                  batch_size=batchSize,
                                                  class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(imageSize, imageSize),
        color_mode='rgb',
        batch_size=batchSize,
        class_mode='categorical')
   
   
   
   
# Here we have used EarlyStopping, because after certain epochs if the validation accuracy not increases then training will be stoped at that epoch only.
earlystop = EarlyStopping(monitor = 'val_acc', 
                          min_delta = 0, 
                          patience = 15,
                          verbose = 1,
                          mode='max'
                        )
# Here we have used ModelCheckpoint, because save the best trained model in the model directory with the help  of earlystopping.
checkpoint = ModelCheckpoint("model/model.h5",
                             monitor="val_acc",
                             mode='max',
                             save_best_only = True,
                             verbose=1)

# Here callbacks act as a sequence of running of earlystopping and checkpoint respectively.
callbacks = [earlystop, checkpoint]    


# Here we allocate the used optimizer, loss factor and metrices, by the help of compile function.
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Here we have initialized the the total number of images in training and validation folder. And number of epochs for training our model.
nb_train_samples = 92
nb_validation_samples = 82
epochs = 20


# Here we have initialized steps per epochs for training and validation used in model.fit_generator() function as follow.
step_size_train=int(math.ceil(1.0 * nb_train_samples / batchSize))
step_size_validation=int(math.ceil(1.0 * nb_train_samples / batchSize))




# This is part of code with actually do the training.
history = model.fit_generator(train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=epochs,
                    callbacks = callbacks,
                    validation_data = validation_generator,
                    validation_steps = step_size_validation)
                    
                    
                    
                    
                    
      
      
                    
###### All these following is to visualize the confusion matrix.
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# We need to recreate our validation generator with shuffle = false
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(imageSize, imageSize),
        batch_size=batchSize,
        class_mode='categorical',
        shuffle=False)

class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())

nb_train_samples = 24
nb_validation_samples = 14

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(validation_generator, step_size_validation)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = list(class_labels.values())
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

plt.figure(figsize=(8,8))
cnf_matrix = confusion_matrix(validation_generator.classes, y_pred)

plt.imshow(cnf_matrix, interpolation='nearest')
plt.colorbar()
tick_marks = np.arange(len(classes))
_ = plt.xticks(tick_marks, classes, rotation=90)
_ = plt.yticks(tick_marks, classes)






# Output of confusion matrix and all.

'''

Found 14 images belonging to 4 classes.
Confusion Matrix
[[5 0 0 0]
 [0 2 0 0]
 [0 0 4 0]
 [0 0 2 1]]
Classification Report
              precision    recall  f1-score   support

        Ajay       1.00      1.00      1.00         5
       Ajeet       1.00      1.00      1.00         2
 Aman pandey       0.67      1.00      0.80         4
        Amit       1.00      0.33      0.50         3

    accuracy                           0.86        14
   macro avg       0.92      0.83      0.82        14
weighted avg       0.90      0.86      0.84        14
'''

