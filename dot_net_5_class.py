# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:56:31 2019

@author: cks
"""


import numpy as np
import time
import tensorflow as tf
import cv2

# https://stackoverflow.com/questions/164865023/circular-masking-an-image-in-python-using-numpy-arrays
def create_circular_mask(h, w, centre=None, radius=None):

    """create circular kernels
       (kernel region is 1.0, else 0.0)

    Args:
        h (int): height of numpy array (dim 0)
        w (int): width of numpy array (dim 1)
        centre (list of 2 ints): centre of kernel region
        radius (int, float): radius of kernel region

    Returns:
        ndarray: shape (h, w)

    """

    if centre is None: # use the middle of the image
        centre = [int(w/2), int(h/2)]

    if radius is None: # use the smallest distance between the centre and image walls
        radius = min(centre[0], centre[1], w-centre[0], h-centre[1])
        
    
    radius = radius
    
    Y, X = np.ogrid[:h, :w]
    dist_from_centre = np.sqrt((X - centre[0])**2 + (Y-centre[1])**2)

    mask = dist_from_centre <= radius

    a = np.zeros((h, w))
    a[mask] = 1.0

    return a


img_size = 32
frames = 16
motion_1 = np.zeros((img_size*img_size,img_size,img_size,frames))
motion_2 = np.zeros((img_size*img_size,img_size,img_size,frames))
motion_3 = np.zeros((img_size*img_size,img_size,img_size,frames))
motion_4 = np.zeros((img_size*img_size,img_size,img_size,frames))
motion_5 = np.zeros((img_size*img_size,img_size,img_size,frames))

count = 0
centre_increment = 1
x_current = 0
y_current = 0
for i in range(img_size):
    for j  in range(img_size):
        
        x_current = i
        y_current = j
        
        for k in range(frames):
            
            
            a = create_circular_mask(img_size,img_size, centre = [x_current,y_current], radius=5)
            
            if(y_current == 31):
                y_current = 0
            else:
                y_current = y_current + centre_increment
            
            #Top to Bottom motion
            motion_1[count,:,:,k] = a
        
        count = count+1



count = 0
    
for i in range(img_size):
    for j  in range(img_size):
        
        x_current = i
        y_current = j
        
        for k in range(frames):

            
            b = create_circular_mask(img_size,img_size, centre = [x_current,y_current], radius=5)
            
            if(x_current == 31):
                x_current = 0
            else:
                x_current = x_current + centre_increment
                
            #Left to Right motion
            motion_2[count,:,:,k] = b

        count = count+1
 

count = 0
          
for i in range(img_size):
    for j  in range(img_size):
        
        x_current = i
        y_current = j
        
        for k in range(frames):

            c = create_circular_mask(img_size,img_size, centre = [x_current, y_current], radius=5)
            
            if(x_current == 31):
                x_current = 0
            else:
                x_current = x_current + centre_increment
                
            if(y_current == 31):
                y_current = 0
            else:
                y_current = y_current + centre_increment
                
            #Diagonal motion
            motion_3[count,:,:,k] = c

        count = count+1

count = 0

for i in range(img_size):
    for j  in range(img_size):
        
        x_current = i
        y_current = j
        
        for k in range(frames):

            d = create_circular_mask(img_size,img_size, centre = [x_current,y_current], radius=5)
            
            if(y_current == 0):
                y_current = 31
            else:
                y_current = y_current - centre_increment
                
            #Bottom to Top motion
            motion_4[count,:,:,k] = d

        count = count+1

count = 0

for i in range(img_size):
    for j  in range(img_size):
        
        x_current = i
        y_current = j
        
        for k in range(frames):

            e = create_circular_mask(img_size,img_size, centre = [x_current, y_current], radius=5)
            
            if(x_current == 0):
                x_current = 31
            else:
                x_current = x_current - centre_increment
                
            #Right to Left motion
            motion_5[count,:,:,k] = e

        count = count+1


            
            


        

X = []
X_test = []
Y_test = []
labels = []

#Storing motion_1 to X
test_count = 0
for i in range(img_size*img_size):
    
    
    if(test_count%20==0):
        X_test.append(motion_1[i,:,:,:])
        Y_test.append(0)
    else:
        X.append(motion_1[i,:,:,:])
        labels.append(0)
        
    test_count += 1
                

#Storing motion_2 to X
test_count = 0
for i in range(img_size*img_size):
    
    
    if(test_count%20==0):
        X_test.append(motion_2[i,:,:,:])
        Y_test.append(1)
    else:
        X.append(motion_2[i,:,:,:])
        labels.append(1)
        
    test_count += 1


#Storing motion_3 to X
test_count = 0
for i in range(img_size*img_size):
    
    
    if(test_count%20==0):
        X_test.append(motion_3[i,:,:,:])
        Y_test.append(2)
    else:
        X.append(motion_3[i,:,:,:])
        labels.append(2)
        
    test_count += 1

#Storing motion_4 to X
test_count = 0
for i in range(img_size*img_size):
    
    
    if(test_count%20==0):
        X_test.append(motion_4[i,:,:,:])
        Y_test.append(3)
    else:
        X.append(motion_4[i,:,:,:])
        labels.append(3)
        
    test_count += 1


#Storing motion_5 to X
test_count = 0
for i in range(img_size*img_size):
    
    
    if(test_count%20==0):
        X_test.append(motion_5[i,:,:,:])
        Y_test.append(4)
    else:
        X.append(motion_5[i,:,:,:])
        labels.append(4)
        
    test_count += 1
###### Creating Labels ########
        
    



X = np.array(X).reshape(-1,img_size,img_size,frames,1)
labels = np.array(labels).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,img_size,img_size,frames,1)
Y_test = np.array(Y_test).reshape(-1,1)
Y = labels




#########Visualizing Training Data #######################
    
for i in range(260):
    
    
   for j in range(16):
       cv2.imshow("Image",X_test[i,:,:,j])
       cv2.waitKey(1)

cv2.destroyAllWindows()






################## Learning Part #########################


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv3D, MaxPooling3D
from keras.callbacks import Callback, TensorBoard
import pickle




#3D filter/kernels and maxpooling window dimensions
kernal_size = 3
pooling_size = 2

#Create a an object of Sequential()
model = Sequential()

#Add one set of 3D convolution + MaxPooling to make the first layer
model.add(Conv3D(64, (kernal_size,kernal_size,kernal_size),padding = 'same', input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling3D(pool_size = (pooling_size,pooling_size,pooling_size),padding = 'same'))


#Add more sets of conv and pooling layers to increase the depth of the network
model.add(Conv3D(128, (kernal_size,kernal_size,kernal_size),padding = 'same'))
model.add(Activation("relu"))
model.add(MaxPooling3D(pool_size = (pooling_size,pooling_size,pooling_size),padding = 'same'))

model.add(Conv3D(128, (kernal_size,kernal_size,kernal_size),padding = 'same'))
model.add(Activation("relu"))
model.add(MaxPooling3D(pool_size = (pooling_size,pooling_size,pooling_size),padding = 'same'))


model.add(Conv3D(256, (kernal_size,kernal_size,kernal_size),padding = 'same'))
model.add(Activation("relu"))
model.add(MaxPooling3D(pool_size = (pooling_size,pooling_size,pooling_size),padding = 'same'))



model.add(Conv3D(256, (kernal_size,kernal_size,kernal_size),padding = 'same'))
model.add(Activation("relu"))
model.add(MaxPooling3D(pool_size = (pooling_size,pooling_size,pooling_size),padding = 'same'))


# Flatten the output of the final conv layer so that it can be fed to the fully connected layer
model.add(Flatten())

#Add the fully connected layer (Dense)
model.add(Dense(512))

model.add(Activation("relu"))

#Add the output layer with 2 neurons for the binary classifier(0 or 1)
model.add(Dense(5))
model.add(Activation("softmax"))

model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])




# Pre-shuffle the data to introduce invariance while learning
from sklearn.utils import shuffle

X_train, Y_train = shuffle(X,Y)

#One hot encoding of labels
from keras.utils.np_utils import to_categorical   
Y_labels = to_categorical(Y_train, num_classes=5)

#Callbacks

#cb = TensorBoard()

#Start the training of the model
history = model.fit(X_train,Y_train,batch_size = 8, epochs=1, validation_split = 0.1, shuffle = 'True')


model.save("C:/Users/cks/Documents/practice codes/dot_3d_cnn_models/model_5_class.h5")


Y_test_pred = model.predict(X_test)

Y_test_pred_classes = np.argmax(Y_test_pred,axis=1)
Y_test_pred_classes = Y_test_pred_classes.reshape(-1,1)

motion_list = ["Down ↓","Right →","Diagonal ↘","Up ↑","Left ←"]

motion_pred = []
for i in range(260):
    
    motion_pred.append(motion_list[Y_test_pred_classes[i][0]])


