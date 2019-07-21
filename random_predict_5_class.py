# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:10:05 2019

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


x_map_reset_1 = 0
x_map_reset_2 = 31
y_map_reset_1 = 0
y_map_reset_2 = 31
increment_multiplier = 1

x_map = 8
y_map = 8
snake_motion = np.zeros((200,img_size,img_size,frames))







for i in range(200):

    z = np.random.randint(0,5)
        
    if(z==0):
    
        for j in range(frames):
            snake_motion[i,:,:,j] = create_circular_mask(img_size,img_size, centre = [x_map,y_map], radius = 5)
            
            if(x_map==x_map_reset_2):
                x_map = x_map_reset_1
            else:
                x_map = x_map + increment_multiplier
            print(x_map,y_map)

            
            
        
    
    if(z==1):
        for j in range(frames):
            snake_motion[i,:,:,j] = create_circular_mask(img_size,img_size, centre = [x_map,y_map], radius = 5)
            
            if(y_map == y_map_reset_2):
                y_map = y_map_reset_1
            else:
                
                y_map = y_map + increment_multiplier
            print(x_map,y_map)

            
            
            
            
        
    if(z==2):
        for j in range(frames):    
        
            snake_motion[i,:,:,j] = create_circular_mask(img_size,img_size, centre = [x_map,y_map], radius = 5)
            if(x_map == x_map_reset_1):
                x_map = x_map_reset_2
            else:
                x_map = x_map - increment_multiplier   
            print(x_map,y_map)

            
            
            
            
        
    if(z==3):
        for j in range(frames):
            
            snake_motion[i,:,:,j] = create_circular_mask(img_size,img_size, centre = [x_map,y_map], radius = 5)
            if(y_map == y_map_reset_1):
                y_map = y_map_reset_2
            else:
                y_map = y_map - increment_multiplier
            print(x_map,y_map)

            
            
            
    
    if(z==4):
        for j in range(frames):
            
            snake_motion[i,:,:,j] = create_circular_mask(img_size,img_size, centre = [x_map,y_map], radius = 5)
            
            if(x_map == x_map_reset_2):
                x_map = x_map_reset_1
            else:
                x_map = x_map + increment_multiplier
                
            if(y_map == y_map_reset_2):
                y_map = y_map_reset_1
            else:
                y_map = y_map + increment_multiplier
            print(x_map,y_map)

            
          

 


X_test = snake_motion.reshape(-1,img_size,img_size,frames,1)



##### Converting the input data into a continuous data to replicate a realtime video
##### We consider 1 new frame at a time in the prediction 
##### The new frame will be added to the previous 15 frames to get a total of 16 frames (sliding window)
X_test_cont = np.zeros((img_size,img_size,100*16,1))


count = 0
for i in range(100):
    
    
    X_test_cont[:,:,count:count+16,:] = X_test[i,:,:,:] 
    
    count = count+16
    






### Visualizinv the continuous stack of data 
#
#for j in range(100*16):
#       cv2.imshow("Image",X_test_cont[:,:,j,:])
#       cv2.waitKey(1)
#       time.sleep(0.2)
#
#cv2.destroyAllWindows()

    

for i in range(100):
    
    
   for j in range(16):
       cv2.imshow("Image",snake_motion[i,:,:,j])
       cv2.waitKey(1)
       #time.sleep(0.1)

cv2.destroyAllWindows()




from keras.models import load_model



#### Visualization of the moving dot along with the prediction made####
#### 5 Conv Layer Network 64,128,128,256,256 #####

new_model = load_model("C:/Users/cks/Documents/practice codes/dot_3d_cnn_models/model_5_class.h5")


Y_test_pred = new_model.predict(X_test)

Y_test_pred_classes = np.argmax(Y_test_pred,axis=1)
Y_test_pred_classes = Y_test_pred_classes.reshape(-1,1)


motion_list = ["Down ↓","Right →","Diagonal ↘","Up ↑","Left ←"]

motion_pred = []



###### Without sliding window one prediction per one data ##########
for i in range(200):
    
    motion_pred.append(motion_list[Y_test_pred_classes[i][0]])
    

for i in range(20):
    
    for j in range(16):
        
        cv2.imshow("Image",snake_motion[i,:,:,j])
        cv2.waitKey(1)
        print(motion_pred[i])
        time.sleep(0.1)
            
cv2.destroyAllWindows()
  


#### Taking the looping square motion framy by frame and making prediction 


#### Manual looping for self-validation      
#i=28
#temp_frames = X_test_cont[:,:,i:i+16,:]
#temp_frames = temp_frames.reshape(1,img_size,img_size,frames,1)
#
#
#
#for j in range(16):
#    cv2.imshow("Image",temp_frames[0,:,:,j,:])
#    cv2.waitKey(1)
#    time.sleep(0.2)
#
#    
#cv2.destroyAllWindows()
#
#temp_pred = new_model.predict(temp_frames)
#temp_class = np.argmax(temp_pred,axis = 1)
#
#temp_class = np.array(temp_class).reshape(1,1)
#
#temp_motion = motion_list[temp_class[0][0]]
#
#print(temp_motion)






motion_pred_fbf = []
Y_test_pred_classes_fbf = []
Y_test_pred_fbf = []
Y_test_pred_motion_fbf = []

for i in range(1600):
    if(i== 1600-16):
        break
    
    temp_frames = X_test_cont[:,:,i:i+16,:]
    temp_frames = temp_frames.reshape(1,img_size,img_size,frames,1)
    temp_pred = new_model.predict(temp_frames)
    temp_class = np.argmax(temp_pred,axis = 1)
    
    
    temp_class = np.array(temp_class).reshape(1,1)

    temp_motion = motion_list[temp_class[0][0]]

    
    Y_test_pred_classes_fbf.append(temp_class)
    Y_test_pred_fbf.append(temp_pred)
    Y_test_pred_motion_fbf.append(temp_motion) 

### Frame by Frame motion of the looping square along with the prediction (sliding window)
    for j in range(16):
        cv2.imshow("Image",temp_frames[0,:,:,j,:])
        cv2.waitKey(1)
        time.sleep(0.1)
        
    print(temp_motion)



Y_test_pred_classes_fbf = np.array(Y_test_pred_classes_fbf).reshape(-1,1)


direction_tracker = np.zeros((len(Y_test_pred_motion_fbf)-1,1))

for i in range(1,len(Y_test_pred_motion_fbf)-1):
    
    prev_motion = Y_test_pred_motion_fbf[i-1]
    next_motion = Y_test_pred_motion_fbf[i+1]
    current_motion = Y_test_pred_motion_fbf[i]
    
    if(current_motion != prev_motion):
        
        direction_tracker[i] = 1
    
    
