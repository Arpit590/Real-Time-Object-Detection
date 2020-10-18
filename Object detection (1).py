#!/usr/bin/env python
# coding: utf-8

# # WHAT IS YOUR CARBON FOOTPRINT ?

# ## 1. Importing necessary libraries

# In[1]:


import numpy as np #For data manipulation
import cv2 # For video capture
import time


# In[2]:


net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg") # Reading weights and configuration files


# In[3]:


classes=[] # To store name of objects 
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

print(classes)


# ## 2. Video Capturing,detecting objects and storing the image captured by user after closing camera  

# In[7]:


Video = cv2.VideoCapture(0) # To video capture object for  camera
a =1 # To provide no.of frames encontured by user until he closes the camera
start_time = time.time()
Cups = cv2.imread("cups1.png")
Cup = cv2.resize(Cups,(800,800))
Tomatoes = cv2.imread("Tomatoes (1).jpg")
Tomato = cv2.resize(Tomatoes, (800,800))
Books = cv2.imread("books.png")
Book = cv2.resize(Books, (800,800))
Phones = cv2.imread("phone.png")
Phone = cv2.resize(Phones, (800,800))
while True:
    a =a+1
    _, img = Video.read()
    height, width,_ =img.shape
    blob = cv2.dnn.blobFromImage(img,1/255,(416,416), (0,0,0),swapRB=True, crop=False) # To convert image to a blob
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layersOutputs = net.forward(output_layers_names)
    
    boxes = []
    confidences =[]
    class_ids =[]
    for output in layersOutputs:
        for detection in output:
            scores = detection[5:] # Because first 4 values are of center x, center y, height and width
            class_id = np.argmax(scores)
            confidence= scores[class_id]
            if confidence>0.5: # If confidence is less than 50% then discard that box
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                
                x =int(center_x - w/2)
                y = int(center_y-h/2)
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255, size=(len(boxes), 3))
    if len(indexes)>0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2)*100)
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
            cv2.putText(img, label+" "+ confidence+"%",(x,y+20), font, 1,(0,255,255), 2)
            cv2.imshow("Object Detection",img)
        key = cv2.waitKey(1)
        if key==ord("s"):
            cv2.imwrite(r"C:\Users\dell\Desktop\ML BOOTCAMP\real time objection detection\img.png",img)
        if key==ord("c"):
            cv2.imshow("Cups",Cup)
        if key==ord("t"):
            cv2.imshow("Tomato", Tomato)
        if key==ord("b"):
            cv2.imshow("Books", Book)
        if key==ord("p"):
            cv2.imshow("Phone", Phone)
        if key==ord("q"):
            break
elapsed_time = time.time() - start_time
fps = a/elapsed_time
print("Number of frames encountered by user and fps: {} and {} respectively".format(a,np.round(fps,2)))

Video.release()
cv2.destroyAllWindows()


# In[ ]:




