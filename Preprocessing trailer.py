import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
DNN = "TF" 
min_confidence = 0.1
if DNN == "CAFFE":
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile= "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "opencv_face_detector_uint8.pb"
    configFile= "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    filename1 ='C:/Users/Ashima/Desktop/new videos/images/'
framecount = 0
count = 0
demo=0
success=1
# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# org 
org = (50, 50) 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2

cap = cv2.VideoCapture('C:/Users/Ashima/Desktop/new videos/Trailer1.avi') #input trailer 
fps = int(cap.get(cv2.CAP_PROP_FPS))
start = time.time()
while success:
    
    success, frame = cap.read()
    if(success):
        frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame1 = cv2.resize(frame,(int(600),int(400)))
        (h, w) = frame1.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame1, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),swapRB=False, crop=False)
        demo += 1

        net.setInput(blob)
        detections = net.forward()
        if len(detections) > 0:
             i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]
            if confidence > min_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                print("BOX:",box)
                (startX, startY, endX, endY) = box.astype("int")   

                text = "{:.2f}%".format(confidence * 100)
                if startY - 10 > 10: 
                    y = startY - 10 
                else:
                    y = startY + 10

                cv2.imwrite(filename1+'/'+str(framecount)+"swing.png",frame1)
                framecount += 1
            else:

                count += 1
                
        
    else:
        end = time.time()
        cap.release()
        cv2.destroyAllWindows()
        (h, w) = frame1.shape[:2]
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > min_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame1, (startX, startY), (endX, endY),(0, 0, 255), 2)
                cv2.putText(frame1, texti, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


framecount = 0
count = 0
success=1

