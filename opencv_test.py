import cv2
import math

videoFile = "/home/shiqi/opencv/chessboard.avi"
imagesFolder = "/home/shiqi/opencv/data/"
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId <150):
        filename = imagesFolder + "image_" +  str(int(frameId)) + ".jpg"
        cv2.imwrite(filename, frame)
cap.release()
print ("Done!")

