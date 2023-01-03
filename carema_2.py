import cv2
import math
import time
import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import random
#matplotlib nbagg

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
board = aruco.CharucoBoard_create(3, 3, 1, 0.8, aruco_dict)

def color_frame (frame,markerCorners,markerIds,rejectedCandidates,color_list):
    if markerIds is not None:
        i = 0
        for marker in markerIds:
            p1 = np.squeeze(markerCorners[i])[0,:]
            p2 = np.squeeze(markerCorners[i])[1,:]
            p3 = np.squeeze(markerCorners[i])[2,:]
            p4 = np.squeeze(markerCorners[i])[3,:]

            print(marker)
            frame = cv2.line(frame, (p1[0],p1[1]), (p2[0],p2[1]), color_list[int(marker)], 2)
            frame = cv2.line(frame, (p2[0],p2[1]), (p3[0],p3[1]), color_list[int(marker)], 2)
            frame = cv2.line(frame, (p4[0],p4[1]), (p1[0],p1[1]), color_list[int(marker)], 2)
            frame = cv2.line(frame, (p4[0],p4[1]), (p3[0],p3[1]), color_list[int(marker)], 2)
            frame = cv2.putText(frame,f'{int(marker)}',(p1[0] ,p1[1] ),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),2)
            i = i + 1
        del i, markerCorners, markerIds
    return frame

def track(matrix_coefficients, distortion_coefficients):
    while True:
        ret, frame = cap.read()
        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)  # Use 5x5 dictionary to find markers
        parameters = aruco.DetectorParameters_create()  # Marker detection parameters
        # lists of ids and the corners beloning to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if np.all(ids is not None):  # If there are markers found by detector
            for i in range(0, len(ids)):  # Iterate in markers
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                           distortion_coefficients)
                (rvec - tvec).any()  # get rid of that nasty numpy value array error
                aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers
                aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  # Draw Axis
        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):  # Quit
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    imagesFolder = "/home/shiqi/opencv/data/"
    mtx = np.genfromtxt(imagesFolder+"calib_mtx_webcam.csv")
    dist = np.genfromtxt(imagesFolder+"calib_dist_webcam.csv")
    print('mtx', mtx.shape)
    print('dist', dist.shape)
    size_of_marker =  0.03 # side lenght of the marker in meter

    cap = cv2.VideoCapture(0)
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    color_list = []
    for x in range(0, 260):
        color_list.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    
    while True:
        time.sleep(1/50)
        ret, frame = cap.read()
        print('frame shape',frame.shape)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters =  aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        
        track(mtx, dist)
