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

def read_chessboards(images):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0

    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(res[0])>0:
            res2 = cv2.aruco.interpolateCornersCharuco(res[0],res[1],gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        decimator+=1

    imsize = gray.shape
    return allCorners,allIds,imsize

def calibrate_camera(allCorners,allIds,imsize):
    """
    Calibrates the camera using the dected corners.
    """
    print("CAMERA CALIBRATION")

    cameraMatrixInit = np.array([[ 2000.,    0., imsize[0]/2.],
                                 [    0., 2000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors
    print("finished")

if __name__ == '__main__':

    imagesFolder = "/home/ubuntu/opencv/data/"
    mtx = np.genfromtxt(imagesFolder+"calib_mtx_webcam.csv")
    dist = np.genfromtxt(imagesFolder+"calib_dist_webcam.csv")
    
    vid = cv2.VideoCapture(0)
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    color_list = []
    length_of_axis = 0.01
    
    for x in range(0, 260):
        color_list.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    
    while True:
        time.sleep(1/50)
        ret, frame = vid.read()
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        frame = color_frame (frame,markerCorners,markerIds,rejectedCandidates,color_list)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(markerCorners, 0.02, mtx,dist)
        print('rvec', rvec)
        print('tvec', tvec)
        imaxis = frame
        if tvec is not None:
            imaxis = aruco.drawDetectedMarkers(frame.copy(), markerCorners, markerIds)
            for K in range(len(tvec)):
                imaxis = aruco.drawAxis(imaxis, mtx, dist, rvec[K], tvec[K], length_of_axis)

        cv2.imshow('frame', imaxis)

        if markerIds is not None:
            j = 0       
            for maker in markerIds:
                p1 = np.squeeze(markerCorners[j])[0,:]
                p2 = np.squeeze(markerCorners[j])[1,:]
                p3 = np.squeeze(markerCorners[j])[2,:]
                p4 = np.squeeze(markerCorners[j])[3,:]
                j = j + 1
        
