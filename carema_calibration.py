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
import numpy as np
import cv2, PIL, os


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


    imagesFolder = "/home/shiqi/opencv/data/"
    images = [imagesFolder + f for f in os.listdir(imagesFolder) if f.startswith("image_")]
    allCorners,allIds,imsize=read_chessboards(images)
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners,allIds,imsize)
    print('ret',ret)
    print('mtx',mtx)
    print('dist',dist)
    np.savetxt(imagesFolder+"calib_mtx_webcam.csv", mtx)
    np.savetxt(imagesFolder+"calib_dist_webcam.csv", dist)
    print('DONE DONE ----------- ----------- ------------ ')
