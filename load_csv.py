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



if __name__ == '__main__':

    imagesFolder = "/home/shiqi/opencv/data/"
    mtx = np.genfromtxt(imagesFolder+"calib_mtx_webcam.csv")
    dist = np.genfromtxt(imagesFolder+"calib_dist_webcam.csv")
    print(mtx)
    print(dist)
