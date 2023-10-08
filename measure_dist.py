import cv2
import math
import time
import numpy as np
import cv2, PIL
#from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import random
import heapq
from sklearn.cluster import KMeans
from collections import Counter
#matplotlib nbagg

#aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
#board = aruco.CharucoBoard_create(3, 3, 1, 0.8, aruco_dict)
length_of_axis = 0.01

def color_frame (frame,markerCorners,markerIds,rejectedCandidates,color_list):
    if markerIds is not None:
        i = 0
        for marker in markerIds:
            p1 = np.squeeze(markerCorners[i])[0,:]
            p2 = np.squeeze(markerCorners[i])[1,:]
            p3 = np.squeeze(markerCorners[i])[2,:]
            p4 = np.squeeze(markerCorners[i])[3,:]

            #print(marker)
            frame = cv2.line(frame, (p1[0],p1[1]), (p2[0],p2[1]), color_list[int(marker)], 2)
            frame = cv2.line(frame, (p2[0],p2[1]), (p3[0],p3[1]), color_list[int(marker)], 2)
            frame = cv2.line(frame, (p4[0],p4[1]), (p1[0],p1[1]), color_list[int(marker)], 2)
            frame = cv2.line(frame, (p4[0],p4[1]), (p3[0],p3[1]), color_list[int(marker)], 2)
            frame = cv2.putText(frame,f'{int(marker)}',(p1[0] ,p1[1] ),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),2)
            i = i + 1
        del i, markerCorners, markerIds
    return frame

def find_frame (img,pp,yellow):
    distance_Y = []
    distance_R2Y = []
    Left_R = None
    Right_R = None
    Mid_Y = None
    for Y in yellow:
        P4_distance = 0
        for R in pp:
            P4_distance += np.sqrt((Y[0][0]-R[0][0])**4 + (Y[0][1]-R[0][1])**4)
        distance_Y.append(P4_distance)

    if len(distance_Y) != 0:
        min_value = min(distance_Y)
        min_index = distance_Y.index(min_value)
        selected_Y = yellow[min_index]
        # draw Y 
        box = cv2.boxPoints(selected_Y)
        box = np.int0(box)
        img = cv2.drawContours(img,[box],0,(255,0,0),2)
        Mid_Y = selected_Y
        for R in pp:
            D_R2Y = np.sqrt((selected_Y[0][0]-R[0][0])**2 + (selected_Y[0][1]-R[0][1])**2)
            distance_R2Y.append(D_R2Y)
        #print(distance_R2Y)
        if len(distance_R2Y) > 1:
            two_smallest_indices = [i for i, x in enumerate(distance_R2Y) if x == min(distance_R2Y)]
            distance_R2Y[two_smallest_indices[0]] = max(distance_R2Y) + 1
            two_smallest_indices += [i for i, x in enumerate(distance_R2Y) if x == min(distance_R2Y)]
            # draw pp
            
            box = cv2.boxPoints(pp[two_smallest_indices[0]])
            box = np.int0(box)
            img = cv2.drawContours(img,[box],0,(255,255,0),2)
            box = cv2.boxPoints(pp[two_smallest_indices[1]])
            box = np.int0(box)
            img = cv2.drawContours(img,[box],0,(255,255,0),2)
            Left_R = pp[two_smallest_indices[0]]
            Right_R = pp[two_smallest_indices[1]]
    return img,Left_R,Right_R,Mid_Y


if __name__ == '__main__':

    imagesFolder = "/home/fyp/opencv_script/data/"
    mtx = np.genfromtxt(imagesFolder+"calib_mtx_webcam.csv")
    dist = np.genfromtxt(imagesFolder+"calib_dist_webcam.csv")
    
    vid = cv2.VideoCapture(0)
    # 640*480  640*360 960*540 320*240 1600*896
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # 1280
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # 720
    #vid.set(cv2.CAP_PROP_FPS, 40)
    #dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    #parameters = cv2.aruco.DetectorParameters_create()
    color_list = []
    data = 0
    yellow = []
    pp = []
    yellow_last = []
    pp_last = []
    #cv2.setNumThreads(6)

    counter = 0

    while True:

        

        #print('------------START NEW ITER-------------')

        ret, img = vid.read()
        ts = time.time()
        t0 = time.time()
        #img = cv2.fastNlMeansDenoisingColored(img, None, 2, 2, 3, 7)

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#normal color
        lower_range_1 = np.array([140, 130, 0])
        upper_range_1 = np.array([175, 255, 255])
        mask_1 = cv2.inRange(img_hsv, lower_range_1, upper_range_1)
        #cv2.imshow('mask_1',mask_1)
        lower_range_2 = np.array([140, 130, 0])#backlighting(beiguang)
        upper_range_2 = np.array([178, 255, 160])
        mask_2 = cv2.inRange(img_hsv, lower_range_2, upper_range_2)
        #cv2.imshow('mask_2',mask_2)
        lower_range_3 = np.array([148, 20, 230])#frontlighting(fanguang)
        upper_range_3 = np.array([160, 130, 255])
        mask_3 = cv2.inRange(img_hsv, lower_range_3, upper_range_3)
        #cv2.imshow('mask_3',mask_3)
        # yellow 
        lower_range_y1 = np.array([16, 130, 0])
        upper_range_y1 = np.array([25, 255, 255])
        mask_y1 = cv2.inRange(img_hsv, lower_range_y1, upper_range_y1)

        lower_range_y2 = np.array([8, 150, 0])
        upper_range_y2 = np.array([16, 255, 160])
        mask_y2 = cv2.inRange(img_hsv, lower_range_y2, upper_range_y2)

        lower_range_y3 = np.array([16, 110, 220])
        upper_range_y3 = np.array([25, 180, 255])
        mask_y3 = cv2.inRange(img_hsv, lower_range_y3, upper_range_y3)

        mask = cv2.bitwise_or(mask_1, mask_2)
        mask = cv2.bitwise_or(mask, mask_3)
        mask_y = cv2.bitwise_or(mask_y1, mask_y2)
        mask_y = cv2.bitwise_or(mask_y, mask_y3)
        cv2.imshow('origin', img)
        cv2.imshow('color Red HSV filter', mask)
        cv2.imshow('color Yellow HSV filter', mask_y)
        #
        # pre process :
        # erosion 
        kernel = np.ones((3,3),dtype = np.uint8)
        fat_kernel = np.ones((5,5),dtype = np.uint8)
        mask = cv2.erode(mask,fat_kernel,iterations = 1)
        mask_y = cv2.erode(mask_y,fat_kernel,iterations = 1)
        # dilate
        mask = cv2.dilate(mask,fat_kernel,iterations = 1)
        mask_y = cv2.dilate(mask_y,fat_kernel,iterations = 1)
        # ================================================================
        #print('kernel 5*5')
        cv2.imshow('color Red HSV filter with erode and dilate', mask)
        cv2.imshow('color Yellow HSV filter with erode and dilate', mask_y)
        # mask HSV thresh
        #thresh = cv2.bitwise_and(gray, gray, mask=mask)
        #cv2.imshow('thresh', thresh)
        
        
        # Find the contours of the objects
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contoursy, hierarchyy = cv2.findContours(mask_y, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Initialize the mask image
        #mask = np.zeros_like(gray)
        #cv2.imshow('mask_y', mask_y)
        #cv2.imshow('mask', mask)
        yellow = []
        pp = []
        for contour in contours:
            # Compute the aspect ratio of the contour
            rect = cv2.minAreaRect(contour)
            ratio_rect = rect[1][0]/rect[1][1]
            H = rect[1][0]
            W = rect[1][1]

            if ratio_rect < 1:
                ratio_rect = 1/ratio_rect
            
            if ratio_rect > 5 and ratio_rect < 40 and H*W > 500:
                #print('----------------------------')
                #print('rect',rect)
                pp.append(rect)
        
        for contour in contoursy:
            # Compute the aspect ratio of the contour
            rect = cv2.minAreaRect(contour)
            ratio_rect = rect[1][0]/rect[1][1]
            if ratio_rect < 1:
                ratio_rect = 1/ratio_rect
            
            if ratio_rect > 2 and ratio_rect < 20:
                yellow.append(rect)

        result = np.zeros(img.shape[:2], dtype=np.uint8)
        mask_attention = np.zeros(img.shape[:2], dtype=np.uint8)
        
        img,R1,R2,Ym = find_frame (img,pp,yellow)
        if (R1 is not None) & (R2 is not None) & (Ym is not None):
            box1 = np.int0(cv2.boxPoints(R1))
            box2 = np.int0(cv2.boxPoints(R2))
            box3 = np.int0(cv2.boxPoints(Ym))
            all_box_points = np.concatenate((box1, box2, box3), axis=0)
            
            num_clusters = 4

            kmeans = KMeans(n_clusters=num_clusters, n_init=10)
            kmeans.fit(all_box_points)
            labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_
            #print('cluster_centers',cluster_centers)
            #print('labels',labels)
            # find the 4 real corners
            hull = cv2.convexHull(all_box_points)
            hull_points = hull[:, 0]
            
            label_counter = Counter(labels)
            most_common_labels = [label for label, _ in label_counter.most_common(2)]
            most_common_centers = [cluster_centers[label] for label in most_common_labels]
            # 找到频率最低的两个标签
            least_common_labels = [label for label, _ in label_counter.most_common()[:-3:-1]]

            # 找到对应最低频率标签的聚类中心
            least_common_centers = [cluster_centers[label] for label in least_common_labels]

            # save the orgain graph 
            origin = np.copy(img)

            for point in least_common_centers:
                cv2.circle(img, (int(point[0]),int(point[1])), 10, (0, 100, 255), 12)
            for point in most_common_centers:
                cv2.circle(img, (int(point[0]),int(point[1])), 10, (0, 0, 255), 12)

            # yellow center
            x1,y1 = Ym[0][0],Ym[0][1]
            # red center
            x2,y2 = R1[0][0]/2+R2[0][0]/2 , R1[0][1]/2+R2[0][1]/2
            delta_x = x2 - x1
            delta_y = y2 - y1
            angle_radians = -math.atan2(delta_y, delta_x)
            angle_degrees = math.degrees(angle_radians)
            #print("向量的方向角度（弧度）:", angle_radians)
            #print("向量的方向角度（角度）:", angle_degrees)

            X0,Y0 = 0,0
            X1,Y1 = 0,0
            X2,Y2 = 0,0
            X3,Y3 = 0,0

            if (angle_degrees > 0 and angle_degrees < 45) : 
                if least_common_centers[0][1] > least_common_centers[1][1]:
                    X0,Y0 = least_common_centers[0]
                    X1,Y1 = least_common_centers[1]
                else :
                    X0,Y0 = least_common_centers[1]
                    X1,Y1 = least_common_centers[0]
            elif (angle_degrees > 45 and angle_degrees < 135):
                if least_common_centers[0][0] > least_common_centers[1][0]:
                    X0,Y0 = least_common_centers[0]
                    X1,Y1 = least_common_centers[1]
                else :
                    X0,Y0 = least_common_centers[1]
                    X1,Y1 = least_common_centers[0]
            elif angle_degrees > 135:
                if least_common_centers[0][1] < least_common_centers[1][1]:
                    X0,Y0 = least_common_centers[0]
                    X1,Y1 = least_common_centers[1]
                else :
                    X0,Y0 = least_common_centers[1]
                    X1,Y1 = least_common_centers[0]
            

            if (angle_degrees > 0 and angle_degrees < 45) : 
                if most_common_centers[0][1] > most_common_centers[1][1]:
                    X2,Y2 = most_common_centers[0]
                    X3,Y3 = most_common_centers[1]
                else :
                    X2,Y2 = most_common_centers[1]
                    X3,Y3 = most_common_centers[0]
            elif (angle_degrees > 45 and angle_degrees < 135):
                if most_common_centers[0][0] > most_common_centers[1][0]:
                    X2,Y2 = most_common_centers[0]
                    X3,Y3 = most_common_centers[1]
                else :
                    X2,Y2 = most_common_centers[1]
                    X3,Y3 = most_common_centers[0]
            elif angle_degrees > 135:
                if most_common_centers[0][1] < most_common_centers[1][1]:
                    X2,Y2 = most_common_centers[0]
                    X3,Y3 = most_common_centers[1]
                else :
                    X2,Y2 = most_common_centers[1]
                    X3,Y3 = most_common_centers[0]
            

            
            cv2.putText(img, '0', (int(X0),int(Y0)), cv2.FONT_HERSHEY_SIMPLEX, 2, (222, 222, 22), 7)
            cv2.putText(img, '1', (int(X1),int(Y1)), cv2.FONT_HERSHEY_SIMPLEX, 2, (222, 222, 22), 7)
            cv2.putText(img, '2', (int(X2),int(Y2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (222, 222, 22), 7)
            cv2.putText(img, '3', (int(X3),int(Y3)), cv2.FONT_HERSHEY_SIMPLEX, 2, (222, 222, 22), 7)
            # calculate 3D distance :
            pos_array = np.array([[X0,Y0],[X1,Y1],[X2,Y2],[X3,Y3]])
            length = 0.1
            width = 0.15
            world_points = np.array([
                [length, 0, 0],       # 右上
                [0, 0, 0],           # 左上
                [length, width, 0],  # 右下
                [0, width, 0]        # 左下
            ], dtype=np.float32)


            success, rvec, tvec = cv2.solvePnP(world_points, pos_array, mtx, dist)
            if success : 
                # 计算四个点的中心到相机的距离
                distance = np.linalg.norm(tvec)
                print("Distance from center to camera:", distance)
                rot_matrix, _ = cv2.Rodrigues(rvec)
                # 计算欧拉角
                sy = np.sqrt(rot_matrix[0, 0] * rot_matrix[0, 0] + rot_matrix[1, 0] * rot_matrix[1, 0])
                singular = sy < 1e-6
                
                if not singular:
                    x_angle = np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2])
                    y_angle = np.arctan2(-rot_matrix[2, 0], sy)
                    z_angle = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
                else:
                    x_angle = np.arctan2(-rot_matrix[1, 2], rot_matrix[1, 1])
                    y_angle = np.arctan2(-rot_matrix[2, 0], sy)
                    z_angle = 0
                
                # 将欧拉角从弧度转换为度
                x_angle = np.degrees(x_angle)
                y_angle = np.degrees(y_angle)
                z_angle = np.degrees(z_angle)
                
                print("X Angle:", x_angle)
                print("Y Angle:", y_angle)
                print("Z Angle:", z_angle)


            if counter % 20 == 0:
                np.savetxt(f"test2/save_pos_{int(counter/20)}.txt", pos_array)
                cv2.imwrite(f'test2/frame_detection/image_{int(counter/20)}.jpg', img)
                cv2.imwrite(f'test2/origin/O_image_{int(counter/20)}.jpg', origin)
            mask_attention = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask_attention, [hull_points], (255, 255, 255))
            result = cv2.bitwise_and(img, img, mask=mask_attention)
        
        orb = cv2.ORB_create(nfeatures=1200)
        keypoints, descriptors = orb.detectAndCompute(img, mask_attention)
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow('img_with_keypoints',img_with_keypoints)

        cv2.imshow('ONLY keep frame part', result)
        cv2.imshow('img', img)
        #print('time used all', time.time() - ts)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        #time.sleep(0.1)
        counter += 1

