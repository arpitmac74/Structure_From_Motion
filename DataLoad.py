#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:30:43 2020

@author: oshi
"""


import numpy as np 
import matplotlib.pyplot as plt 
import sys
import cv2
import pptk


sys.dont_write_bytecode = True

np.set_printoptions(threshold = np.inf, linewidth = 1000 ,precision=3, suppress=True)

def FindCorrespondence(a,b,path):

    matching_list = []
    #if (1 <= a <= 6):
    
    if 1<= a <=6:
        with open(path + "matching" + str(a) + ".txt") as f:
            line_no = 1
            for line in f:
                if line_no == 1:
                    line_no += 1
                    nfeatures = line[11:15]
                    nfeatures = int(nfeatures)

                else:
                    matching_list.append(line.rstrip('\n'))
    final_list = []            
    for i in range(0, len(matching_list)):
          current_row = matching_list[i]
          splitStr = current_row.split()
          current_row = []
          for j in splitStr:
              current_row.append(float(j))
          final_list.append(np.transpose(current_row))
    rgb_list = []
    image1_points = []
    image2_points = []
    
    for i in range(0, len(final_list)):
        rgb_row = []
        P_1 = []
        P_2 = []
        current_row = final_list[i]
        current_row = current_row[1:len(current_row)]
        
        res = np.where(current_row == b)
        
        P_1.append((current_row[3],current_row[4]))
        rgb_row.append(current_row[0])
        rgb_row.append(current_row[1])
        rgb_row.append(current_row[2])
        
        if (len(res[0]) != 0):
            index = res[0][0]
            P_2.append((current_row[index + 1],current_row[index + 2]))
            
        else:
            P_1.remove((current_row[3],current_row[4]))
            
        
        if (len(P_1) != 0):
            image1_points.append((P_1))
            image2_points.append((P_2))
            rgb_list.append(np.transpose(rgb_row))
        
    image1_points = np.array(image1_points).reshape(-1,2)
    image2_points = np.array(image2_points).reshape(-1,2)
                    
    return image1_points,image2_points,rgb_list






K =  np.array([[568.996140852, 0, 643.21055941], 
    [0, 568.988362396, 477.982801038],
    [0,0,1]])
    
path = ("text/")
pts1,pts2,c=FindCorrespondence(1,2,path)

E, mask_E = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=0.2)
print("Essential Matrix: \n",E)


_, R, t, mask, X = cv2.recoverPose(E, pts1, pts2, K, distanceThresh = 100)

print("Rotation Matrix: \n",R)
print("Translation Vector: \n",t)    

# normalization
X = (X/X[3,:]).T

# getting color values using the mask

color = []
visPts = []
for i in range(len(mask)):
    if mask[i] == 255:
        color.append(c[i])
        visPts.append(X[i])

color = np.array(color)/255
visPts = np.array(visPts)

points = pptk.points(visPts[:,:3])

v = pptk.viewer(points)
v.attributes(pptk.points(color))
v.set(show_grid=False)
v.set(point_size = 0.2)


#img1 = cv2.imread("Data/1.jpg")
#img2 = cv2.imread("Data/2.jpg")
#cv2.imshow("Img1", img1)
#cv2.imshow("Img2", img2)

# Non- linear triangulation
I = np.eye(3)
P1 = K @ I @ np.hstack((I,np.zeros((3,1))))
P2 = K @ R @ np.hstack((I,t))

