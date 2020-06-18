# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:15:18 2020

@author: amris
"""

import numpy as np
import cv2
import glob
import pickle
import pptk
import matplotlib.pyplot as plt

np.set_printoptions(threshold = np.inf, linewidth = 1000 ,precision=3, suppress=True)


def construct(img1, img2):
    img2 = img2[:,:,::-1]
    img1 = img1[:,:,::-1]
    #cv2.imshow("1", img1[200:1399,:,:])
    #cv2.imshow("2", img2[200:1399,:,:])
    
    kp1 , des1 = detector.detectAndCompute(img1,None)
    kp2 , des2 = detector.detectAndCompute(img2,None)
    
    
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors.
    matches = bf.match(des1,des2)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    mainMatch = [m for m in matches if m.distance < 50]
    
    img11 = cv2.drawMatches(img1,kp1,img2,kp2,mainMatch,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#    plt.imshow(img11),plt.show()
    
    cv2.waitKey(1)
    
    x1 = np.float32([kp1[match.queryIdx].pt for match in mainMatch])
    x2 = np.float32([kp2[match.trainIdx].pt for match in mainMatch])
    
    c = np.float32([img1[int(x[1]), int(x[0]),:] for x in x1])
    E, mask_E = cv2.findEssentialMat(x1, x2, K, method=cv2.RANSAC, prob=0.999, threshold=0.2)
    print("Essential Matrix: \n",E)
    
    
    _, R, t, mask, X = cv2.recoverPose(E, x1, x2, K, distanceThresh = 100)
    
    print("Rotation Matrix: \n",R)
    print("Translation Vector: \n",t)    
    
    # normalization
    X = (X/X[3,:]).T
    
    color = []
    visPts = []
    u1 = []
    u2 = []
    for i in range(len(mask)):
        if mask[i] == 255:
            color.append(c[i])
            visPts.append(X[i])
            u1.append(x1[i])
            u2.append(x2[i])
            
    
    color = np.array(color)/255
    visPts = np.array(visPts)
    
    return visPts, color, R, t

def transform(X, R2, t2, R1, t1):
    T2 = np.hstack((R2,t2))
    T2 = np.vstack((T2, np.array([0,0,0,1])))
    
    T1 = np.hstack((R1,t1))
    T1 = np.vstack((T1, np.array([0,0,0,1])))
    
    T1inv = np.linalg.pinv(T1)
    
    T_new = T1 @ T2
    
    X1 = X @ T1inv.T
    
    X1 = np.divide(X1.T, X1[:,3].T).T
    return X1, T_new[:3,:3], T_new[3,:3]

with open('camCalib.pickle', 'rb') as handle:
    K = pickle.load(handle) 
    
# detector
detector = cv2.ORB_create(8000)

filenames = glob.glob('data/*.jfif')
#filenames = glob.glob('duomo/*.JPG')

imgList = []
for fn in filenames:
    imgList.append(cv2.imread(fn, cv2.IMREAD_COLOR))
    
X01, c01, R01, t01 = construct(imgList[0], imgList[1])
X12, c12, R12, t12 = construct(imgList[1], imgList[2])

X03, R03_, t03_ = transform(X12, R12, t12, R01, t01)

X_all = np.vstack((X01, X03))
c_all = np.vstack((c01, c12))

points = pptk.points(X_all[:,:3])

v = pptk.viewer(points)
v.attributes(pptk.points(c_all))
v.set(show_grid=False)
v.set(point_size = 0.002)

cv2.waitKey(0)

cv2.destroyAllWindows()