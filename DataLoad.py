#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:30:43 2020

@author: oshi
"""


import numpy as np 
import matplotlib.pyplot as plt 
import sys


sys.dont_write_bytecode = True



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
    x_list = []
    y_list = []
    binary_list = []
    
    for i in range(0, len(final_list)):
        rgb_row = []
        x_row = []
        y_row = []
        current_row = final_list[i]
        current_row = current_row[1:len(current_row)]
        
        res = np.where(current_row == b)
        
        x_row.append(current_row[3])
        y_row.append(current_row[4])
        rgb_row.append(current_row[0])
        rgb_row.append(current_row[1])
        rgb_row.append(current_row[2])
        
        if (len(res[0]) != 0):
            index = res[0][0]
            x_row.append(current_row[index + 1])
            y_row.append(current_row[index + 2])
            
        else:
            x_row.append(0)
            y_row.append(0)
        
        if (len(x_row) != 0):
            x_list.append(np.transpose(x_row))
            y_list.append(np.transpose(y_row))
            rgb_list.append(np.transpose(rgb_row))
        
    
    
                    
    return np.array(x_list), np.array(y_list), np.array(rgb_list)






    
path = ("/home/oshi/SLAM/Structure_from_motion/text/")
a=FindCorrespondence(1,2,path)
print(a)






    