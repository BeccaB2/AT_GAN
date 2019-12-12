# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 21:12:14 2019

@author: rm2-bradburn
"""
def read_image(src):
    img = cv2.imread(src)
if img is None:
 raise FileNotFoundError
     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 return img
