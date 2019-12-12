# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 21:09:25 2019

@author: rm2-bradburn
"""
dog_breed_dict = {}
for annotation in os.listdir(image_ann_dir):
    annotations = annotation.split('-')
    dog_breed_dict[annotations[0]] = annotations[1]