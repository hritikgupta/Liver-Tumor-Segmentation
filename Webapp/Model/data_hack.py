#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 19:54:00 2019

@author: dhanunjaya
"""

import os
import nibabel as nib
import numpy as np

volume = []
segment= []
#data_patient = []

volume_files = os.listdir('./volume/')
segment_files = os.listdir('./segment/')

volume_files = sorted(volume_files)
segment_files = sorted(segment_files)
for i in range(len(volume_files)):
        volume_img = nib.load('./volume/'+volume_files[i])
        tep = np.asarray(volume_img.get_fdata())
        print (tep.shape)
        for j in range(tep.shape[2]):
            temp = tep[:,:,j]
            volume.append(temp)
        
for i in range(len(segment_files)):
        segment_img = nib.load('./segment/'+segment_files[i])
        tep = np.asarray(segment_img.get_fdata())
        print (tep.shape)
        for j in range(tep.shape[2]):
            temp = tep[:,:,j]
            segment.append(temp)
            
print("done")

segment = np.asarray(segment)
volume = np.asarray(volume)
np.save("segment.npy", segment)
np.save("volume.npy", volume)
