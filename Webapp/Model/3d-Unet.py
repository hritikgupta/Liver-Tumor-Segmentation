#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 01:50:56 2019

@author: dhanunjaya
"""


from keras.layers import ZeroPadding3D, Input, Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, Activation, Dropout, Conv3DTranspose, concatenate, add 
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

n_filters=16
dropout=0.5
batchnorm=True

def conv3d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv3D(filters=n_filters, kernel_size=(kernel_size, kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv3D(filters=n_filters, kernel_size=(kernel_size, kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

input_img = Input((512, 512, 72, 1), name='img')
#zpad1 = ZeroPadding3D(padding=(0, 0, 1))(input_img)
c1 = conv3d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
p1 = MaxPooling3D((2, 2, 2)) (c1)
p1 = Dropout(dropout*0.5)(p1)

c2 = conv3d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
p2 = MaxPooling3D((2, 2, 2)) (c2)
p2 = Dropout(dropout)(p2)

c3 = conv3d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
p3 = MaxPooling3D((2, 2, 2)) (c3)
p3 = Dropout(dropout)(p3)

#c4 = conv3d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
#p4 = MaxPooling3D(pool_size=(2, 2, 2)) (c4)
#p4 = Dropout(dropout)(p4)
    
c5 = conv3d_block(p3, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
# expansive path
#u6 = Conv3DTranspose(n_filters*8, (3, 3, 3), strides=(2, 2, 2), padding='same') (c5)
#u6 = concatenate([u6, c4])
#u6 = Dropout(dropout)(u6)
#c6 = conv3d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

u7 = Conv3DTranspose(n_filters*4, (3, 3, 3), strides=(2, 2, 2), padding='same') (c5)
u7 = concatenate([u7, c3])
u7 = Dropout(dropout)(u7)
c7 = conv3d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

u8 = Conv3DTranspose(n_filters*2, (3, 3, 3), strides=(2, 2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
u8 = Dropout(dropout)(u8)
c8 = conv3d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

u9 = Conv3DTranspose(n_filters*1, (3, 3, 3), strides=(2, 2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=4)
u9 = Dropout(dropout)(u9)
c9 = conv3d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
outputs = Conv3D(1, (1, 1, 1), activation='sigmoid') (c9)
model = Model(inputs=[input_img], outputs=[outputs])

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()