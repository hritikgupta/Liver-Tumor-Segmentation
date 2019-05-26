from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers 
from keras.layers.normalization import BatchNormalization as bn
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import model_from_json
# import tensorflow as tf

import nibabel as nib
import matplpotlib.pyplot as plt
from glob import glob

vol_path = glob('./media/nas/01_Datasets/CT/LITS/TrainingBatch1/volume-*.nii')
mask_path = glob('./media/nas/01_Datasets/CT/LITS/TrainingBatch1/mask-*.nii')

# def u_net(input_shape, dropout_rate, l2_lambda):
#   # Encoder
#   input = Input(shape = input_shape, name = "input")
#   conv1_1 = Conv2D(32, (3, 3), padding = "same", activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), name = "conv1_1")(input)
#   conv1_1 = bn(name = "conv1_1_bn")(conv1_1)
#   conv1_2 = Conv2D(32, (3, 3), padding = "same", activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), name = "conv1_2")(conv1_1)
#   conv1_2 = bn(name = "conv1_2_bn")(conv1_2)
#   pool1 = MaxPooling2D(name = "pool1")(conv1_2)
#   drop1 = Dropout(dropout_rate)(pool1)
  
#   conv2_1 = Conv2D(64, (3, 3), padding = "same", activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), name = "conv2_1")(pool1)
#   conv2_1 = bn(name = "conv2_1_bn")(conv2_1)
#   conv2_2 = Conv2D(64, (3, 3), padding = "same", activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), name = "conv2_2")(conv2_1)
#   conv2_2 = bn(name = "conv2_2_bn")(conv2_2)
#   pool2 = MaxPooling2D(name = "pool2")(conv2_2)
#   drop2 = Dropout(dropout_rate)(pool2)
  
#   conv3_1 = Conv2D(128, (3, 3), padding = "same", activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), name = "conv3_1")(pool2)
#   conv3_1 = bn(name = "conv3_1_bn")(conv3_1)
#   conv3_2 = Conv2D(128, (3, 3), padding = "same", activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), name = "conv3_2")(conv3_1)
#   conv3_2 = bn(name = "conv3_2_bn")(conv3_2)
#   pool3 = MaxPooling2D(name = "pool3")(conv3_2)
#   drop3 = Dropout(dropout_rate)(pool3)  

#   conv4_1 = Conv2D(256, (3, 3), padding = "same", activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), name = "conv4_1")(pool3)
#   conv4_1 = bn(name = "conv4_1_bn")(conv4_1)
#   conv4_2 = Conv2D(256, (3, 3), padding = "same", activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), name = "conv4_2")(conv4_1)
#   conv4_2 = bn(name = "conv4_2_bn")(conv4_2)
#   pool4 = MaxPooling2D(name = "pool4")(conv4_2)
#   drop4 = Dropout(dropout_rate)(pool4)  

#   conv5_1 = Conv2D(512, (3, 3), padding = "same", activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), name = "conv5_1")(pool4)
#   conv5_1 = bn(name = "conv5_1_bn")(conv5_1)
#   conv5_2 = Conv2D(512, (3, 3), padding = "same", activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), name = "conv5_2")(conv5_1)
#   conv5_2 = bn(name = "conv5_2_bn")(conv5_2)
  
#   # Decoder
#   upconv6 = Conv2DTranspose(256,(2, 2), strides=(2, 2), padding='same')(conv5_2)
#   upconv6 = Dropout(dropout_rate)(upconv6)
#   concat6 = concatenate([conv4_2, upconv6], name = "concat6")
#   conv6_1 = Conv2D(256, (3, 3), padding = "same", kernel_regularizer=regularizers.l2(l2_lambda), name = "conv6_1")(concat6)
#   conv6_1 = bn(name = "conv6_1_bn")(conv6_1)
#   conv6_2 = Conv2D(256, (3, 3), padding = "same", kernel_regularizer=regularizers.l2(l2_lambda), name = "conv6_2")(conv6_1)
#   conv6_2 = bn(name = "conv6_2_bn")(conv6_2)
    
#   upconv7 = Conv2DTranspose(128,(2, 2), strides=(2, 2), padding='same')(conv6_2)
#   upconv7 = Dropout(dropout_rate)(upconv7)
#   concat7 = concatenate([conv3_2, upconv7], name = "concat7")
#   conv7_1 = Conv2D(128, (3, 3), padding = "same", kernel_regularizer=regularizers.l2(l2_lambda), name = "conv7_1")(concat7)
#   conv7_1 = bn(name = "conv7_1_bn")(conv7_1)
#   conv7_2 = Conv2D(128, (3, 3), padding = "same", kernel_regularizer=regularizers.l2(l2_lambda), name = "conv7_2")(conv7_1)
#   conv7_2 = bn(name = "conv7_2_bn")(conv7_2)

#   upconv8 = Conv2DTranspose(64,(2, 2), strides=(2, 2), padding='same')(conv7_2)
#   upconv8 = Dropout(dropout_rate)(upconv8)
#   concat8 = concatenate([conv2_2, upconv8], name = "concat8")
#   conv8_1 = Conv2D(64, (3, 3), padding = "same", kernel_regularizer=regularizers.l2(l2_lambda), name = "conv8_1")(concat8)
#   conv8_1 = bn(name = "conv8_1_bn")(conv8_1)
#   conv8_2 = Conv2D(64, (3, 3), padding = "same", kernel_regularizer=regularizers.l2(l2_lambda), name = "conv8_2")(conv8_1)
#   conv8_2 = bn(name = "conv8_2_bn")(conv8_2)

#   upconv9 = Conv2DTranspose(32,(2, 2), strides=(2, 2), padding='same')(conv8_2)
#   upconv9 = Dropout(dropout_rate)(upconv9)
#   concat9 = concatenate([conv1_2, upconv9], name = "concat9")
#   conv9_1 = Conv2D(32, (3, 3), padding = "same", kernel_regularizer=regularizers.l2(l2_lambda), name = "conv9_1")(concat9)
#   conv9_1 = bn(name = "conv9_1_bn")(conv9_1)
#   conv9_2 = Conv2D(32, (3, 3), padding = "same", kernel_regularizer=regularizers.l2(l2_lambda), name = "conv9_2")(conv9_1)
#   conv9_2 = bn(name = "conv9_2_bn")(conv9_2)
#   dropout = Dropout(dropout_rate)(conv9_2)
  
#   conv10 = Conv2D(1, (1, 1), padding = "same", activation = 'sigmoid', name = "conv10")(dropout)
  
#   model = Model(input, conv10)  
#   return model

# input_shape = [512, 512, 1]
# dropout_rate = 0.3
# l2_lambda = 0.0002
# model = u_net(input_shape, dropout_rate, l2_lambda)
# model.summary()


# # def weighted_binary_crossentropy(y_true, y_pred):
# #     y_pred = tf.clip_by_value(y_pred, 10e-8, 1.-10e-8)
# #     loss = - (y_true * K.log(y_pred) * 0.90 + (1 - y_true) * K.log(1 - y_pred) * 0.10)
# #     return K.mean(loss)

# # # returns the similarity between 2 sets
# # def dice_coeff(pred, target):
# #     smooth = 1.
# #     num = pred.size(0)
# #     m1 = pred.view(num, -1)  # Flatten
# #     m2 = target.view(num, -1)  # Flatten
# #     intersection = (m1 * m2).sum()
# #     return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

# for i in range(len(vol_path)):
#     img_3D = nib.load(vol_path[i]).get_data()
#     mask_3D = nib.load(mask_path[i]).get_data()
