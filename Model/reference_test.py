from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers 
from keras.layers.normalization import BatchNormalization as bn
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import model_from_json
import keras.backend as K
import tensorflow as tf

import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from keras.models import load_model


img_path = glob("/media/biometric/Data2/FM/Hackathon/media/nas/01_Datasets/CT/LITS/TrainingBatch1/volume/volume-*.nii")
mask_path = glob("/media/biometric/Data2/FM/Hackathon/media/nas/01_Datasets/CT/LITS/TrainingBatch1/segment/segmentation-*.nii")
test_path = glob("/media/biometric/Data2/FM/Hackathon/media/nas/01_Datasets/CT/LITS/Testing/test-volume-*.nii")
input_image_from_user = glob("/media/biometric/Data2/FM/Hackathon/test-volume-*.nii")

def weighted_binary_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 10e-8, 1.-10e-8)
    loss = - (y_true * K.log(y_pred) * 0.90 + (1 - y_true) * K.log(1 - y_pred) * 0.10)
    
    return K.mean(loss)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


patch_ratio = []

for i in range(16 + 1):
  patch_ratio.append(32 * i)

input_shape = [64, 64, 1]
dropout_rate = 0.3
l2_lambda = 0.0002

def slice_to_patch(slice, patch_ratio):
  
  slice[slice == 1] = 0
  slice[slice == 2] = 1
  
  patch_list = []
  
  for x_bin in range(2, len(patch_ratio)):
    for y_bin in range(2, len(patch_ratio)):
      patch = slice[patch_ratio[x_bin-2] : patch_ratio[x_bin], patch_ratio[y_bin - 2] : patch_ratio[y_bin]]
      patch = patch.reshape(patch.shape + (1,))
      patch_list.append(patch)
  
  return np.array(patch_list)

def patch_to_slice(patch, patch_ratio, input_shape, conf_threshold):
  
  slice = np.zeros((512, 512, 1))
  row_idx = 0
  col_idx = 0
  
  for i in range(len(patch)):
    
    slice[patch_ratio[row_idx]:patch_ratio[row_idx + 2], patch_ratio[col_idx]:patch_ratio[col_idx + 2]][patch[i] > conf_threshold] = 1
    
    col_idx += 1
    
    if i != 0 and (i+1) % 15 == 0:
      row_idx += 1
      col_idx = 0
  
  return slice

json_file = open('/media/biometric/Data2/FM/Hackathon/model_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/media/biometric/Data2/FM/Hackathon/model_weights.h5")
print("Loaded model from disk")

img_ex_obj = nib.load('/media/biometric/Data2/FM/Hackathon/input_volume.nii')
img_ex = nib.load('/media/biometric/Data2/FM/Hackathon/input_volume.nii').get_data()
slices = []
print (img_ex.shape[2])
fig, (ax1,ax3) = plt.subplots(1, 2, figsize = ((15, 15)))

all_with_lesions = []

for i in range(img_ex.shape[2]):
    patch_ex = slice_to_patch(img_ex[:, :, i], patch_ratio)
    prediction = loaded_model.predict(patch_ex)
    prediction_mask = patch_to_slice(prediction, patch_ratio, input_shape, conf_threshold = 0.97)
    
    # superimpose = prediction_mask + img_ex[:, :, i]
    slices.append(prediction_mask)
    
    if 1 in prediction_mask:  
      all_with_lesions.append((i, (img_ex[:, :, i], prediction_mask)))

mid = int(len(all_with_lesions)/2)
m=np.squeeze(all_with_lesions[mid][1][1])
n=all_with_lesions[mid][1][0]
m[m==1]=n.max()
n[n<0]=0
# print(all_with_lesions[mid][1][0].max())
# print(m)
# print(all_with_lesions[mid][1][0])
# print(m.max())
# print(m.min())
# print(m.shape)
z=m+n
# print(z)
# ax1.imshow(np.rot90(all_with_lesions[mid][1][0], 3), cmap = 'bone')
ax1.imshow(all_with_lesions[mid][1][0], cmap = 'bone')
ax1.set_title("CT Scan", fontsize = "x-large")
ax1.grid(False)
ax3.imshow(np.squeeze(z), cmap = 'bone')
ax3.set_title("Predicted Mask", fontsize = "x-large")
ax3.grid(False)
# plt.show()
plt.savefig('/media/biometric/Data2/FM/Hackathon/mask.png')

np_slices = np.asarray(slices)
np_slices = np.swapaxes(np_slices, 0, 2)
new_img = nib.Nifti1Image(np_slices, affine=img_ex_obj.affine)
nib.save(new_img, "/media/biometric/Data2/FM/Hackathon/output.nii")




















# for i in range(img_ex.shape[2]):
#     patch_ex = slice_to_patch(img_ex[:, :, i], patch_ratio)
#     prediction = loaded_model.predict(patch_ex)
#     prediction_mask = patch_to_slice(prediction, patch_ratio, input_shape, conf_threshold = 0.97)
#     # print (prediction_mask)
#     # _, count = np.unique(prediction_mask[:, :, i], return_counts=True)
#     if 1 in prediction_mask:
#         cout = np.where(prediction == 1)
#         if cout[0].size != 0:
#             print ("yes", cout)
#             fig, (ax1,ax3) = plt.subplots(1, 2, figsize = ((15, 15)))
                
            
#             ax1.imshow(np.rot90(img_ex[:, :, i], 3), cmap = 'bone')
#             ax1.set_title("Image", fontsize = "x-large")
#             ax1.grid(False)
#             ax3.imshow(np.rot90(prediction_mask.reshape((512, 512)), 3), cmap = 'bone')
#             ax3.set_title("Mask (Pred)", fontsize = "x-large")
#             ax3.grid(False)
#             plt.show()
#             # print (prediction_mask, mask_ex)
#             # print (prediction.shape)
















#     _, count = np.unique(mask_ex[:, :, i], return_counts=True)
    
#     if len(count) > 1 and count[1] > 300:
        
#         patch_ex = slice_to_patch(img_ex[:, :, i], patch_ratio)
#         prediction = loaded_model.predict(patch_ex)
#         prediction_mask = patch_to_slice(prediction, patch_ratio, input_shape, conf_threshold = 0.97)
        
#         print (prediction_mask, mask_ex)
        # print (prediction.shape)
        # fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize = ((15, 15)))
        
        
        # ax1.imshow(np.rot90(img_ex[:, :, i], 3), cmap = 'bone')
        # ax1.set_title("Image", fontsize = "x-large")
        # ax1.grid(False)
        # ax2.imshow(np.rot90(mask_ex[:, :, i], 3), cmap = 'bone')
        # ax2.set_title("Mask (True)", fontsize = "x-large")
        # ax2.grid(False)
        # ax3.imshow(np.rot90(prediction_mask.reshape((512, 512)), 3), cmap = 'bone')
        # ax3.set_title("Mask (Pred)", fontsize = "x-large")
        # ax3.grid(False)
        # plt.show()
