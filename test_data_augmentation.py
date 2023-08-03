from torch.utils.data import Dataset
import os, sys
import numpy as np
from torchvision import transforms
import random
from scipy.ndimage import rotate
import cv2
import nibabel as nib
import matplotlib.pyplot as plt

"""  functions for numpy array image augmentations:
https://medium.com/@schatty/image-augmentation-in-numpy-the-spell-is-simple-but-quite-unbreakable-e1af57bb50fd  """

def translate(img, shift=10, direction='right', roll=False):
    assert direction in ['right', 'left', 'down', 'up'], 'Directions should be down|up|left|right'
    img = img.copy()
    if direction == 'right':
        right_slice = img[:, -shift:].copy()
        img[:, shift:] = img[:, :-shift]
        if roll:
            img[:,:shift] = np.fliplr(right_slice)
    if direction == 'left':
        left_slice = img[:, :shift].copy()
        img[:, :-shift] = img[:, shift:]
        if roll:
            img[:, -shift:] = left_slice
    if direction == 'down':
        down_slice = img[-shift:, :].copy()
        img[shift:, :] = img[:-shift,:]
        if roll:
            img[:shift, :] = down_slice
    if direction == 'up':
        upper_slice = img[:shift, :].copy()
        img[:-shift, :] = img[shift:, :]
        if roll:
            img[-shift:,:] = upper_slice
    return img

def random_crop(img, crop_size=(10, 10)):
    assert crop_size[0] <= img.shape[0] and crop_size[1] <= img.shape[1], "Crop size should be less than image size"
    img = img.copy()
    w, h = img.shape[:2]
    x, y = np.random.randint(h-crop_size[0]), np.random.randint(w-crop_size[1])
    img = img[y:y+crop_size[0], x:x+crop_size[1]]
    img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    return img

def rotate_img(img, angle=10, bg_patch=(5,5)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img

if __name__ == "__main__":

    image = nib.load("/home/kszuyen/project/rwAsPIB020-0001-00001-001034.nii").get_fdata()

    IMAGE_SHAPE = image.shape

    data = np.empty((IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    data = image[:, :, 22]
                
    h = data.shape[0]

    ### data augmentation ###
    # if self.DataAugmentation:
    if random.choice([0, 1]):
        data = np.flipud(data)
    # random crop
    if random.choice([0, 1]):
        l = random.randint(int(h*0.90), h-1)
        data = random_crop(data, crop_size=(l, l))
    # translation
    for directions in (['up', 'down'], ['left', 'right']):
        shift, d = random.randint(0, int(h*0.02)), random.choice(directions)
        if shift:
            data = translate(data, shift, d)
    
    if random.choice([0, 1]):
        angle = random.randint(-5, 5)
        data = rotate_img(data, angle)
    ### data augmentation output shape
    # print(data.shape, ground_truth.shape)
    ###

    plt.imshow(data)
    plt.savefig("test_data_augmentation.png")



