#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import torch
import cv2
import random
import imageio

import numpy as np
import matplotlib.pyplot as plt
import pretrained_microscopy_models as pmm
import segmentation_models_pytorch as smp
import albumentations as albu

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import time

# set random seeds for repeatability
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# model parameters
architecture = 'UnetPlusPlus'
encoder = 'resnet50' #resnext101_32x8d
pretrained_weights = 'micronet' #'micronet' - giving error
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#print (device)

# Create the Unet model with a resnet backbone that is pre-trained on micronet
model = pmm.segmentation_training.create_segmentation_model(
    architecture=architecture,
    encoder = encoder,
    encoder_weights=pretrained_weights, # use encoder pre-trained on micronet
    classes=3 # secondary precipitates, tertiary precipitates, matrix
    )
    

DATA_DIR = 'AlSi10Mg-Training'

x_train_dir = os.path.join(DATA_DIR, 'Train')
y_train_dir = os.path.join(DATA_DIR, 'Train_annot')

x_valid_dir = os.path.join(DATA_DIR, 'Val')
y_valid_dir = os.path.join(DATA_DIR, 'Val_annot')
'''
#x_test_dir = os.path.join(DATA_DIR, 'test')
#y_test_dir = os.path.join(DATA_DIR, 'test_annot')
'''
def get_training_augmentation():
    train_transform = [
        albu.Flip(p=0.75),
        albu.RandomRotate90(p=1),       
        albu.GaussNoise(p=0.5),
        
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1, limit=0.25),
                albu.RandomGamma(p=1),
            ],
            p=0.50,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                #albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.50,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1, limit=0.3),
                albu.HueSaturationValue(p=1),
            ],
            p=0.50,
        ),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    # This is turned off for this dataset
    test_transform = [
        #albu.Resize(1000,1000)#height,width)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

print ("Check 1")
def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# how the images will be normalized. Use imagenet statistics even on micronet pre-training
preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, 'imagenet') 

# pixel values of the annotations for each mask.
class_values = {'matrix': [0,0,255],
               'secondary': [255,255,255],
               'tertiary' : [255,0,0]
}

training_dataset = pmm.io.Dataset(
    images=x_train_dir,
    masks=y_train_dir,
    class_values=class_values,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn)
)

validation_dataset = pmm.io.Dataset(
    images=x_valid_dir,
    masks=y_valid_dir,
    class_values=class_values,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn)
)

'''
test_dataset = pmm.io.Dataset(
    images=x_test_dir,
    masks=y_test_dir,
    class_values=class_values,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn)
)

# validation data

visualize_dataset = pmm.io.Dataset(
    images=x_valid_dir,
    masks=y_valid_dir,
    class_values=class_values,
    augmentation=get_validation_augmentation(),
    #preprocessing=get_preprocessing(preprocessing_fn)
)
print ("Check 2")
for im, mask in visualize_dataset:
    pmm.util.visualize(
        image=im,
        matrix_mask=mask[...,0].squeeze(),
        secondary_mask=mask[...,1].squeeze(),
        tertiary=mask[...,2].squeeze(),
    )
'''
# augmented training data

augmented_dataset = pmm.io.Dataset(
    images=x_train_dir,
    masks=y_train_dir,
    class_values=class_values,
    augmentation=get_training_augmentation(),
    #preprocessing=get_preprocessing(preprocessing_fn)
)
'''
for im, mask in augmented_dataset:
    pmm.util.visualize(
        image=im,
        matrix_mask=mask[...,0].squeeze(),
        secondary_mask=mask[...,1].squeeze(),
        tertiary=mask[...,2].squeeze(),
    )
    
'''

#Originally Training the model or finetuning. 
'''
#model_path = 'AlSi10Mg_model.tar'
state = pmm.segmentation_training.train_segmentation_model(
    model=model,
    architecture=architecture,
    encoder=encoder,
    train_dataset=training_dataset,
    validation_dataset=validation_dataset,
    class_values=class_values,
    patience=30,
    lr=1e-5,
    batch_size=6,
    val_batch_size=6,
    device = device,
    save_folder='models',
    save_name='AlSi10Mg_model.tar'
)
'''
'''

plt.plot(state['train_loss'], label='train_loss')
plt.plot(state['valid_loss'], label='valid_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# drop the learning rate and keep training to see if we can squeeze a little more out.
model_path = Path('models', 'UnetPlusPlus_resnet50_high_lr.pth.tar')
state = pmm.segmentation_training.train_segmentation_model(
    model=str(model_path),
    architecture=architecture,
    encoder=encoder,
    patience=30,
    lr=1e-5,
    batch_size=6,
    val_batch_size=6,
    train_dataset=training_dataset,
    validation_dataset=validation_dataset,
    class_values=class_values,
    save_folder='models',
    save_name='UnetPlusPlus_resnet50_low_lr.pth.tar'
)


plt.plot(state['train_loss'], label='train_loss')
plt.plot(state['valid_loss'], label='valid_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# load best model
best_model_path = Path('models', 'model_best.pth.tar')
state = torch.load(best_model_path) 
print ("Check")
best_model = pmm.segmentation_training.create_segmentation_model(
    architecture=architecture,
    encoder = encoder,
    encoder_weights=pretrained_weights, # use encoder pre-trained on micronet
    classes=3 # secondary precipitates, tertiary precipitates, matrix
    )
best_model.load_state_dict(pmm.util.remove_module_from_state_dict(state['state_dict']))
print ("Check2")

# create test dataset
test_dataset = pmm.io.Dataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    class_values=class_values,
)

test_dataloader = DataLoader(test_dataset)
print ("Check 3")
# test dataset without transformations for image visualization
test_dataset_vis = pmm.io.Dataset(
    x_test_dir, y_test_dir, 
    class_values=class_values,
)

# evaluate model on test set
loss = pmm.losses.DiceBCELoss(weight=0.7)
metrics = [smp.utils.metrics.IoU(threshold=0.5),]
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=device,
)

logs = test_epoch.run(test_dataloader)
print ("Check 4")
for n in range(len(test_dataset)):
    
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]
    
    gt_mask_background = gt_mask[0].squeeze()
    gt_mask_second = gt_mask[1].squeeze()
    gt_mask_tert = gt_mask[2].squeeze()

    x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)


    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    pr_mask_second = pr_mask[1].squeeze()
    pr_mask_tert = pr_mask[2].squeeze()
    
    pr_mask_background = pr_mask[0].squeeze()
        
    pmm.util.visualize(
        image=image_vis, 
        ground_truth_secondary=gt_mask_second,
        predicted_secondary=pr_mask_second,
        ground_truth_tertiary=gt_mask_tert,
        predicted_tertiary=pr_mask_tert
    )
'''

model_path = Path('models', 'AlSi10Mg_model.tar')
model, preprocessing_fn = pmm.segmentation_training.load_segmentation_model(model_path, classes=3)

im_path ='crop180.tif' 
im = imageio.imread(im_path)
#truth = imageio.imread(annot_path)
#truth = truth[:,:,:3]
st = time.time()
pred = pmm.segmentation_training.segmentation_models_inference(im, model, preprocessing_fn, batch_size=4, patch_size=512,
                                                     probabilities=None)
end = time.time()
execution = end - st
print (execution)
#print (pred.shape)
#meltpools = pred[...,0]
#background = pred[...,1]

np.save('crop180.npy',pred)
#np.save('Back.npy', background)

'''
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16,16))
ax0.imshow(im)
ax0.set_title('Original Image')
ax1.imshow(pred[...,0])
ax1.set_title('Tertiary precipitates')
ax2.imshow(pred[...,1])
ax2.set_title('Tertiary precipitates')
plt.savefig('crop45_seg_new2.png')
'''
'''
labels = [[0,0,255], [255,0,0]]
visual = pmm.segmentation_training.visualize_prediction_accuracy(pred, truth, labels)
plt.figure(figsize=(12,12))
plt.imshow(visual)
plt.savefig('New.png')
'''
