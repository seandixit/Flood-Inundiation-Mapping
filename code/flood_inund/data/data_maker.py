import torch
import os
import re
import numpy as np
import math
from matplotlib import pyplot as plt
import sys

sys.path.append("../")
import config
from config import *

import shutil

from tqdm import tqdm

"""
SPATIAL_SIZE: A constant defining the size of spatial patches.
EXTRA_PIXELS: A constant defining the extra pixels to be considered for padding.
"""

def pad_data(unpadded_data, is_feature = False):
    """
    pads the input data to ensure its dimensions are multiples of SPATIAL_SIZE. 
    It also calculates padding based on the size of the input data and returns the padded data.
    """
    
    height = unpadded_data.shape[0]
    width = unpadded_data.shape[1]
    
#     print("height: ", height)
#     print("width: ", width)
    
    width_multiplier = math.ceil(width/SPATIAL_SIZE)
    height_multiplier = math.ceil(height/SPATIAL_SIZE)
    
#     print("width_multiplier: ", width_multiplier)
#     print("height_multiplier: ", height_multiplier)
    
    new_width = SPATIAL_SIZE*width_multiplier
    new_height = SPATIAL_SIZE*height_multiplier
#     print("new_width: ", new_width)
#     print("new_height: ", new_height)
    
    width_pad = new_width-width
    height_pad = new_height-height
    
#     print("width_pad: ", width_pad)
#     print("height_pad: ", height_pad)
    
        
    if width_pad%2 == 0:
        left = int(width_pad/2)
        right = int(width_pad/2)
    else:
        print("Odd Width")
        left = math.floor(width_pad/2)
        right = left+1
    
    if height_pad%2 == 0:
        top = int(height_pad/2)
        bottom = int(height_pad/2)
    else:
        print("Odd Height")
        top = math.floor(height_pad/2)
        bottom = top+1
    
#     print("left: ", left)
#     print("right: ", right)
#     print("top: ", top)
#     print("bottom: ", bottom)
        
    if is_feature:
        data_padded = np.pad(unpadded_data, pad_width = ((top, bottom),(left, right), (0, 0)), mode = 'reflect')
        
#         plt.figure(figsize=(10,10))
#         plt.imshow(data_padded[:,:,:3].astype('int'))
    else:
        data_padded = np.pad(unpadded_data, pad_width = ((top, bottom), (left, right)), mode = 'reflect')
        
    assert data_padded.shape[0]%SPATIAL_SIZE == 0, f"Padded height must be multiple of SPATIAL_SIZE: {SPATIAL_SIZE}"
    assert data_padded.shape[1]%SPATIAL_SIZE == 0, f"Padded width must be multiple of SPATIAL_SIZE: {SPATIAL_SIZE}"

    # print(left, right, top, bottom)
        
#     print("data_padded: ", data_padded.shape, "\n")
    return data_padded

def pad_data_augment(unpadded_data, is_feature = False):
    """
    pads the input data but with an additional mechanism to handle extra pixels using EXTRA_PIXELS
    """
    height = unpadded_data.shape[0]
    width = unpadded_data.shape[1]
    
#     print("height: ", height)
#     print("width: ", width)
    CENTER_SIZE = SPATIAL_SIZE - (EXTRA_PIXELS*2)

    complete_patches_x = width//CENTER_SIZE
    complete_patches_y = height//CENTER_SIZE
    
    rem_pixels_x = width - (complete_patches_x * CENTER_SIZE)
    rem_pixels_y = height - (complete_patches_y * CENTER_SIZE)
    
    # print(complete_patches_x, complete_patches_y)
    # print(rem_pixels_x, rem_pixels_y)
    
    left = EXTRA_PIXELS
    right = EXTRA_PIXELS
    top = EXTRA_PIXELS
    bottom = EXTRA_PIXELS
    
    extra_x = False
    extra_y = False
    
    total_patches_x = complete_patches_x
    total_patches_y = complete_patches_y
    if rem_pixels_x > 0:
        right = SPATIAL_SIZE - rem_pixels_x - EXTRA_PIXELS
        total_patches_x += 1
        extra_x = True
    
    if rem_pixels_y > 0:
        bottom = SPATIAL_SIZE - rem_pixels_y - EXTRA_PIXELS
        total_patches_y += 1
        extra_y = True
    
    new_width = left + width + right
    new_height = top + height + bottom
    
    # print(new_width, new_height)
        
    if is_feature:
        data_padded = np.pad(unpadded_data, pad_width = ((top, bottom),(left, right), (0, 0)), mode = 'reflect')
        
#         plt.figure(figsize=(10,10))
#         plt.imshow(data_padded[:,:,:3].astype('int'))
    else:
        data_padded = np.pad(unpadded_data, pad_width = ((top, bottom), (left, right)), mode = 'reflect')
        
#     assert data_padded.shape[0]%SPATIAL_SIZE == 0, f"Padded height must be multiple of SPATIAL_SIZE: {SPATIAL_SIZE}"
#     assert data_padded.shape[1]%SPATIAL_SIZE == 0, f"Padded width must be multiple of SPATIAL_SIZE: {SPATIAL_SIZE}"
        
#     print("data_padded: ", data_padded.shape, "\n")
    return data_padded, total_patches_x, total_patches_y

def crop_data_augment(uncropped_data, filename, horizontal_patches, vertial_patches, TEST_REGION, is_feature = False, is_conf = False, is_forest = False):
    """
    crops the padded data into patches of size SPATIAL_SIZE x SPATIAL_SIZE. 
    It saves the cropped patches to disk and returns the total number of horizontal and vertical patches.
    """

    output_path = "./data/cropped_data"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    height = uncropped_data.shape[0]
    width = uncropped_data.shape[1]

    # print(f"height: {height} width: {width} ")
    # print("spatial size: ", SPATIAL_SIZE)

    # print("VP: ", vertial_patches)
    # print("HP: ", horizontal_patches)
    
    cropped_data = []
    
    x_start = 0
    y_start = 0
    
    for y in range(0, vertial_patches):
        for x in range(0, horizontal_patches):
            
            if is_feature:
                new_name = f"Region_{TEST_REGION}"+"_y_"+str(y)+"_x_"+str(x)+"_features.npy"
            elif is_conf:
                new_name = f"Region_{TEST_REGION}"+"_y_"+str(y)+"_x_"+str(x)+"_label_conf.npy"
            elif is_forest:
                new_name = f"Region_{TEST_REGION}"+"_y_"+str(y)+"_x_"+str(x)+"_label_forest.npy"
            else:
                new_name = f"Region_{TEST_REGION}"+"_y_"+str(y)+"_x_"+str(x)+"_label.npy"
            
            x_end = x_start + SPATIAL_SIZE
            y_end = y_start + SPATIAL_SIZE
            
            patch = uncropped_data[y_start: y_end, x_start:x_end]
            
            np.save(os.path.join(output_path, new_name), patch)
            
            # for next patch
            # x_start = x_start + SPATIAL_SIZE - (EXTRA_PIXELS*2)
            x_start = x_start + SPATIAL_SIZE - (EXTRA_PIXELS*2)
        x_start = 0
        # y_start = y_start + SPATIAL_SIZE - (EXTRA_PIXELS*2)
        y_start = y_start + SPATIAL_SIZE - (EXTRA_PIXELS*2)

def crop_data(uncropped_data, filename, TEST_REGION, is_feature = False):
    """
    crops the unpadded data into patches of size SPATIAL_SIZE x SPATIAL_SIZE. 
    It saves the cropped patches to disk.
    """
    output_path = "./data/cropped_data"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    height = uncropped_data.shape[0]
    width = uncropped_data.shape[1]
    
    # print("crop input height: ", height)
    # print("crop input width: ", width)
    
    vertial_patches = height//SPATIAL_SIZE
    horizontal_patches = width//SPATIAL_SIZE
    
    # print("vertial_patches: ", vertial_patches)
    # print("horizontal_patches: ", horizontal_patches)
    # print(filename)
    
    cropped_data = []
    
    for y in range(0, vertial_patches):
        for x in range(0, horizontal_patches):
            
            if is_feature:
                new_name = f"Region_{TEST_REGION}"+"_y_"+str(y)+"_x_"+str(x)+"_features.npy"
            else:
                new_name = f"Region_{TEST_REGION}"+"_y_"+str(y)+"_x_"+str(x)+"_label.npy"
            
            # print("new_name: ", new_name)
            
            x_start = (x)*SPATIAL_SIZE
            x_end = (x+1)*SPATIAL_SIZE
            
            y_start = (y)*SPATIAL_SIZE
            y_end = (y+1)*SPATIAL_SIZE
            
            patch = uncropped_data[y_start: y_end, x_start:x_end]
            
            # print(patch.shape)
            
            np.save(os.path.join(output_path, new_name), patch)

def count_to_ratio(label_data):
    """
    calculates the ratio of flood to dry counts for each pixel based on the label data. 
    It returns a ratio array.
    """
    #print("label_data: ", label_data.shape)
    
    total_count = np.sum(label_data, axis = -1, keepdims = True)
    #print("total_count: ", total_count.shape)
    
    max_count_idx = np.expand_dims(np.argmax(label_data, axis=-1), axis =-1)
    #print("max_count_idx: ", max_count_idx.shape)
    
    flood_mask = np.where(max_count_idx == 0, 1, 0)
    dry_mask = np.where(max_count_idx == 1, 1, 0)
    #print("flood_mask: ", flood_mask.shape)
    
    flood_count, dry_count = np.split(label_data, 2, axis = -1)
    #print("flood_count: ", flood_count.shape)
    
    flood_count_masked = flood_count*flood_mask
    dry_count_masked = -dry_count*dry_mask
    #print("flood_count_masked: ", flood_count_masked.shape)

    merged_count = flood_count_masked+dry_count_masked
    #print("merged_count: ", merged_count.shape)
    
    label_data_ratio = merged_count/total_count
    label_data_ratio = np.squeeze(label_data_ratio, axis = -1)
    
    ## Remove NaN cause by 0/total count
    label_data_ratio = np.nan_to_num(label_data_ratio, nan=0.0)
    #print("label_data_ratio: ", label_data_ratio.shape)
    
    
    return label_data_ratio


def make_data(feature_files, feature_data_path, label_data_path, TEST_REGION):
    """
    processes the feature and label data by padding, cropping, and saving the data patches. 
    It uses pad_data_augment and crop_data_augment for padding and cropping, respectively.
    """
    for feature_file in tqdm(feature_files):
        
        ##print("feature_file: ", feature_file)
        region_num = int(feature_file.split("_")[1])
        
        if region_num == TEST_REGION:
            ## Load feature data:
            feature_data = np.load(os.path.join(feature_data_path, feature_file))
            # print("feature_data.shape: ", feature_data.shape)

            ## Load label data:
            label_file = f"Region_{TEST_REGION}_GT_Labels.npy"
            # print(label_file)

            try:
                label_data = np.load(os.path.join(label_data_path, label_file))
            except:
                print(f"No such files as {label_file}")
                
            ###########Padd data to fit SPATIAL_SIZE pathches######################################
            padded_feature, hor_patches, ver_patches = pad_data_augment(feature_data, is_feature = True)
            padded_label, hor_patches, ver_patches = pad_data_augment(label_data)

            ###########Crop data to SPATIAL_SIZE pathches######################################
            cropped_feature = crop_data_augment(padded_feature, feature_file, hor_patches, ver_patches, TEST_REGION, is_feature = True)
            cropped_label = crop_data_augment(padded_label, label_file, hor_patches, ver_patches, TEST_REGION)

def make_data_augmented(feature_files, feature_data_path, label_data_path, reg_nums):
    """
    Similar to make_data, but this function works with multiple regions specified by reg_nums. 
    It uses pad_data_augment and crop_data_augment for padding and cropping, respectively.
    """

    # print("augmenting data")
    for feature_file in tqdm(feature_files):
        
        ##print("feature_file: ", feature_file)
        region_num = int(feature_file.split("_")[1])
        
        ##print("region_num: ", region_num)
        ##print("reg_nums: ", reg_nums)
        
        if region_num in reg_nums:
            ## Load feature data:
            feature_data = np.load(os.path.join(feature_data_path, feature_file))
            # print("feature_data.shape: ", feature_data.shape)

            ## Load label data:
            label_file = feature_file[:8]+"_GT_Labels.npy"
            # print(label_file)

            try:
                label_data = np.load(os.path.join(label_data_path, label_file))
            except:
                print(f"No such files as {label_file}")


            ###########Padd data to fit SPATIAL_SIZE pathches######################################
            padded_feature, hor_patches, ver_patches = pad_data_augment(feature_data, is_feature = True)
            padded_label, hor_patches, ver_patches = pad_data_augment(label_data)
            # padded_label_conf, hor_patches, ver_patches = pad_data_augment(label_data_conf)


            # print("padded_label: ", padded_label.shape)
            # print("padded_label max: ", np.max(padded_label))
            # print("padded_label min: ", np.min(padded_label))
            ## plt.imshow(padded_label)

            # print("padded_feature.shape: ", padded_feature.shape)
            # print("padded_label.shape: ", padded_label.shape)

            ###########Crop data to SPATIAL_SIZE pathches######################################
            cropped_feature = crop_data_augment(padded_feature, feature_file, hor_patches, ver_patches, region_num, is_feature = True)
            cropped_label = crop_data_augment(padded_label, label_file, hor_patches, ver_patches, region_num)

def make_dir(TRAIN_REGIONS, TEST_REGIONS):
    """
    creates directories to store the cropped data based on the specified training and test regions.
    """
    train_region_names = sorted(TRAIN_REGIONS)
    test_region_names = sorted(TEST_REGIONS)

    print(f"TRAIN Region: Region_{min(train_region_names)}-{max(train_region_names)}_TRAIN")
    print(f"TEST Region: Region_{min(test_region_names)}_TEST")
    
    if not os.path.exists(f"./data/Region_{min(train_region_names)}-{max(train_region_names)}_TRAIN_Region_{min(test_region_names)}_TEST"):
        os.mkdir(f"./data/Region_{min(train_region_names)}-{max(train_region_names)}_TRAIN_Region_{min(test_region_names)}_TEST")
        os.mkdir(f"./data/Region_{min(train_region_names)}-{max(train_region_names)}_TRAIN_Region_{min(test_region_names)}_TEST/cropped_data_train")
        os.mkdir(f"./data/Region_{min(train_region_names)}-{max(train_region_names)}_TRAIN_Region_{min(test_region_names)}_TEST/cropped_data_val_test")


def move_files(TRAIN_REGIONS, TEST_REGIONS):
    """
    moves the cropped data files to the appropriate directories based on the training and test regions.
    """
    for file in tqdm(os.listdir("./data/cropped_data")):
        file_region_num = int(file.split("_")[1])
        source = os.path.join("./data/cropped_data", file)

        if file_region_num in TEST_REGIONS:
            destination = os.path.join(f"./data/Region_{min(TRAIN_REGIONS)}-{max(TRAIN_REGIONS)}_TRAIN_Region_{min(TEST_REGIONS)}_TEST/cropped_data_val_test", file)
        elif file_region_num in TRAIN_REGIONS:
            destination = os.path.join(f"./data/Region_{min(TRAIN_REGIONS)}-{max(TRAIN_REGIONS)}_TRAIN_Region_{min(TEST_REGIONS)}_TEST/cropped_data_train", file)
        
        shutil.move(source, destination)


def data_maker_main(TRAIN_REGIONS, TEST_REGIONS):
    """
    This is the main function that orchestrates the entire data processing pipeline. 
    It prepares directories, processes the data using make_data_augmented, and moves the data using move_files.
    TRAIN_REGIONS: This list contains region numbers that will be used for training the machine learning model. The data from these regions will be used to train the model to recognize and predict flood inundation areas.
    TEST_REGIONS: This list contains region numbers that will be used for testing the trained model's performance. The data from these regions will not be used during the training phase but will be used to evaluate how well the model generalizes to unseen data.
    """
    
    feature_data_path = "./data/repo/Features_7_Channels"
    data_files = os.listdir(feature_data_path)
    
    label_data_path = "./data/repo/groundTruths"

    ## only keep .npy file
    feature_files = [file for file in data_files if file.endswith(".npy") ]
    
    ## Make directories for train_test regions 
    make_dir(TRAIN_REGIONS, TEST_REGIONS)
    
    ## Pad and crop data
    # make_data(feature_files, feature_data_path, label_data_path, TRAIN_REGIONS+TEST_REGIONS)
    make_data_augmented(feature_files, feature_data_path, label_data_path, TRAIN_REGIONS+TEST_REGIONS)
    
    ## Move image crops to directory
    move_files(TRAIN_REGIONS, TEST_REGIONS)

if __name__ == "__main__":
    TEST_REGIONS = [2]    # region 2 used for testing
    TRAIN_REGIONS = [1, 5] # region 1 and 5 used for training

    data_maker_main(TRAIN_REGIONS, TEST_REGIONS)