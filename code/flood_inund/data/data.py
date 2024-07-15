import os
import re
import numpy as np

import torch
import torchvision


from torchvision import transforms

import math
import sys

sys.path.append('../')

import config



"""
data.py used to preprocess data, preparing data in format for training
"""

################################################################################
## Create Torch Dataset from cropped images
################################################################################

class ElevationDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_path, transforms, input_channel): # TODO
        """
        data_path, transforms (data transformations), input_channel (number of channels in input data)
        we could input 6 channels instead (prithvi model only supports 6 channels)
        """
        self.data_path = data_path
        self.transforms = transforms
        
        self.feature_files = os.listdir(data_path)
        self.feature_files = [file for file in self.feature_files if file.endswith(".npy") and re.match(".*features.*", file) ]
        
        self.data_len = len(self.feature_files)
        self.input_channel = input_channel
#         print(self.data_len)
        
        assert self.data_len>0, "No data found!!"
    
    
    def normalize(self, data):        
        """
        data (input data to be normalized)
        """
        # global_max = 76.05
        # global_min = -4.965000152587891
        
        global_max = config.GLOBAL_MAX
        global_min = config.GLOBAL_MIN
        
        normalized_data = (data-(global_min))/(global_max-global_min)
        
        assert np.min(normalized_data) >=0, "Normalized value should be greater than equal 0"
        assert np.max(normalized_data) <=1, "Normalized value should be lesser than equal 1"
        
        return normalized_data
    
    
    
    def __getitem__(self, idx):
        """
        idx: index of item to retrieve from dataset
        -Retrieves a data item (feature and label) at the specified index.
        -Separates elevation data and RGB channels from the feature data.
        -Formats the label data to fit the requirements of the cross-entropy loss.
        -Concatenates the RGB and elevation data based on the input_channel value.
        -Applies the specified transformations to the RGB data.
        -Returns a dictionary containing the filename, transformed RGB data, elevation data, normalized elevation data, and labels.
        """
        self.data_dict = dict()
        
        ## Get the feature patch
        self.feature_file = self.feature_files[idx]
        self.feature_data = np.load(os.path.join(self.data_path, self.feature_file))
        
        ## Get the corresponding label for forests (forest model)
        self.label_file = re.sub("features", "label", self.feature_file)
        self.label_data = np.load(os.path.join(self.data_path, self.label_file)).astype('int')
        
        
        ## Seperate elevation data from RGB
        self.disaster_rgb = self.feature_data[:,:, :3].astype('uint8')
        self.elev_data = self.feature_data[:,:, 3].astype('float32')
        self.elev_data = np.expand_dims(self.elev_data, -1).astype('float32')
        self.regular_rgb = self.feature_data[:,:, 4:].astype('uint8')
        
        
        ## Format labels for Loss function   
        
        ### For testing labels with CE loss 
        # original label: 1: flood, -1: dry, 0: unknown
        # required for CE loss: 0: unknown, 1: flood, 2: dry
        self.formatted_label_data = np.where(self.label_data == -1, 2, self.label_data).astype('int')
        
        ## Merge Disaster and regular time RGB
        self.rgb_data = self.disaster_rgb
        # print("self.elev_data.shape: ", self.elev_data.shape)
        
        if self.input_channel == 7:
            self.rgb_data = np.concatenate((self.rgb_data, self.elev_data, self.regular_rgb), axis = -1)
        elif self.input_channel == 4:
            self.rgb_data = np.concatenate((self.rgb_data, self.elev_data), axis = -1)
        elif self.input_channel == 3:
            self.rgb_data = self.rgb_data
        elif self.input_channel == 6:
            self.rgb_data = np.concatenate((self.rgb_data, self.regular_rgb), axis = -1)
        # print("self.rgb_data.shape: ", self.rgb_data.shape)
        
        
        
        ## Apply torchvision tranforms to rgb data
        self.transformed_rbg = self.transforms(self.rgb_data)
        
        ## Normalize to elev_data
        self.norm_elev_data = self.normalize(self.elev_data)
        
        ## Put all data in one dictionary
        self.data_dict['filename'] = self.feature_file
        self.data_dict['rgb_data'] = self.transformed_rbg
        self.data_dict['elev_data'] = self.elev_data
        self.data_dict['norm_elev_data'] = self.norm_elev_data
        self.data_dict['labels'] = self.formatted_label_data
        
        return self.data_dict
    
    
    def __len__(self):
        """
        returns length of dataset
        """
        return self.data_len
        

def get_dataset(cropped_data_path, input_channel):
    """
    cropped_data_path (path to directory with cropped data), input_channel (number of channels in input data)
    Returns instance of elevation dataset
    """
    # print("get_dataset")
    
    training_transforms = []
    training_transforms += [transforms.ToTensor()]
    
    if input_channel == 7:
        training_transforms += [torchvision.transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                                                                 (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5))]
    elif input_channel == 4:
        training_transforms += [torchvision.transforms.Normalize((0.5, 0.5, 0.5, 0.5),
                                                                 (0.5, 0.5, 0.5, 0.5))]
    elif input_channel == 3:
        training_transforms += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                 (0.5, 0.5, 0.5))]
    elif input_channel == 6:
        training_transforms += [torchvision.transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                                                                 (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))]
    else:
        print("Invalid number of input channels")
        exit(0)
    
    data_transforms = transforms.Compose(training_transforms)
    
    elev_dataset = ElevationDataset(cropped_data_path, data_transforms, input_channel)
    
    return elev_dataset
################################################################################
        