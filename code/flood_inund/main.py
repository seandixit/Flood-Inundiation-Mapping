import os
import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import time
import torchvision.models as models
import segmentation_models_pytorch as smp
import numpy as np

from data.data import get_dataset
from models import *
from loss import *
from metrics import *
import config

from tqdm import tqdm

import matplotlib.pyplot as plt
import cv2
import re

import numpy as np
import torch
import time


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

CKPT_PATH = "weights/Prithvi_100M.pt"
CFG_PATH = "weights/Prithvi_100M_config.yaml"


def get_meta_data(DATASET_PATH):
    """
    reads metadata from the dataset to get the height and width of each image file. It also 
    updates global maximum and minimum elevation values (config.GLOBAL_MAX and config.GLOBAL_MIN).
    """
    DATASET = os.listdir(DATASET_PATH)
    DATASET = [file for file in DATASET if  file.endswith(".npy") and re.search("Features", file)]

    META_DATA = dict()

    for file_name in DATASET:
        file = np.load(os.path.join(DATASET_PATH, file_name))
        #print(file.shape)
        file_height, file_width, _ = file.shape
        #print(file_height)
        #print(file_width)

        elev_data = file[:, :, 3]
        file_elev_max = np.max(elev_data)
        file_elev_min = np.min(elev_data)
        # print(file_elev_max)
        # print(file_elev_min)

        if file_elev_max>config.GLOBAL_MAX:
            config.GLOBAL_MAX = file_elev_max
        if file_elev_min<config.GLOBAL_MIN:
            config.GLOBAL_MIN = file_elev_min


        META_DATA[file_name] = {"height": file_height,
                                "width": file_width}
        
    return META_DATA

def run_pred(model, data_loader):
    """
    performs predictions using the trained model on the test dataset 
    and returns the predicted patches in a dictionary format.
    """
    ## Model gets set to evaluation mode
    model.eval()
    pred_patches_dict = dict()

    for data_dict in tqdm(data_loader):

        ## RGB data
        rgb_data = data_dict['rgb_data'].float().to(DEVICE)

        ## Data labels
        labels = data_dict['labels'].float().to(DEVICE)

        ## Get filename
        filename = data_dict['filename']
        # print("filename: ", filename)

        ## Get model prediction
        pred = model(rgb_data)

        ## Remove pred and GT from GPU and convert to np array
        pred_labels_np = pred.detach().cpu().numpy()
        gt_labels_np = labels.detach().cpu().numpy()

        ## Save Image and RGB patch
        for idx in range(rgb_data.shape[0]):
            pred_patches_dict[filename[idx]] = pred_labels_np[idx, :, :, :]

    return pred_patches_dict


def find_patch_meta(pred_patches_dict):
    """
    finds the maximum y and x coordinates for the stitched image based on the patch keys.
    """
    y_max = 0
    x_max = 0

    for item in pred_patches_dict:

        temp = int(item.split("_")[3])
        if temp>y_max:
            y_max = temp

        temp = int(item.split("_")[5])
        if temp>x_max:
            x_max = temp


    y_max+=1
    x_max+=1
    
    return y_max, x_max


def stitch_patches_GT_labels(pred_patches_dict, TEST_REGION):
    """
    stitches the predicted and ground truth patches for the test dataset based on the provided test region.
    """
    cropped_data_path = f"./data/Region_{TEST_REGION}_TEST/cropped_data_val_test"
    y_max, x_max = find_patch_meta(pred_patches_dict)
    
    for i in range(y_max):
        for j in range(x_max):
            dict_key = f"Region_{TEST_REGION}_y_{i}_x_{j}_features.npy"
            dict_key_label = f"Region_{TEST_REGION}_y_{i}_x_{j}_label.npy"
            #print(dict_key)
        
            pred_patch = pred_patches_dict[dict_key]
            pred_patch = np.transpose(pred_patch, (1, 2, 0))

            label_patch = np.load(os.path.join(cropped_data_path, dict_key_label))

            if j == 0:
                label_x_patches = label_patch
                pred_x_patches = pred_patch
            else:
                label_x_patches = np.concatenate((label_x_patches, label_patch), axis = 1)
                pred_x_patches = np.concatenate((pred_x_patches, pred_patch), axis = 1)

            ## rgb_patches.append(rgb_patch)
            ## pred_patches.append(pred_patch)
    
        if i == 0:
            label_y_patches = label_x_patches
            pred_y_patches = pred_x_patches
        else:
            label_y_patches = np.vstack((label_y_patches, label_x_patches))
            pred_y_patches = np.vstack((pred_y_patches, pred_x_patches))
        

    label_stitched = label_y_patches
#     pred_stitched = np.argmax(pred_y_patches, axis = -1)
    pred_stitched = pred_y_patches.copy()
    
    return label_stitched, pred_stitched

def stitch_patches_augmented(pred_patches_dict, TRAIN_REGION, TEST_REGION):
    """
    similar to the above but includes an additional step for augmentation before stitching
    """
    cropped_data_path = f"./data/{TRAIN_REGION}_{TEST_REGION}/cropped_data_val_test"
    y_max, x_max = find_patch_meta(pred_patches_dict)

    for i in range(y_max):
        for j in range(x_max):
            TEST_REGION_ID = TEST_REGION.split("_")[1]
            dict_key = f"Region_{TEST_REGION_ID}_y_{i}_x_{j}_features.npy"

            pred_patch = pred_patches_dict[dict_key]
            pred_patch = np.transpose(pred_patch, (1, 2, 0))

            rgb_patch = np.load(os.path.join(cropped_data_path, dict_key))[:, :, :3]

            if j == 0:
                rgb_x_patches = rgb_patch[config.EXTRA_PIXELS:config.SPATIAL_SIZE-config.EXTRA_PIXELS, config.EXTRA_PIXELS:config.SPATIAL_SIZE-config.EXTRA_PIXELS, :3]
                pred_x_patches = pred_patch[config.EXTRA_PIXELS:config.SPATIAL_SIZE-config.EXTRA_PIXELS, config.EXTRA_PIXELS:config.SPATIAL_SIZE-config.EXTRA_PIXELS] # 4:124, 4:124
            else:
                rgb_x_patches = np.concatenate((rgb_x_patches, rgb_patch[config.EXTRA_PIXELS:config.SPATIAL_SIZE-config.EXTRA_PIXELS, config.EXTRA_PIXELS:config.SPATIAL_SIZE-config.EXTRA_PIXELS,:3]), axis = 1)
                pred_x_patches = np.concatenate((pred_x_patches, pred_patch[config.EXTRA_PIXELS:config.SPATIAL_SIZE-config.EXTRA_PIXELS, config.EXTRA_PIXELS:config.SPATIAL_SIZE-config.EXTRA_PIXELS,:3]), axis = 1)

        if i == 0:
            rgb_y_patches = rgb_x_patches
            pred_y_patches = pred_x_patches
        else:
            rgb_y_patches = np.vstack((rgb_y_patches, rgb_x_patches))
            pred_y_patches = np.vstack((pred_y_patches, pred_x_patches))


    rgb_stitched = rgb_y_patches.astype('uint8')
    # pred_stitched = np.argmax(pred_y_patches, axis = -1)
    pred_stitched = pred_y_patches.copy()

    return rgb_stitched, pred_stitched

def stitch_patches(pred_patches_dict, TEST_REGION):
    cropped_data_path = f"./data/Region_{TEST_REGION}_TEST/cropped_data_val_test"
    y_max, x_max = find_patch_meta(pred_patches_dict)
    
    for i in range(y_max):
        for j in range(x_max):
            dict_key = f"Region_{TEST_REGION}_y_{i}_x_{j}_features.npy"
            #print(dict_key)
        
            pred_patch = pred_patches_dict[dict_key]
            pred_patch = np.transpose(pred_patch, (1, 2, 0))

            rgb_patch = np.load(os.path.join(cropped_data_path, dict_key))[:, :, :3]


            if j == 0:
                rgb_x_patches = rgb_patch
                pred_x_patches = pred_patch
            else:
                rgb_x_patches = np.concatenate((rgb_x_patches, rgb_patch), axis = 1)
                pred_x_patches = np.concatenate((pred_x_patches, pred_patch), axis = 1)

            ## rgb_patches.append(rgb_patch)
            ## pred_patches.append(pred_patch)
    
        if i == 0:
            rgb_y_patches = rgb_x_patches
            pred_y_patches = pred_x_patches
        else:
            rgb_y_patches = np.vstack((rgb_y_patches, rgb_x_patches))
            pred_y_patches = np.vstack((pred_y_patches, pred_x_patches))
        

    rgb_stitched = rgb_y_patches.astype('uint8')
#     pred_stitched = np.argmax(pred_y_patches, axis = -1)
    pred_stitched = pred_y_patches.copy()
    
    return rgb_stitched, pred_stitched

def center_crop(stictched_data, original_height, original_width, image = False):
    """
    crops the stitched data to the original dimensions.
    """

    current_height, current_width = stictched_data.shape[0], stictched_data.shape[1]

    height_diff = current_height-original_height
    width_diff = current_width-original_width

    cropped = stictched_data[height_diff//2:current_height-height_diff//2, width_diff//2: current_width-width_diff//2]

    return cropped

def center_crop_augmented(stictched_data, original_height, original_width, image = False):

    current_height, current_width = stictched_data.shape[0], stictched_data.shape[1]

    height_diff = current_height-original_height
    width_diff = current_width-original_width

    if image:
        cropped = stictched_data[0:current_height-height_diff, 0:current_width-width_diff, :]
    else:
        cropped = stictched_data[0:current_height-height_diff, 0:current_width-width_diff]

    return cropped



def ann_to_labels(png_image):
    """
    converts annotated images to binary labels representing forest and non-forest areas.
    """
    ann = cv2.imread(png_image)
    ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)

    forest = ann[:, :, 1] > 0
    not_forest = ann[:, :, 2] > 0

    forest_arr = np.where(forest, 1, 0)
    not_forest_arr = np.where(not_forest, -1, 0)

    final_arr = forest_arr + not_forest_arr
    
    return final_arr


def train(TRAIN_REGION, TEST_REGION, NUM_EPOCHS, modelNo, RESUME_EPOCH=0):
    """
    performs the training of the U-Net model. It includes data loading, model initialization, loss calculation, and optimization.
    """
    if not os.path.exists(f"./saved_models/{TEST_REGION}"):
        os.mkdir(f"./saved_models/{TEST_REGION}")

    config.EPOCHS = NUM_EPOCHS

    cropped_train_data_path = f"./data/{TRAIN_REGION}_{TEST_REGION}/cropped_data_train"

    cropped_val_test_data_path = f"./data/{TRAIN_REGION}_{TEST_REGION}/cropped_data_val_test"

    DATASET_PATH = "./data/repo/Features_7_Channels"
    META_DATA = get_meta_data(DATASET_PATH)

    elev_train_dataset = get_dataset(cropped_train_data_path, config.IN_CHANNEL)
    elev_val_test_dataset = get_dataset(cropped_val_test_data_path, config.IN_CHANNEL) 

    train_seq = np.arange(0, len(elev_train_dataset), dtype=int)
    # print(len(train_seq))

    d_len = len(elev_val_test_dataset)
    # print(d_len)

    val_idx = int(0.5*d_len)
    val_seq = np.arange(0, val_idx, 1, dtype=int)
    # print(len(val_seq))

    test_seq = np.arange(0, d_len, 1, dtype=int)
    # test_seq = np.arange(0, d_len, 1, dtype=int)
    # print(len(test_seq))

    half_test_seq = np.arange(len(val_seq), d_len, 1, dtype=int)
    # print(len(half_test_seq))

    train_dataset = torch.utils.data.Subset(elev_train_dataset, train_seq)
    val_dataset = torch.utils.data.Subset(elev_val_test_dataset, val_seq)
    test_dataset = torch.utils.data.Subset(elev_val_test_dataset, test_seq)

    half_test_dataset = torch.utils.data.Subset(elev_val_test_dataset, half_test_seq)

    train_loader = DataLoader(train_dataset, batch_size = config.BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size = config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size = config.BATCH_SIZE)

    half_test_loader = DataLoader(half_test_dataset, batch_size = config.BATCH_SIZE)

    if (modelNo == 0):
        model = UNet(config.IN_CHANNEL, config.N_CLASSES, ultrasmall = True).to(DEVICE)
        optimizer = SGD(model.parameters(), lr = 2e-6)
    elif (modelNo == 1):
        model = SimpleUNet(config.IN_CHANNEL, config.N_CLASSES).to(DEVICE)
        optimizer = SGD(model.parameters(), lr = 2e-6)
    elif (modelNo == 3):
        model = PrithviEncoderDecoder(
            cfg_path=CFG_PATH,
            num_classes=config.N_CLASSES,
            in_chans=config.IN_CHANNEL,
            img_size=128,
            freeze_encoder=False,
            num_neck_filters=64,
        ).to(DEVICE)
        optimizer = SGD(model.parameters(), lr = 1e-7)
    elif (modelNo == 4):
        model = smp.Unet(
            encoder_name="efficientnet-b5",
            # include_top=False,
            in_channels=config.IN_CHANNEL,
            classes=config.N_CLASSES
        ).to(DEVICE)
        optimizer = SGD(model.parameters(), lr = 2e-6)
    elif (modelNo == 5):
        model = PrithviUnet(num_classes=3,
        cfg_path= CFG_PATH,
        ckpt_path = None,
        in_chans=config.IN_CHANNEL,
        img_size = 128,
        n = [2, 5, 8, 11],
        norm= True,
        decoder_channels = [128, 64, 32, 16],
        freeze_encoder= False).to(DEVICE)
        #optimizer = torch.optim.AdamW(model.parameters(), eps = 5e-7, weight_decay=0.01)
        optimizer = SGD(model.parameters(), lr = 1e-7)
    elif (modelNo == 6):
        model = Segformer(
            #dims = (32, 64, 128, 256),      # dimensions of each stage
            heads = (1, 1, 1, 1),             # heads of each stage
            ff_expansion = (1, 2, 4, 8),      # feedforward expansion factor of each stage
            #reduction_ratio = (1, 1, 1, 1),    # reduction ratio of each stage for efficient attention
            num_layers = 3,                 # num layers of each stage
            decoder_dim = 128,              # decoder dimension
            num_classes = 3,                # number of segmentation classes
            channels = config.IN_CHANNEL,                   # number of input channels
        ).to(DEVICE)
        #optimizer = torch.optim.AdamW(model.parameters(), eps = 5e-7, weight_decay=0.01)
        optimizer = SGD(model.parameters(), lr = 1e-7)
    elif (modelNo == 2):
        model = smp.Linknet(encoder_name="efficientnet-b5", in_channels=config.IN_CHANNEL, classes=config.N_CLASSES).to(DEVICE)
        optimizer = SGD(model.parameters(), lr = 2e-6)

    criterion = torch.nn.CrossEntropyLoss(reduction = 'sum', ignore_index=0)
    elev_eval = Evaluator()

    model_path = f"./saved_models/{TEST_REGION}/saved_model_{RESUME_EPOCH}.ckpt"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device(DEVICE))
        model.load_state_dict(checkpoint['model'])
        print(f"Resuming from epoch {RESUME_EPOCH}")
    
    
    
    train_loss_dict = dict()
    val_loss_dict = dict()
    min_val_loss = 1e10

    for epoch in range(config.EPOCHS):
        
        ## Model gets set to training mode
        model.train()
        train_loss = 0
        
        for data_dict in tqdm(train_loader):
            #print(data_dict['labels'].shape)
            
            ## Retrieve data from data dict and send to deivce
            
            ## RGB data
            rgb_data = data_dict['rgb_data'].float().to(DEVICE)
            rgb_data.requires_grad = True
            
            """
            ## Data labels
            Elev Loss function label format: Flood = 1, Unknown = 0, Dry = -1 
            """
            labels = data_dict['labels'].long().to(DEVICE)
            labels.requires_grad = False   
            
            #print("Input Shape: ", rgb_data.shape)
            ## Get model prediction
            pred = model(rgb_data)
            
            ## Backprop Loss
            optimizer.zero_grad() 
            loss = criterion.forward(pred, labels)
            
            loss.backward()
            optimizer.step()
            
            ## Record loss for batch
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_loss_dict[epoch+1] = train_loss
        print(f"Epoch: {epoch+1} Training Loss: {train_loss}" )
        
        
        #=====================================================================================
        
        
        ## Do model validation for epochs that match VAL_FREQUENCY
        if (epoch+1)%config.VAL_FREQUENCY == 0:    
            
            ## Model gets set to evaluation mode
            model.eval()
            val_loss = 0 
            print("Starting Validation")
            
            for data_dict in tqdm(val_loader):
                
                ## RGB data
                rgb_data = data_dict['rgb_data'].float().to(DEVICE)
                ## Data labels
                labels = data_dict['labels'].long().to(DEVICE)
                
                ## Get model prediction
                pred = model(rgb_data)            
                
                ## Backprop Loss
                loss = criterion.forward(pred, labels)
                ##print("Loss: ", loss.item())

                ## Record loss for batch
                val_loss += loss.item()
                
                ## Remove pred and GT from GPU and convert to np array
                pred_labels_np = pred.detach().cpu().numpy() 
                gt_labels_np = labels.detach().cpu().numpy()
                
            
            val_loss /= len(val_loader)
            val_loss_dict[epoch+1] = val_loss
            print(f"Epoch: {epoch+1} Validation Loss: {val_loss}" )
            
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                print("Saving Model")
                torch.save({'epoch': epoch + 1,  # when resuming, we will start at the next epoch
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()}, 
                            f"./saved_models/{TEST_REGION}/saved_model_{epoch+1}.ckpt")
    
    return

def run_prediction(TRAIN_REGION, TEST_REGION, RESUME_EPOCH, modelNo):
    """
    uns the prediction using a trained model and evaluates its performance against the ground truth labels.
    """
    start = time.time()
    DATASET_PATH = "./data/repo/Features_7_Channels"

    TEST_REGION_ID = TEST_REGION.split("_")[1]
    gt_labels = np.load(f"./data/repo/groundTruths/Region_{TEST_REGION_ID}_GT_Labels.npy")
    height, width = gt_labels.shape[0], gt_labels.shape[1]

    if (modelNo == 0):
        model = UNet(config.IN_CHANNEL, config.N_CLASSES, ultrasmall = True).to(DEVICE)
    elif (modelNo == 1):
        model = SimpleUNet(config.IN_CHANNEL, config.N_CLASSES).to(DEVICE)
    elif (modelNo == 3):
        model = PrithviEncoderDecoder(
            cfg_path=CFG_PATH,
            ckpt_path=f"./saved_models/{TEST_REGION}/saved_model_{RESUME_EPOCH}.ckpt",
            num_classes=config.N_CLASSES,
            in_chans=config.IN_CHANNEL,
            img_size=128,
            freeze_encoder=False,
            num_neck_filters=64,
        ).to(DEVICE)
    elif (modelNo == 4):
        model = smp.Unet(
            encoder_name="efficientnet-b5",
            # include_top=False,
            in_channels=config.IN_CHANNEL,
            classes=config.N_CLASSES
        ).to(DEVICE)
        model.to(DEVICE)
    elif (modelNo == 5):
        model = PrithviUnet(num_classes=3,
        cfg_path= CFG_PATH,
        ckpt_path = f"./saved_models/{TEST_REGION}/saved_model_{RESUME_EPOCH}.ckpt",
        in_chans=config.IN_CHANNEL,
        img_size = 128,
        n = [2, 5, 8, 11],
        norm= True,
        decoder_channels = [128, 64, 32, 16],
        freeze_encoder= False).to(DEVICE)
    elif (modelNo == 6):
        model = Segformer(
            #dims = (32, 64, 128, 256),      # dimensions of each stage
            heads = (1, 1, 1, 1),             # heads of each stage
            ff_expansion = (1, 2, 4, 8),       # feedforward expansion factor of each stage
            #reduction_ratio = (1, 1, 1, 1),    # reduction ratio of each stage for efficient attention
            num_layers = 3,                 # num layers of each stage
            decoder_dim = 128,              # decoder dimension
            num_classes = 3,                # number of segmentation classes
            channels = config.IN_CHANNEL,                   # number of input channels
        ).to(DEVICE)
    
    elif (modelNo == 2):
        model = smp.Linknet(encoder_name="efficientnet-b5", in_channels=config.IN_CHANNEL, classes=config.N_CLASSES).to(DEVICE)
    

    elev_eval = Evaluator()


    model_path = f"./saved_models/{TEST_REGION}/saved_model_{RESUME_EPOCH}.ckpt"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device(DEVICE))
        model.load_state_dict(checkpoint['model'])

    
    ## Model gets set to evaluation mode
    model.eval()
    pred_patches_dict = dict()

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

    # META_DATA = get_meta_data(DATASET_PATH)
    
    ## Run prediciton
    cropped_data_path_al = f"./data/{TRAIN_REGION}_{TEST_REGION}/cropped_data_val_test"
    test_dataset = get_dataset(cropped_data_path_al, config.IN_CHANNEL)

    test_loader = DataLoader(test_dataset, batch_size = config.BATCH_SIZE)

    pred_patches_dict = run_pred(model, test_loader)

    rgb_stitched, pred_stitched = stitch_patches_augmented(pred_patches_dict, TRAIN_REGION, TEST_REGION)
    rgb_unpadded = center_crop_augmented(rgb_stitched, height, width, image=True)
    pred_unpadded = center_crop_augmented(pred_stitched, height, width, image = False)

    # print("pred_unpadded.shape: ", pred_unpadded.shape)
    pred_final = pred_unpadded[:,:,1]

    pred_binarized = np.where(pred_final > 0.5, 1, -1)

    elev_eval.run_eval(pred_binarized, gt_labels)

    return rgb_unpadded, pred_binarized, gt_labels

    
if __name__ == "__main__":
    TEST_REGION = "1"

    







