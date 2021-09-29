import argparse
import scipy, math
from scipy import ndimage
import cv2
import numpy as np
import sys
import json
import models
import dataloaders
from utils.helpers import colorize_mask
from utils.pallete import get_voc_pallete
from utils import metrics
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from math import ceil
from PIL import Image
from pathlib import Path
import SimpleITK as sitk
from LungSeg import lung_segmentation
from medpy import metric
import pandas as pd
from scipy.ndimage import zoom
from skimage import measure



os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def resample_ct_image(image, new_size, new_spacing, interpolator=sitk.sitkLinear, min_value=None):
    resample = sitk.ResampleImageFilter()

    resample.SetSize(new_size)
    resample.SetInterpolator(interpolator)
    resample.SetOutputSpacing(new_spacing)

    if min_value:
        resample.SetDefaultPixelValue(min_value)

    new_image = resample.Execute(image)

    return new_image

def resample_ct_image_with_shape(image, spacing, new_shape, interpolator=sitk.sitkLinear, min_value=None):
    size = [image.shape[2], image.shape[1], image.shape[0]]

    new_spacing = [size[0] * spacing[0] / new_shape[0],
                size[1] * spacing[1] / new_shape[1],
                size[2] * spacing[2] / new_shape[2]]
    print (new_spacing)
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(spacing)
    return resample_ct_image(image, new_shape, new_spacing, interpolator=interpolator, min_value=min_value)

def resample_ct_image_with_spacing(image, spacing, new_spacing, interpolator=sitk.sitkLinear, min_value=None):
    size = [image.shape[2], image.shape[1], image.shape[0]]

    new_size = [int(size[0] * spacing[0] / new_spacing[0]),
                int(size[1] * spacing[1] / new_spacing[1]),
                int(size[2] * spacing[2] / new_spacing[2])]
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(spacing)
    return resample_ct_image(image, new_size, new_spacing, interpolator=interpolator, min_value=min_value)


def read_dcm_series(path):
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    # asd = 0

    return dice, jc, hd, asd

def get_patch_position(shape, window, stride):
    patch_position = []

    def get_anchors(axis_idx):
        anchors = [anchor for anchor in range(0, shape[axis_idx] - window[axis_idx], stride[axis_idx])]
        cur = anchors[-1] + stride[axis_idx]
        if cur + window[axis_idx] <= shape[axis_idx]:
            anchors.append(cur)
        else:
            anchors.append(shape[axis_idx] - window[axis_idx])

        return anchors

    x_anchors = get_anchors(2)
    y_anchors = get_anchors(1)
    z_anchors = get_anchors(0)

    for z in z_anchors:
        for y in y_anchors:
            for x in x_anchors:
                patch_position.append((z, y, x))

    return patch_position


def normalize(img, max, min):
    img [img>max] = max
    img[img<min] = min
    image = (img-min)/(max-min)
    return image

class testDataset(Dataset):
    def __init__(self, images):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        images_path = Path(images)
        self.filelist = list(images_path.glob("*.jpg"))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        image_path = self.filelist[index]
        image_id = str(image_path).split("/")[-1].split(".")[0]
        image = Image.open(image_path)
        image = self.normalize(self.to_tensor(image))
        return image, image_id

def multi_scale_predict(model, image, scales, num_classes, flip=True):
    H, W = (image.size(2), image.size(3))
    upsize = (ceil(H / 8) * 8, ceil(W / 8) * 8)
    upsample = nn.Upsample(size=upsize, mode='bilinear', align_corners=True)
    pad_h, pad_w = upsize[0] - H, upsize[1] - W
    image = F.pad(image, pad=(0, pad_w, 0, pad_h), mode='reflect')

    total_predictions = np.zeros((num_classes, image.shape[2], image.shape[3]))

    for scale in scales:
        scaled_img = F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=False)
        scaled_prediction = upsample(model(scaled_img))

        if flip:
            fliped_img = scaled_img.flip(-1)
            fliped_predictions = upsample(model(fliped_img))
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)
    return total_predictions[:, :H, :W]

def predict(model, image, num_classes):
    prediction = model(image)
    pred_mask = F.softmax(prediction, dim=1)
    pred_mask = pred_mask.cpu().numpy()

    return pred_mask

def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction==i)
        label_tmp = (label==i)

        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return dice

def dice_loss(input, target):
    # N = target.size(0)
    smooth = 1

    input_flat = input
    target_flat = target

    intersection = input_flat * target_flat

    loss = 2 * (np.sum(intersection + smooth) / (np.sum(input_flat) + np.sum(target_flat) + smooth))
    loss = 1 - loss
    return loss

def main():
    args = parse_arguments()

    # CONFIG
    assert args.config
    config = json.load(open(args.config))
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]

    num_classes = 2

    # MODEL
    config['model']['supervised'] = True; config['model']['semi'] = False
    model = models.CCT(num_classes=num_classes,
                        conf=config['model'], testing=True)

    checkpoint = torch.load(args.model)
    model = torch.nn.DataParallel(model)

    pretrained_dict_2 = {}
    model_dict = model.state_dict()
    for k, v in checkpoint['state_dict'].items():
        if k in model_dict:
            pretrained_dict_2[k] = v
        else:
            print (k)

    model_dict.update(pretrained_dict_2)
    model.load_state_dict(model_dict)

    model.eval()
    model.cuda()

    test_path = args.data_path
    test_mask_path = args.mask_path

    window = [80, 112, 112]
    stride = [80, 112, 112]
    dice_test_all_avg = 0
    pat_num = 0

    pat_predict_result = []


    count = 0
    for data_path in os.listdir(test_path):
        dice_test_avg = 0
        if data_path == 'test_val':
            continue

        for pat in os.listdir(test_path + data_path):
            print(pat)


            pat_img = np.load(test_path + data_path + '/'+pat)
            pat_mask = np.load(test_mask_path + data_path + '_mask'+'/' + pat[:-4]+'_mask.npy')

            pat_pred = np.zeros_like(pat_img)

            try:
                patch_positions = get_patch_position(pat_img.shape, window, stride)
            except:
                continue
            for position in patch_positions:
                patch_img_arr = np.zeros([window[0], window[1], window[2]], dtype='float32')

                patch_z_min = position[0]
                patch_z_max = patch_z_min + window[0]
                patch_y_min = position[1]
                patch_y_max = patch_y_min + window[1]
                patch_x_min = position[2]
                patch_x_max = patch_x_min + window[2]
                patch_img_arr = pat_img[patch_z_min:patch_z_max,
                                patch_y_min:patch_y_max,
                                patch_x_min:patch_x_max]
                patch_img_arr = np.transpose(patch_img_arr, (1, 2, 0))

                patch_img_arr = patch_img_arr.reshape(1, 1, patch_img_arr.shape[0], patch_img_arr.shape[1], patch_img_arr.shape[2]).astype(np.float32)
                patch_img_arr = torch.from_numpy(patch_img_arr.astype(np.float32)).cuda()

                # PREDICT
                with torch.no_grad():
                    output_patch = predict(model, patch_img_arr, num_classes)
                    output_patch = output_patch[0]
                pred_patch = np.asarray(np.argmax(output_patch, axis=0), dtype=np.uint8)

                pred_patch = np.transpose(pred_patch, (2, 0, 1))
                pat_pred[patch_z_min:patch_z_max, patch_y_min:patch_y_max, patch_x_min:patch_x_max] = pred_patch

            dice_coef = cal_dice(pat_pred, pat_mask)
            dice_test_avg += dice_coef
            print ('dice_coef', dice_coef)

            metrix = calculate_metric_percase(pat_pred, pat_mask[:])
            print(metrix[0])
            pat_predict_result.append([pat, metrix[0], metrix[1], metrix[2], metrix[3]])


        dice_test_all_avg += dice_test_avg
        pat_num += len(os.listdir(test_path + data_path))
        dice_test_avg = dice_test_avg / len(os.listdir(test_path + data_path))
        print('test dataset'+ data_path + 'avg dice', dice_test_avg)

    print(count)
    dice_test_all_avg = dice_test_all_avg/pat_num
    print(args.model)
    print ('test_all_dice', dice_test_all_avg)
    #


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default='./saved_semi_confi_1125_flip/CCT/config.json',type=str,
                        help='Path to the config file')
    parser.add_argument( '--model', default='./saved_semi_confi_1125_flip/CCT/checkpoint_best.pth', type=str,
                        help='Path to the trained .pth model')
    parser.add_argument('--data_path', default="/home2/pneumonia_ct/test/image/", type=str,
                        help='Test images for Pascal VOC')
    parser.add_argument('--mask_path', default="/home2/pneumonia_ct/test/label", type=str,
                        help='Test images for Pascal VOC')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()

