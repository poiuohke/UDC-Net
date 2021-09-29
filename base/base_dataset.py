import random, math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from scipy import ndimage
from math import ceil

class BaseDataSet(Dataset):
    def __init__(self, data_dir, split, mean, std, base_size=None, augment=True, val=False,
                jitter=False, use_weak_lables=False, weak_labels_output=None, crop_size=None, scale=False, flip=False, rotate=False,
                blur=False, return_id=False, n_labeled_examples=None):

        self.root = data_dir
        self.split = split
        self.mean = mean
        self.std = std
        self.augment = augment
        self.crop_size = crop_size
        self.jitter = jitter
        self.image_padding = (np.array(mean)*255.).tolist()
        # self.ignore_index = ignore_index
        self.return_id = return_id
        self.n_labeled_examples = n_labeled_examples
        self.val = val

        self.use_weak_lables = use_weak_lables
        self.weak_labels_output = weak_labels_output

        if self.augment:
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur

        self.jitter_tf = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        # self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

        self.files = []
        self._set_files()

        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError
    
    def _load_data(self, index):
        raise NotImplementedError

    def _rotate(self, image, label):
        # Rotate the image with an angle between -10 and 10
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)

        return image, label

    def _crop(self, image, label):
        output_size = (112, 112, 80)
        # pad the sample if necessary
        if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= \
                output_size[2]:
            pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - output_size[0])
        h1 = np.random.randint(0, h - output_size[1])
        d1 = np.random.randint(0, d - output_size[2])

        label = label[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
        image = image[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
        # print (np.max(label))

        return image, label

    def _blur(self, image, label):
        # Gaussian Blud (sigma between 0 and 1.5)
        sigma = random.random() * 1.5
        noise = np.clip(sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*sigma, 2*sigma)
        image = image + noise
        return image, label

    def _flip(self, image, label):
        # Random H flip
        if random.random() > 0.5:
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
        return image, label

    def _centercrop(self, image, label):
        output_size = (112, 112, 80)
        # pad the sample if necessary
        if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= \
                output_size[2]:
            pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - output_size[0]) / 2.))
        h1 = int(round((h - output_size[1]) / 2.))
        d1 = int(round((d - output_size[2]) / 2.))

        label = label[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
        image = image[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]

        if (np.max(label)) > 1:
            print(np.max(label))

        return image, label

    # def _resize(self, image, label, bigger_side_to_base_size=True):
    #     if isinstance(self.base_size, int):
    #         h, w, _ = image.shape
    #         if self.scale:
    #             longside = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
    #             #longside = random.randint(int(self.base_size*0.5), int(self.base_size*1))
    #         else:
    #             longside = self.base_size
    #
    #         if bigger_side_to_base_size:
    #             h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
    #         else:
    #             h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h < w else (int(1.0 * longside * h / w + 0.5), longside)
    #         image = np.asarray(Image.fromarray(np.uint8(image)).resize((w, h), Image.BICUBIC))
    #         label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
    #         return image, label

        # elif (isinstance(self.base_size, list) or isinstance(self.base_size, tuple)) and len(self.base_size) == 2:
        #     h, w, _ = image.shape
        #     if self.scale:
        #         scale = random.random() * 1.5 + 0.5 # Scaling between [0.5, 2]
        #         h, w = int(self.base_size[0] * scale), int(self.base_size[1] * scale)
        #     else:
        #         h, w = self.base_size
        #     image = np.asarray(Image.fromarray(np.uint8(image)).resize((w, h), Image.BICUBIC))
        #     label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        #     return image, label
        #
        # else:
        #     raise ValueError

    def _val_augmentation(self, image, label):
        # if self.base_size is not None:
            # image, label = self._resize(image, label)
        # image, label = self._crop(image, label)
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)

        image = torch.from_numpy(image.astype(np.float32))
        return image, label

    def _augmentation(self, image, label):
        h, w, d = image.shape

        # if self.base_size is not None:
        #     image, label = self._resize(image, label)

#        if self.crop_size is not None:
#            image, label = self._crop(image, label)

        # if self.flip:
        #     image, label = self._flip(image, label)
        #
        # if self.rotate:
        #     image, label = self._rotate(image, label)

        # if self.blur:
        #     image, label = self._blur(image, label)

        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)

        return torch.from_numpy(image.astype(np.float32)), label

    def __len__(self):
        return len(self.files)
  
    def __getitem__(self, index):
        image, label, image_id =  self._load_data(index)
        if self.val:
            image, label = self._val_augmentation(image, label)
        elif self.augment:
            image, label = self._augmentation(image, label)

        # print (np.max(label))

        label = torch.from_numpy(np.array(label, dtype=np.int8)).long()
        return image, label, image_id

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str

