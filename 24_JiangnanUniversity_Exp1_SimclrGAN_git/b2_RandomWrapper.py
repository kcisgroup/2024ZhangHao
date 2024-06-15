import random
from torchvision.datasets import STL10
import torchvision.transforms.functional as tvf
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
import torch
import h5py
from PIL import Image

def random_rotate(image):
    return tvf.rotate(image, angle=random.choice([0, 90, 180, 270]))

class PretrainingDatasetWrapper(Dataset):
    def __init__(self,datasetFile = './train_snowed.hdf5' , debug=False, simclr_classifier_train = False):
        super().__init__()
        self.datasetFile = datasetFile
        self.split = './train'
        self.dataset = h5py.File(self.datasetFile, mode='r')
        self.dataset_keys = [str(k) for k in self.dataset[self.split].keys()]
        self.debug = debug
        # 如果dataloader用于训练分类器，那么其返回并不是变换后的(t1,t2)，而是batch个未经变换的图片
        self.simclr_classifier_train = simclr_classifier_train

        if debug:
            print("DATASET IN DEBUG MODE")

        self.randomize = transforms.Compose([
            transforms.RandomChoice([
                transforms.Lambda(random_rotate),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.5),
                transforms.ColorJitter(contrast=0.5),
                transforms.ColorJitter(saturation=1.5),
            ])
        ])

    def __len__(self):
        f = h5py.File(self.datasetFile, 'r')
        length = len(f[self.split])
        f.close()
        return length

    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)

    def __getitem_internal__(self, idx, preprocess=True):
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        raw_img = Image.fromarray(np.array(example['img']))

        if self.simclr_classifier_train:
            img = np.array(raw_img)
            img = self.validate_image(img)
            img = torch.FloatTensor(img)
            return (img,np.array(example['img']))

        if self.debug:
            random.seed(idx)
            t1 = self.randomize(raw_img)
            random.seed(idx + 1)
            t2 = self.randomize(raw_img)
        else:
            t1 = self.randomize(raw_img)
            t2 = self.randomize(raw_img)

        t1 = np.array(t1)
        t1 = self.validate_image(t1)
        t2 = np.array(t2)
        t2 = self.validate_image(t2)
        t1 = torch.FloatTensor(t1)
        t2 = torch.FloatTensor(t2)

        return (t1, t2), torch.tensor(0)

    def __getitem__(self, idx):
        return self.__getitem_internal__(idx, True)

    def raw(self, idx):
        return self.__getitem_internal__(idx, False)


