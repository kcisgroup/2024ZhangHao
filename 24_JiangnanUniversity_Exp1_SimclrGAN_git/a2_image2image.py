from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import torch

class Image2ImageDataset(Dataset):
    #继承Dataset类重新写一个Dataset
    def __init__(self, datasetFile = './train_snowed.hdf5'):
        self.datasetFile = datasetFile
        #这里的datasetfile是h5py文件的路径，即datasetFile = './camouflage.hdf5'
        self.split = 'train'
        self.dataset = h5py.File(self.datasetFile, mode='r')
        self.dataset_keys = [str(k) for k in self.dataset[self.split].keys()]

    def __len__(self):
        #Python中存在一些特殊的方法，有些方法以双下划线 “__” 开头和结尾，它们是Python的魔法函数
        f = h5py.File(self.datasetFile, 'r')
        length = len(f[self.split])
        f.close()
        return length

    def __getitem__(self, idx):
        #当一个类中定义了__getitem__方法，那么它的实例对象便拥有了通过下标来索引的能力
        #比如A类定义了，那么A【2】就自动执行检索功能
        if self.dataset is None:
            self.dataset = h5py.File(self.datasetFile, mode='r')

        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        # example = self.dataset[self.split][idx]
        name = example['name']
        img = example['img']

        right_image = img
        wrong_image = self.find_wrong_image(idx)
        #返回一张不同类的图片
        right_image = self.validate_image(right_image)
        #维度从image格式的（64，64，3）变成numpy格式的（3，64，64）
        wrong_image = self.validate_image(wrong_image)
        right_embed = np.array(example['embeddings'], dtype=float).squeeze(0)#(1,1024)降到(1024)
        inter_embed = np.array(self.find_inter_embed(idx)).squeeze(0)
        txt = np.array(example['txt']).astype(str)

        sample = {
                'name': idx,
                'right_images': torch.FloatTensor(right_image),
                'right_embed': torch.FloatTensor(right_embed),
                'wrong_images': torch.FloatTensor(wrong_image),
                'inter_embed': torch.FloatTensor(inter_embed),
                'txt': str(txt)
        }

        sample['right_images'] = sample['right_images'].sub_(127.5).div_(127.5)
        sample['wrong_images'] = sample['wrong_images'].sub_(127.5).div_(127.5)
        return sample

    def find_wrong_image(self, idx):
        f = h5py.File(self.datasetFile, 'r')

        wrong_idx = np.random.randint(len(f[self.split]))
        wrong_example_name = self.dataset_keys[wrong_idx]
        example = self.dataset[self.split][wrong_example_name]

        if idx != wrong_idx:
            return example['img']

        return self.find_wrong_image(wrong_idx)

    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)

    def find_inter_embed(self,idx):
        wrong_idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[wrong_idx]
        example = self.dataset[self.split][example_name]
        embeddings = example['embeddings']

        if wrong_idx != idx:
            return embeddings

        return self.find_inter_embed(wrong_idx)




