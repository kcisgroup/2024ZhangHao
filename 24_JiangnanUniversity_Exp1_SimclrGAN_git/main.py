import os
import h5py
import torch
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from a2_image2image import Image2ImageDataset
from models.gan_cls import generator,discriminator
from torch.autograd import Variable
'''
datasetFile = './train_forest.hdf5'
dataset = h5py.File(datasetFile, mode='r')
dataset_keys = [str(k) for k in dataset['train'].keys()]
example_name = dataset_keys[0]
example = dataset['train'][example_name]

name = example['name']
img = example['img']
txt = example['txt']
embed = example['embeddings']
print(torch.FloatTensor(np.array(embed)).size())
img = np.array(img, dtype=float)
img = torch.FloatTensor(img)
# print((img).size())
# print(np.array(txt))
# print(np.array(name))
'''

dataset = Image2ImageDataset('./train_forest.hdf5')
data_loader = DataLoader(dataset, batch_size=6, shuffle=True,num_workers=0)

generator = torch.nn.DataParallel(generator().cuda())
discriminator = torch.nn.DataParallel(discriminator().cuda())
noise = Variable(torch.randn(6, 1024)).cuda()
# fake_images = generator(noise)
# print(fake_images.size())

for sample in data_loader:
    right_images = sample['right_images']
    right_images = Variable(right_images.float()).cuda()
    right_embed = sample['right_embed']
    right_embed = Variable(right_embed.float()).cuda()
    noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
    noise = noise.view(noise.size(0), 100, 1, 1)
    print(right_images.size())
    fake_images = generator(right_embed, noise)
    outputs, activation_real = discriminator(right_images, right_embed)

