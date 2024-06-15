import os

import numpy as np
import torch
from torch.autograd import Variable
from models.gan_cls import generator
from PIL import Image , ImageEnhance
from a3_text2vec import txt2vec
from c1_SimCLRClassifier import SimCLRClassifier
from c2_SimCLRClassifier_Camouflage import  Camouflage
import sys

class model_to_png():
    def __init__(self, img_num):
        self.img_num = img_num

    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)

    def main(self):
        sentence = input("请输入描述：")
        sentence_mode, sentence_main = sentence.split(':')[0],sentence.split(':')[1]
        if sentence_mode == 'forest':
            self.mode = 'forest'
        elif sentence_mode == 'desert':
            self.mode = 'desert'
        elif sentence_mode == 'rivers':
            self.mode = 'rivers'
        elif sentence_mode == 'snowed':
            self.mode = 'snowed'
        else:
            print("输入非法！请输入正确的环境模式！")
            return

        self.GEN_path = './checkpoints/save__{}/gen_1000.pth'.format(self.mode)
        self.modelG = torch.nn.DataParallel(generator())
        self.modelG.load_state_dict(torch.load(self.GEN_path))
        self.modelG.eval()

        self.simclr_path = './checkpoints/save_embedding_{}/simclr_100.pth'.format(self.mode)
        self.simclr_classifier_path = './checkpoints/save_classifier_{}/simclr_classifier_100.pth'.format(self.mode)
        self.simclr_classifier = SimCLRClassifier(n_classes=16, freeze_base=True,
                                                  embeddings_model_path=self.simclr_path)
        self.simclr_classifier.load_state_dict(torch.load(self.simclr_classifier_path))

        self.camo = Camouflage(self.mode)

        # 接受输入 转化为词向量
        txt_to_vec = txt2vec(path='./text_{}.txt'.format(self.mode), save_name='./word2vec_{}.txt'.format(self.mode))
        embedding = txt_to_vec.build_sentence_vector(sentence=sentence_main)
        embedding = np.array(embedding, dtype=float).squeeze(0)
        # 升维至(n,1024)喂给modelG
        embeddings = []
        for i in range(self.img_num):
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        embeddings = torch.FloatTensor(embeddings)
        embeddings = Variable(embeddings.float()).cuda()

        noise = Variable(torch.randn(self.img_num, 100)).cuda()
        noise = noise.view(self.img_num, 100, 1, 1)

        fake_images = self.modelG(embeddings, noise)

        for i in range(self.img_num):
            # 以下是用颜色编码方式进行迷彩化
            raw_img = Image.fromarray(fake_images[i].data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())

            simclr_img = np.array(raw_img)
            simclr_img = self.validate_image(simclr_img)
            # 将simclr_img变四维
            simclr_img = torch.FloatTensor(simclr_img).unsqueeze(0)
            # results维度[batch,16]
            results = self.simclr_classifier(simclr_img)

            camouflage_img = self.camo.get_dominantColorReplaced_imgs(raw_img, results)
            camouflage_img = Image.fromarray(camouflage_img.astype('uint8')).convert('RGB')
            camouflage_img.show()
            camouflage_img.save('./d_result/topView{}_{}.png'.format(self.mode,i))

txt = [
    "forest:a dense lush arbor forest overlooking",
    "forest:sparse shrub forest",
    "desert:scorching expansive lightyellow desert",
    "desert:lightyellow sandy expansive dunes",
    "rivers:shallow rippling river",
    "rivers:swift murky river",
    "snowed:calm snow-covered snowfield",
    "snowed:frozen serene snowfield"
]
model = model_to_png(img_num = 1)
model.main()