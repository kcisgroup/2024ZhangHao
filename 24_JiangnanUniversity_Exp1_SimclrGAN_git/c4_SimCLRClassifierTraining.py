# SimclrClassifierTrainer.py
# 将[训练好的]simclr网络模型与分类器结合起来，将图片集作为输入，得到分类结果后进行颜色替换，然后根据loss更新参数
import torch
from torch.optim import RMSprop
from c1_SimCLRClassifier import SimCLRClassifier
from c3_SimCLRClassifierLoss import SSIM
from torch.utils.data import DataLoader
from b2_RandomWrapper import PretrainingDatasetWrapper
from c2_SimCLRClassifier_Camouflage import Camouflage
import numpy as np

'''
[
[65,96,54],#浅绿
[33,52,50],#深绿
[25,26,31],#黑绿
[202,201,153],#黄绿
[156,153,140],#褐色
[187,181,155],#棕褐色
[78,82,59],#绿褐色
[67,73,63],#深绿色
[98,83,76],#褐色
[44,39,45],#深灰色
[48,82,84],#深绿色
[165,170,116],#橄榄绿
[88.98,73],#深橄榄绿
[71,74,57],#褐绿色
[58,114,79],#深亮绿色
[210,220,211]#浅浅绿
]
'''
class SimCLRClassifierTrainer(object):
    def __init__(self, datasetFile, batch_size, n_classes, freeze_base, simclr_model_path, epoch, lr = 1e-3):
        super().__init__()

        self.dataset = PretrainingDatasetWrapper(datasetFile, simclr_classifier_train = True)
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        self.batch = batch_size
        self.n_classes = n_classes
        self.freeze_base = freeze_base
        self.simclr_model_path = simclr_model_path
        self.simclr_classifier = SimCLRClassifier(self.n_classes, self.freeze_base,
                                      self.simclr_model_path)

        self.camo = Camouflage()

        self.loss = SSIM()
        self.eopch = epoch
        self.lr = lr
        self.optimizer = RMSprop(self.simclr_classifier.parameters(), lr=self.lr)

        self.checkpoints_path = './checkpoints/save_classifier_snowed'



    #def camouflage(self,sample,result):

    def save_checkpoint(self,epoch):

        torch.save(self.simclr_classifier.state_dict(), '{0}/simclr_classifier_{1}.pth'.format(self.checkpoints_path, epoch))

    def train_simclr_classifier(self):
        iteration = 0
        for epoch in range(self.eopch):
            for sample in self.data_loader:
                # 将raw_img转化为迷彩图片camouflage_img，将其与raw_img进行比较得到损失，进行参数更新
                imgs, raw_imgs = sample
                iteration += 1
                results = self.simclr_classifier(imgs)
                camouflages = self.camo.get_colorReplaced_imgs(raw_imgs,results,self.batch)


                # 接下来计算损失，raw_imgs类型是np，camouflages是形状为(6，64，64，3的图片)
                camouflages = torch.tensor(np.array(camouflages))

                # camouflages : <class 'list'> (batch, 64, 64, 3)

                # 接下来，计算camouflage与raw_img的loss，前者类型是list，后者是tensor

                raw_imgs = raw_imgs.float()
                camouflages = camouflages.float()
                # 将raw_imgs与camouflages归一化
                raw_imgs = torch.nn.functional.normalize(raw_imgs)
                camouflages = torch.nn.functional.normalize(camouflages)

                loss = self.loss(raw_imgs, camouflages)
                loss.requires_grad_(True)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if iteration % 5 == 0:
                    print("Epoch: %d, loss= %f" % (
                        epoch, loss.data.cpu().mean()))

            if epoch % 10 == 0:
                self.save_checkpoint(epoch)

trainer = SimCLRClassifierTrainer(datasetFile = './train_snowed.hdf5', batch_size = 6, n_classes = 16, freeze_base = True, simclr_model_path ='./checkpoints/save_embedding_snowed/simclr_100.pth', epoch = 101, lr = 1e-3)

trainer.train_simclr_classifier()