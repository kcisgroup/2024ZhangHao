# b4_ImageEmbeddingTraining.py
'''
训练simclr网络，并根据损失更新网络参数
'''
import torch
from b1_ContrastiveLoss import ContrastiveLoss
from torch.utils.data import DataLoader
from b3_ImageEmbedding import ImageEmbedding
from b2_RandomWrapper import PretrainingDatasetWrapper

class Trainer(object):
    def __init__(self,datasetFile,batch_size,embedding_size,initial_Lr,checkpoints_path,epochs):
        self.dataset = PretrainingDatasetWrapper(datasetFile)
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.simclr = ImageEmbedding(embedding_size)
        self.lossFunction = ContrastiveLoss(batch_size = batch_size)
        self.optimizer = torch.optim.Adam(self.simclr.parameters(), lr=initial_Lr, weight_decay=1e-6)

        # 保存训练参数日志
        self.checkpoints_path = checkpoints_path
        self.epoch = epochs

    def save_checkpoint(self,epoch):

        torch.save(self.simclr.state_dict(), '{0}/simclr_{1}.pth'.format(self.checkpoints_path, epoch))

    def _train_simclr(self):
        iteration = 0
        for epoch in range(self.epoch):
            for sample in self.data_loader:
                iteration += 1
                (X,Y),_ = sample
                embX, projectionX = self.simclr(X)
                embY, projectionY = self.simclr(Y)
                loss = self.lossFunction(projectionX,projectionY)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if iteration % 5 == 0:
                    print("Epoch: %d, loss= %f" % (
                        epoch, loss.data.cpu().mean()))

            if epoch % 10 == 0:
                self.save_checkpoint(epoch)
        return