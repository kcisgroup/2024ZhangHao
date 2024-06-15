'''
本模块的功能：
每次训练图像嵌入模块时，将得到的两批向量输入给此模块
该模块可以计算出两组向量的总损失。设batch为N，相似图像对应向量为emb_i和emb_j，
即可计算出emb_i与其他2N-1个emb的损失。损失最小为1。
再将所有损失叠加取均值
'''
import torch
from torch import nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        '''
        计算相似度矩阵
        '''
        emb_i = F.normalize(emb_i, dim=1)
        emb_j = F.normalize(emb_j, dim=1)
        representations = torch.cat([emb_i, emb_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        '''
        从数据集中提取batch = N个数据后，分别得到它们对应的负样例，组成2N个图像以及嵌入后的向量
        sim_ij和sim_ji是包含了所有正负样例的损失的数组
        '''
        sim_ij = torch.diag(similarity_matrix,  self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positive_losses = torch.cat([sim_ij, sim_ji], dim=0)
        '''
        将反掩码对角矩阵与相似矩阵相乘，得到除sim(i,i)以外的相似矩阵
        将该矩阵按行求和得到分母，根据损失公式计算得到每个向量的损失
        再将每个损失求和取均值，得到该batch的损失
        '''
        nominator = torch.exp(positive_losses / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss