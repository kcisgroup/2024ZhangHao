import os
import sys

import torch
from torch import nn
from b3_ImageEmbedding import ImageEmbedding

class SimCLRClassifier(nn.Module):
    def __init__(self, n_classes, freeze_base, embeddings_model_path):
        super().__init__()
        '''
        模型通过torch.save_Imbedding(self.simclr.state_dict())实现，加载模型时使用modelG.load_state_dict
        '''
        base_model = ImageEmbedding()
        base_model.load_state_dict(torch.load(embeddings_model_path))
        # self.embeddings是efficientnet-b0网络

        self.embeddings = base_model.embedding
        # 如果 freeze_base = True，那么训练分类器时冻结simclr的embedding部分参数
        # 只有simclr的projection部分得到更新
        if freeze_base:
            # print("Freezing embeddings")
            for param in self.embeddings.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(in_features=base_model.projection[0].in_features,
                                    out_features=n_classes)

    def forward(self, X):
        emb = self.embeddings(X)
        return self.classifier(emb)