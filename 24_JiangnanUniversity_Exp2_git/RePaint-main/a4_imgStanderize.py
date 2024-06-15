# -*- encoding:utf-8 -*-
"""
@author:zsiming
@fileName:ISODATA.py
@Time:2022/1/9  12:33
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances
from PIL import Image
import pandas as pd
from a1a2_generateFourMask import genertor

class ISODATA():
    def __init__(self, data, n_samples,designCenterNum, LeastSampleNum, StdThred, LeastCenterDist, minIterationNum, maxIterationNum):
        #  指定预期的聚类数、每类的最小样本数、标准差阈值、最小中心距离、迭代次数
        self.K = designCenterNum
        self.thetaN = LeastSampleNum
        self.thetaS = StdThred
        self.thetaC = LeastCenterDist
        self.minIterationNum = minIterationNum
        self.maxIterationNum = maxIterationNum

        # 初始化
        self.n_samples = n_samples
        # 选一
        self.data = data
        self.lables = np.random.randint(low=0, high=3, size=self.n_samples, dtype=int)

        self.center = self.data[0, :].reshape((1, -1))
        self.centerNum = 1
        self.centerMeanDist = 0

        # seaborn风格
        sns.set()

    def updateLabel(self):
        """
            更新中心
        """
        print("开始进行中心点更新。当前中心数为{}，各中心为{}".format(self.centerNum,self.center))
        indexes_size = np.array([])
        zero_center = np.array([], dtype=int)
        for i in range(self.centerNum):
            # 计算样本到中心的距离
            print("第{}个中心开始更新".format(i))
            distance = euclidean_distances(self.data, self.center.reshape((self.centerNum, -1)))
            # 为样本重新分配标签
            self.label = np.argmin(distance, 1)
            # 找出同一类样本
            index = np.argwhere(self.label == i).squeeze()
            print("当前中心样本数为{}".format(index.size))
            indexes_size = np.append(indexes_size, index.size)
            if index.size == 0:
                print("当前中心所管理的类，其样本数为0，因此对其进行标记且不更新，后续删除")
                zero_center = np.append(zero_center, i)
                print("zero_center = {}".format(zero_center))
            else:
                sameClassSample = self.data[index, :]
                # 更新中心
                newCenter = np.mean(sameClassSample, 0)
                print("第{}个中心由{}更新为{}".format(i,self.center[i, :],newCenter))
                self.center[i, :] = newCenter
            ###注意，当分裂后的中心，一个元素都没有时，下面的循环会报错，因此将没有元素的中心删去
        print("本次中心更新后，各中心的样本数为{}，应删去的中心下标为{}".format(indexes_size,zero_center))
        self.center = np.delete(self.center,[zero_center],0)
        self.centerNum = self.centerNum - zero_center.size
        print("删去样本为0的类后，中心数为{}，各中心为{}".format(self.centerNum, self.center))
        # 更新self.lable
        distance = euclidean_distances(self.data, self.center.reshape((self.centerNum, -1)))
        self.label = np.argmin(distance, 1)

        # 计算所有类到各自中心的平均距离之和
        for i in range(self.centerNum):
            # 找出同一类样本
            index = np.argwhere(self.label == i).squeeze()
            sameClassSample = self.data[index, :]
            # 计算样本到中心的距离
            if index.size == 0:
                print("存在某类无元素")
            if index.size == 1:
                print("存在某类仅仅1个元素")
                sameClassSample = sameClassSample.reshape(1,-1)
            distance = np.mean(euclidean_distances(sameClassSample, self.center[i, :].reshape((1, -1))))
            # 更新中心
            self.centerMeanDist += distance
        self.centerMeanDist /= self.centerNum

    def divide(self):
        # 临时保存更新后的中心集合,否则在删除和添加的过程中顺序会乱
        newCenterSet = self.center
        # 计算每个类的样本在每个维度的标准差
        for i in range(self.centerNum):
            # 找出同一类样本
            index = np.argwhere(self.label == i).squeeze()
            sameClassSample = self.data[index, :]
            # 计算样本到中心每个维度的标准差
            stdEachDim = np.mean((sameClassSample - self.center[i, :])**2, axis=0)
            # 找出其中维度的最大标准差
            maxIndex = np.argmax(stdEachDim)
            maxStd = stdEachDim[maxIndex]
            # 计算样本到本类中心的距离
            distance = np.mean(euclidean_distances(sameClassSample, self.center[i, :].reshape((1, -1))))
            # 如果最大标准差超过了阈值
            if maxStd > self.thetaS:
                # 还需要该类的样本数大于于阈值很多 且 太分散才进行分裂
                if self.centerNum <= self.K//2 or \
                        sameClassSample.shape[0] > 2 * (self.thetaN+1) and distance >= self.centerMeanDist:
                    newCenterFirst = self.center[i, :].copy()
                    newCenterSecond = self.center[i, :].copy()

                    newCenterFirst[maxIndex] += 0.5 * maxStd
                    newCenterSecond[maxIndex] -= 0.5 * maxStd

                    # 删除原始中心
                    newCenterSet = np.delete(newCenterSet, i, axis=0)
                    # 添加新中心
                    newCenterSet = np.vstack((newCenterSet, newCenterFirst))
                    newCenterSet = np.vstack((newCenterSet, newCenterSecond))

            else:
                continue
        # 更新中心集合
        self.center = newCenterSet
        self.centerNum = self.center.shape[0]

    def combine(self):
        # 计算中心之间的距离
        centerDist = euclidean_distances(self.center, self.center)
        print("初步计算centerDist为：\n{}".format(pd.DataFrame(np.around(centerDist,1))))
        #将centerDist的对角线元素改为inf，即self.thetaC，使得检测最小中心距离时忽略对角线元素
        inf = self.thetaC
        for i in range(self.centerNum):
            centerDist[i][i] = inf
        # centerDist += (np.eye(self.centerNum)) * 10**10
        print("修改对角线元素后，centerDist为：\n{}".format(pd.DataFrame(np.around(centerDist,1))))
        # 把中心距离小于阈值的中心对找出来

        while True:
            # 临时保存更新后的中心集合,否则在删除和添加的过程中顺序会乱
            delIndexList = []
            # 如果最小的中心距离都大于阈值的话，则不再进行合并
            minDist = np.min(centerDist)
            if minDist >= self.thetaC:
                print("最小中心距离为{}大于阈值，故不进行合并！".format(minDist))
                break
            # 否则合并
            print("最小中心距离为{}小于阈值，故进行合并！".format(minDist))
            index = np.argmin(centerDist)
            row = index // self.centerNum
            col = index % self.centerNum
            # 找出合并的两个类别
            index = np.argwhere(self.label == row).squeeze()
            classNumFirst = len(index)
            index = np.argwhere(self.label == col).squeeze()
            classNumSecond = len(index)
            newCenter = self.center[row, :] * (classNumFirst / (classNumFirst + classNumSecond)) + \
                        self.center[col, :] * (classNumSecond / (classNumFirst + classNumSecond))
            print("现在将中心{}和{}合并为新中心{}".format(self.center[row, :],self.center[col, :],newCenter))
            #更新中心
            self.center = np.vstack((self.center, newCenter))
            delIndexList.append(row)
            delIndexList.append(col)
            self.center = np.delete(self.center, delIndexList, axis=0)
            self.center = self.center.astype(np.uint8)
            #更新中心数
            self.centerNum = self.center.shape[0]
            #更新标签
            distance = euclidean_distances(self.data, self.center.reshape((self.centerNum, -1)))
            # 为样本重新分配标签
            self.label = np.argmin(distance, 1)
            #更新中心距离矩阵
            centerDist = euclidean_distances(self.center, self.center)
            for i in range(self.centerNum):
                centerDist[i][i] = inf
        print("所有合并操作完成！新的中心点为：\n{}\n新的中心距离矩阵为\n{}".format(pd.DataFrame(self.center),pd.DataFrame(centerDist)))
        '''
        while True:
            # 如果最小的中心距离都大于阈值的话，则不再进行合并
            minDist = np.min(centerDist)
            if minDist >= self.thetaC:
                print("最小中心距离为{}大于阈值，故不进行合并！".format(minDist))
                break
            # 否则合并
            print("最小中心距离为{}小于阈值，故进行合并！".format(minDist))
            index = np.argmin(centerDist)
            row = index // self.centerNum
            col = index % self.centerNum
            # 找出合并的两个类别
            index = np.argwhere(self.label == row).squeeze()
            classNumFirst = len(index)
            index = np.argwhere(self.label == col).squeeze()
            classNumSecond = len(index)
            newCenter = self.center[row, :] * (classNumFirst / (classNumFirst+ classNumSecond)) + \
                        self.center[col, :] * (classNumSecond / (classNumFirst+ classNumSecond))
            # 记录被合并的中心
            delIndexList.append(row)
            delIndexList.append(col)
            # 增加合并后的中心
            self.center = np.vstack((self.center, newCenter))
            self.centerNum += 1
            # 标记，以防下次选中
            #这里代码写的问题非常大，首先经过本次while循环后,距离矩阵没有马上更新，很多地方空着inf，这导致如果有很多中心点对
            #距离小于阈值，那么centerDist的尺寸会非常大，第二个循环的index会越来越大，但是centerNum却不变，从而导致越界
            print(centerDist)
            print(centerDist.shape)
            centerDist[row, :] = float("inf")
            centerDist[col, :] = float("inf")
            centerDist[:, col] = float("inf")
            centerDist[:, row] = float("inf")
            print("更新后的距离矩阵为{}".format(centerDist))
            '''
        '''
        # 更新中心
        self.center = np.delete(self.center, delIndexList, axis=0)

        self.center = self.center.astype(np.uint8)
        self.centerNum = self.center.shape[0]
        print("更新后的中心为：{}".format(self.center))
        '''
    def drawResult(self):
        ax = plt.gca()
        ax.clear()
        ax.scatter(self.data[:, 0], self.data[:, 1], c=self.label, cmap="cool")
        # ax.set_aspect(1)
        # 坐标信息
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        plt.show()


    def train(self):
        # 初始化中心和label
        self.updateLabel()
        # self.drawResult()

        # 到设定的次数自动退出
        i = 0
        while True:
        # for i in range(self.iteration):
            if self.K >= 50 and self.centerNum < self.K+10  and self.centerNum >= self.K - 30:
                print("循环数大于等于最小循环数，且中心数等于预设中心数")
                break
            if self.K >= 30 and self.centerNum < self.K+10  and self.centerNum >= self.K - 20:
                print("循环数大于等于最小循环数，且中心数等于预设中心数")
                break
            if self.K >= 10 and self.centerNum < self.K+10  and self.centerNum >= self.K - 6:
                print("循环数大于等于最小循环数，且中心数等于预设中心数")
                break
            if i >= self.minIterationNum and self.centerNum < self.K+10  and self.centerNum >= self.K - 6:
                print("循环数大于等于最小循环数，且中心数等于预设中心数")
                break
            elif i > self.maxIterationNum:
                print("循环数超过最大循环数")
                break
            # 如果是偶数次迭代或者中心的数量太多，那么进行合并
            if self.centerNum < self.K //2:
                print("迭代次数为{},中心数过小，执行分裂".format(i))
                self.divide()
                print("分裂完成")
            elif (i > 0 and i % 2 == 0) or self.centerNum > 2 * self.K:
                print("迭代次数为{},中心数过多或为偶数次迭代，执行合并".format(i))
                self.combine()
                print("合并完成")
            else:
                print("迭代次数为{},执行分裂".format(i))
                self.divide()
            i = i + 1
            # 更新中心
            print("迭代次数为{},中心数为{}，执行中心更新".format(i,self.centerNum))
            self.updateLabel()
            print("中心更新完成")
            # self.drawResult()
            print("中心数量：{}".format(self.centerNum))
        # self.drawResult()
        return self.label,self.center

class standerize():
    def __init__(self, imgPath, maskPath, colorNum, savePath):
        self.imgPath = imgPath
        self.maskPath = maskPath
        self.colorNum = colorNum
        self.data = np.array(Image.open(imgPath))
        self.data = self.data.reshape((-1,3))
        self.n_samples = len(self.data)
        self.savePath = savePath

        img_gt = Image.open(self.imgPath)
        self.np_gt = np.array(img_gt)

        img_mask = Image.open(self.maskPath)
        img_mask = img_mask.convert('L')
        self.np_mask = np.array(img_mask)

        #注意，np.append无法将元素添加到空数组，且其速度非常慢

        # for i in range(256):
        #     for j in range(256):
        #         if self.np_mask[i][j] == 0:
        #             self.data = np.append(self.data,self.np_gt[i][j])
        # self.data = self.data.reshape((-1, 3))
        # self.n_samples = len(self.data)

        self.isodata = ISODATA(data = self.data,n_samples = self.n_samples,designCenterNum=self.colorNum, LeastSampleNum=2, StdThred=0.2, LeastCenterDist=50, minIterationNum=20, maxIterationNum = 20)

    def standerize(self):
        lable, center = self.isodata.train()
        lable, center = lable.astype(int), center.astype(int)
        for i in range(self.n_samples):
            self.data[i] = center[lable[i]]
        self.data = self.data.reshape(256, 256, 3)

        for i in range(256):
            for j in range(256):
                if self.np_mask[i][j] == 0:
                    self.np_gt[i][j] = self.data[i][j]

        # print("k的大小为{},像素点数为{}".format(k,self.data.shape[0]))
        result = self.np_gt.reshape(256, 256, 3)
        result = Image.fromarray(result, mode="RGB")
        result.save(self.savePath)

# x = standerize('./.inpainted.png','./.maskForStanderize/rt.png',20, './')
# x.standerize()

# data = np.array(Image.open('./.inpainted.png'))
# data = data.reshape((-1,3))
# n_samples = len(data)
'''
if __name__ == "__main__":
    isoData = ISODATA(data = data,n_samples = n_samples,designCenterNum=64, LeastSampleNum=2, StdThred=0.2, LeastCenterDist=50, minIterationNum=20, maxIterationNum = 100)
    lable,center = isoData.train()
    lable,center = lable.astype(int),center.astype(int)
    print(data.shape)
    print(lable.shape)
    print(center.shape)
    for i in range(65536):
        data[i] = center[lable[i]]
    data = data.reshape(256,256,3)
    data = Image.fromarray(data,mode="RGB")
    data.save('./reslt4.png')
'''

