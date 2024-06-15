import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from a2_image2image import Image2ImageDataset
from models.gan_cls import generator,discriminator
from utils import Utils

class Trainer(object):
    def __init__(self, gparam, dparam,lr, l1_coef, l2_coef, save_path, data_path,batch_size, num_workers, epochs):
        # 从头开始训练
        self.generator = torch.nn.DataParallel(generator().cuda())
        self.discriminator = torch.nn.DataParallel(discriminator().cuda())
        # 从断点开始训练
        self.gparam = gparam
        self.dparam = dparam
        pathG = './checkpoints/save__snowed/gen_{}.pth'.format(self.gparam)
        pathD = './checkpoints/save__snowed/disc_{}.pth'.format(self.dparam)
        self.generator.load_state_dict(torch.load(pathG))
        self.discriminator.load_state_dict(torch.load(pathD))

        self.dataset = Image2ImageDataset(data_path)
        # 重要参数
        self.noise_dim = 100
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.beta1 = 0.5
        self.num_epochs = epochs
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        # 利用dataloader加载dataset，返回batch个四元组
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        # 定义优化器，迭代时与backward()方法配合更新参数
        self.optimD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        # 保存训练参数日志
        self.checkpoints_path = './checkpoints'
        self.save_path = save_path

    def _train_gan(self):
        criterion = nn.BCELoss()
        # BCELoss是Binary CrossEntropyLoss的缩写，nn.BCELoss()为二分类交叉熵损失函数，求出来是标量
        # 只能解决二分类问题。 在使用nn.BCELoss()作为损失函数时，需要在该层前面加上Sigmoid函数，一般使用nn.Sigmoid()即可
        l2_loss = nn.MSELoss()
        # nn.MSELoss均方损失函数,平方的差 累加 ，再除以样本数
        l1_loss = nn.L1Loss()
        # 将样本所有元素与标签求差的绝对值再平均
        iteration = 0
        # 迭代次数
        # for epoch in range(self.num_epochs):
        for epoch in range(self.gparam +1,self.gparam + self.num_epochs):
            #if epoch > self.param+20:
             #   self.discriminator.load_state_dict(torch.load('./checkpoints/save__forest/disc_{}.pth'.format(self.param+20)))
            for sample in self.data_loader:
                iteration += 1
                print("第{}次迭代".format(iteration))
                '''
                这里需要注意，因为dala_loader在设置的时候batch_size是64，所以这里的sample可以看成是64个sample字典组成的列表
                (实际上数据类型还是字典)
                这样传进鉴别器的就是一批数据而不是一个数据
                '''
                # 用Variable这个“篮子”把tensor这个“鸡蛋”装起来
                # 图片
                right_images = sample['right_images']
                right_images = Variable(right_images.float()).cuda()
                wrong_images = sample['wrong_images']
                wrong_images = Variable(wrong_images.float()).cuda()
                # 文本和句子编码
                right_embed = sample['right_embed']
                right_embed = Variable(right_embed.float()).cuda()
                # 标签
                real_labels = torch.ones(right_images.size(0))
                real_labels = Variable(real_labels).cuda()
                fake_labels = torch.zeros(right_images.size(0))
                fake_labels = Variable(fake_labels).cuda()
                smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.cpu().numpy(), -0.2))
                smoothed_real_labels = Variable(smoothed_real_labels).cuda()
                # 平滑标签,这里是为了防止鉴别器的性能压倒性的高于生成器，因此在鉴别器过于自信的时候，对鉴别器施加惩罚，0928将-0.1改为-0.25
                # 在label smoothing中有个参数epsilon，描述了将标签软化的程度，
                # 该值越大，经过label smoothing后的标签向量的标签概率值越小，标签越平滑，

                # 训练鉴别器，清空过往梯度
                self.discriminator.zero_grad()
                # 1、真实图片和真实标签训练鉴别器
                outputs, activation_real = self.discriminator(right_images, right_embed)
                real_score = outputs
                real_loss = criterion(outputs, smoothed_real_labels)
                # 2、使用虚假图片和0标签训练鉴别器
                # 注意：生成器由于先经过全连接层，输入是二维张量
                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, _ = self.discriminator(fake_images, right_embed)
                fake_score = outputs
                fake_loss = criterion(outputs, fake_labels)
                # 梯度更新，鉴别器训练完毕
                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.optimD.step()

                # 训练生成器，清空梯度
                self.generator.zero_grad()
                # 1、为了提高对鉴别器的欺骗性，计算鉴别器的output与真实图片的loss
                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, activation_fake = self.discriminator(fake_images, right_embed)
                _, activation_real = self.discriminator(right_images, right_embed)

                # 2、为了提升和真实图片的接近程度，根据activation计算loss
                # activation_fake是二维张量，通过mean按列求平均值，输出形状为[1,n]的张量
                # [batch,512,4,4]，求平均值之后得到的张量维度是[512,4,4]
                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)

                g_loss = criterion(outputs, real_labels) \
                         + self.l2_coef * l2_loss(activation_fake, activation_real.detach()) \
                         + self.l1_coef * l1_loss(fake_images, right_images)
                # 更新网络参数
                g_loss.backward()
                self.optimG.step()

                if iteration % 5 == 0:
                    #self.logger.log_iteration_gan(epoch, d_loss, g_loss, real_score, fake_score)
                    print("Epoch: %d, d_loss= %f, g_loss= %f, D(X)= %f, D(G(X))= %f" % (
                        epoch, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(), real_score.data.cpu().mean(),
                        fake_score.data.cpu().mean()))

            if (epoch) % 100 == 0 :
                Utils.save_gen_checkpoint(self.generator, self.checkpoints_path, self.save_path, epoch)
                Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, self.save_path, epoch)
            # checkpoints每十步一个检查点