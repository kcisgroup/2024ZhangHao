import torch
import torch.nn as nn
from utils import Concat_embed

class generator(nn.Module):
	def __init__(self):
		super(generator, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.noise_dim = 100
		self.embed_dim = 1024
		self.projected_embed_dim = 128
		self.latent_dim = self.noise_dim + self.projected_embed_dim
		self.ngf = 64

		#注意，全连接层仅接受二维数据，因此，噪声数据的输入必须是二维[1024,1]才可以
		#然后通过unsqueeze()提升维度形成四维向量，再输入到Conv2d网络中
		self.projection = nn.Sequential(
			nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
			#全连接层1024维映射到128维
			nn.BatchNorm1d(num_features=self.projected_embed_dim),
			nn.LeakyReLU(negative_slope=0.2, inplace=True)
			)


		self.netG = nn.Sequential(

			nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(self.ngf * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 4 x 4

			nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf * 4),
			nn.ReLU(True),
			# state size. (ngf*4) x 8 x 8

			nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf * 2),
			nn.ReLU(True),
			# state size. (ngf*2) x 16 x 16

			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=None),
			nn.Conv2d(in_channels=self.ngf * 2, out_channels=self.ngf, kernel_size=(3, 3), stride=(1, 1),
					  padding=(1, 1), bias=False),
			nn.BatchNorm2d(self.ngf),
			nn.ReLU(True),
			# state size. (ngf) x 32 x 32

			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=None),
			nn.Conv2d(in_channels=self.ngf, out_channels=self.num_channels, kernel_size=(3, 3), stride=(1, 1),
					   padding=(1, 1),bias=False),
			nn.Tanh()
			 # state size. (num_channels) x 64 x 64
			)

	def forward(self,embed_vector,z):
		projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(3)
		latent_vector = torch.cat([projected_embed,z],1)
		output = self.netG(latent_vector)
		return output

class discriminator(nn.Module):
	def __init__(self):
		super(discriminator, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.embed_dim = 1024
		self.projected_embed_dim = 128
		self.ndf = 64

		self.netD_1 = nn.Sequential(
			# input is (nc) x 64 x 64，输入是3*64*64
			nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 4 x 4
		)

		self.projector = Concat_embed(self.embed_dim,self.projected_embed_dim)

		self.netD_2 = nn.Sequential(
			# state size. (ndf*8) x 4 x 4
			nn.Conv2d(self.ndf * 8 + self.projected_embed_dim, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
			)	

	def forward(self, inp, embed):
		x_intermediate = self.netD_1(inp)
		x = self.projector(x_intermediate, embed)
		x = self.netD_2(x)
		return x.view(-1, 1).squeeze(1) , x_intermediate