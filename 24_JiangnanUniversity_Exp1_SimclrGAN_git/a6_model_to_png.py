import torch
from torch.autograd import Variable
from models.gan_cls import generator
from PIL import Image
import numpy as np
from scipy.spatial.distance import pdist
from a3_text2vec import txt2vec

class model_to_png():
    def __init__(self, param, img_num = 10):
        self.epoch = param
        self.img_num = img_num

    def get_dominant_colors(self,small_image):
        result = small_image.convert(
            "P", palette=Image.ADAPTIVE, colors=5
        )
        # num个主要颜色的图像
        # 找到主要的颜色
        palette = result.getpalette()
        color_counts = sorted(result.getcolors(), reverse=True)
        colors = list()

        for i in range(5):
            palette_index = color_counts[i][1]
            dominant_color = palette[palette_index * 3: palette_index * 3 + 3]
            colors.append(tuple(dominant_color))
        return [colors[0], colors[2], colors[4]]

    def cosine(self,x, y):
        dist = pdist(np.vstack([x, y]), 'cosine')
        return dist

    def replace_color(self,small_img,colors):
        colors = np.asarray(colors)
        color1, color2, color3 = colors[0], colors[1], colors[2]
        small_img = np.asarray(small_img, dtype=np.double)
        for i in range(64):
            for j in range(64):
                dist1 = self.cosine(small_img[j][i], color1)
                dist2 = self.cosine(small_img[j][i], color2)
                dist3 = self.cosine(small_img[j][i], color3)
                if dist1 <= min(dist2, dist3):
                    small_img[j][i] = color1
                elif dist2 <= min(dist1, dist3):
                    small_img[j][i] = color2
                else:
                    small_img[j][i] = color3
        camouflage_img = Image.fromarray(small_img.astype('uint8')).convert('RGB')
        return camouflage_img

    def convert(self):
        path = 'checkpoints/save__snowed/gen_{}.pth'.format(self.epoch)
        modelG = torch.nn.DataParallel(generator())
        modelG.load_state_dict(torch.load(path))
        modelG.eval()

        # 转化为词向量
        sentence = input("请输入描述：")
        sentence = sentence.split()
        txt_to_vec = txt2vec()
        embedding = txt_to_vec.build_sentence_vector(sentence=sentence)
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
        fake_images = modelG(embeddings, noise)

        for i in range(self.img_num):
            # noise = Variable(torch.randn(1, 1024)).cuda()
            # fake_images = modelG(noise)
            im = Image.fromarray(fake_images[i].data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
            #im.show()
            im.save('./a_result/n{}.png'.format(i))

    def convert2camouflage(self):
        path = 'checkpoints/save__forest/gen_{}.pth'.format(self.epoch)
        modelG = torch.nn.DataParallel(generator())
        modelG.load_state_dict(torch.load(path))
        modelG.eval()

        for i in range(10):
            noise = Variable(torch.randn(1, 1024)).cuda()
            fake_images = modelG(noise)
            im = Image.fromarray(fake_images[0].data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
            im.save('./a_result/reference_snowed{}.png'.format(i))

            colors = self.get_dominant_colors(im)
            camouflage_img = self.replace_color(im, colors)
            camouflage_img.save('./a_result/camouflage{}.png'.format(i))
model = model_to_png(1000,3)
model.convert()