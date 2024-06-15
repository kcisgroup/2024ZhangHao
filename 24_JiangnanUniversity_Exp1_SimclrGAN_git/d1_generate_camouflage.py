import numpy as np
import torch
from torch.autograd import Variable
from models.gan_cls import generator
from PIL import Image , ImageEnhance
from a3_text2vec import txt2vec
# from tif_to_camouflage import Tif_to_Camouflage
from c1_SimCLRClassifier import SimCLRClassifier
from c2_SimCLRClassifier_Camouflage import  Camouflage

class model_to_png():
    def __init__(self, simclr_path = './checkpoints/save_embedding_snowed/simclr_100.pth', simclr_classifier_path = './checkpoints/save_classifier_snowed/simclr_classifier_100.pth'):
        self.GEN_path = 'checkpoints/save__snowed/gen_1001.pth'
        self.modelG = torch.nn.DataParallel(generator())
        self.modelG.load_state_dict(torch.load(self.GEN_path))

        self.simclr_path = simclr_path
        self.simclr_classifier_path = simclr_classifier_path
        self.simclr_classifier = SimCLRClassifier(n_classes = 16, freeze_base = True,
                                                  embeddings_model_path = self.simclr_path)
        self.simclr_classifier.load_state_dict(torch.load(self.simclr_classifier_path))

        self.camo = Camouflage()

    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)

    def convert_to_topView(self, img_num = 1):
        self.modelG.eval()
        # 接受输入 转化为词向量
        sentence = input("请输入描述：")
        sentence = sentence.split()
        txt_to_vec = txt2vec()
        embedding = txt_to_vec.build_sentence_vector(sentence=sentence)
        embedding = np.array(embedding, dtype=float).squeeze(0)
        # 升维至(n,1024)喂给modelG
        embeddings = []
        for i in range(img_num):
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        embeddings = torch.FloatTensor(embeddings)
        embeddings = Variable(embeddings.float()).cuda()

        noise = Variable(torch.randn(img_num, 100)).cuda()
        noise = noise.view(img_num, 100, 1, 1)
        # noise = Variable(torch.randn(img_num, 100)).cuda()
        # noise = noise.view(img_num, 100, 1, 1)

        fake_images = self.modelG(embeddings, noise)

        for i in range(img_num):
            # 以下是用颜色编码方式进行迷彩化
            raw_img = Image.fromarray(fake_images[i].data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())

            simclr_img = np.array(raw_img)
            simclr_img = self.validate_image(simclr_img)
            # 将simclr_img变四维
            simclr_img = torch.FloatTensor(simclr_img).unsqueeze(0)
            # results维度[batch,16]
            results = self.simclr_classifier(simclr_img)

            # dominant_colors = self.camo.get_dominant_colors(raw_img)
            camouflage_img = self.camo.get_dominantColorReplaced_imgs(raw_img,results)
            # print(camouflage_img.shape)
            camouflage_img = Image.fromarray(camouflage_img.astype('uint8')).convert('RGB')
            camouflage_img.show()
            camouflage_img.save('./d_result/topViewSnowed_{}.png'.format(i))


            '''
            # 以下是用主色提取方式进行迷彩化
            raw_img = Image.fromarray(fake_images[i].data_forest.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
            camouflage_img = self.camo.get_dominantColorReplaced_imgs_two(raw_img)
            # print(camouflage_img.shape)
            camouflage_img = Image.fromarray(camouflage_img.astype('uint8')).convert('RGB')
            camouflage_img.show()
            camouflage_img.save__forest('./save_result_raw/topView_{}.png'.format(i))
            '''

            '''
            raw_img = Image.fromarray(fake_images[i].data_forest.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())

            img = np.array(raw_img)
            img = self.validate_image(img)
            img = torch.FloatTensor(img).unsqueeze(0)
            results = self.simclr_classifier(img)
            # camouflages = self.camo.get_colorReplaced_imgs(torch.from_numpy(np.array(raw_img)).unsqueeze(0), results, 1)
            raw_img = Image.fromarray(fake_images[i].data_forest.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())


            # 显示图片
            camouflages = np.array(camouflages)
            camouflages = camouflages.squeeze(0)
            print(camouflages.shape)
            camouflages = Image.fromarray(camouflages.astype('uint8')).convert('RGB')
            # camouflages = Image.fromarray(camouflages)
            camouflages.show()
            return

            #img.save__forest('./save_result_raw/topView_{}.png'.format(i))
            '''
model = model_to_png()
model.convert_to_topView(2)