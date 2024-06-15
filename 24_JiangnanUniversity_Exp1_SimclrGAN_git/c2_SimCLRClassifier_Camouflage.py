'''
将[3,64,64]的np类型图片转化为三色迷彩
'''

import numpy as np
from scipy.spatial.distance import pdist
from PIL import Image

class Camouflage:
    def __init__(self, mode):
        self.mode = mode
        # 军用森林迷彩色
        self.colors_forest = [
            [150, 215, 119],[113, 168,  95],[ 79, 127,  67],[158, 184, 127],
            [122, 177,  98],[100, 157,  82],[198, 225, 151],[111, 193, 94],
            [ 68, 140,  34],[ 96, 157,  77],[ 76, 130,  59],[146, 221, 124],
            [116, 176,  98],[119, 192, 107],[195, 221, 151],[131, 167,  96],
        ]
        # 军用沙漠迷彩色
        self.colors_desert = [
            [227, 138, 49], [225, 157, 50], [183, 119, 47],[219, 149, 35],
            [252, 172, 74], [250, 163, 60], [218, 151, 56], [243, 147, 37],
            [216, 158, 85], [222, 157, 57], [220, 153, 63], [200, 147, 44],
            [183, 128, 34], [232, 163, 47], [229, 168, 75], [205, 175, 134],
        ]
        # 军用河流迷彩色
        self.colors_rivers = [
            [64, 94, 66], [80, 108, 84], [78, 102, 81], [49, 87, 62],
            [76, 100, 82], [127, 164, 130], [78, 115, 98], [106, 139, 118],
            [55, 89, 55], [40, 79, 53], [57, 101, 74], [84, 128, 93],
            [82, 136, 105], [87, 136, 107], [105, 135, 97], [101, 125, 89],
        ]
        # 军用雪地迷彩色
        self.colors_snowed = [
            [198, 204, 220], [230, 233, 248], [186, 193, 212], [186, 192, 210],
            [173, 182, 201], [214, 220, 236], [149, 165, 196], [155, 170, 196],
            [131, 147, 175], [183, 198, 229], [136, 162, 185], [177, 202, 224],
            [118, 145, 162], [156, 181, 203], [130, 159, 170], [156, 176, 199],
        ]
        if self.mode == 'forest':
            self.colors = self.colors_forest
        elif self.mode == 'desert':
            self.colors = self.colors_desert
        elif self.mode == 'rivers':
            self.colors = self.colors_rivers
        elif self.mode == 'snowed':
            self.colors = self.colors_snowed

        # self.colors = [
        #     [52, 56, 62],
        #     [36, 37, 47],
        #     [29, 29, 40],
        #
        #     [79, 91, 86],  # 黄绿
        #     [57, 65, 67],  # 褐色
        #     [48, 52, 58],  # 棕褐色
        #
        #     [134, 140, 134],  # 绿褐色
        #     [126, 129, 127],  # 深绿色
        #     [111, 111, 114],  # 褐色
        #
        #     [66, 72, 73],  # 深灰色
        #     [50, 51, 58],  # 深绿色
        #     [41, 41, 50],  # 橄榄绿
        #
        #     [81, 80, 81],  # 深橄榄绿
        #     [69, 66, 71],  # 褐绿色
        #     [62, 57, 65],  # 深亮绿色
        #
        #     [92, 95, 97]
        #     ]
        # self.batch = batch #n
        # self.imgs = imgs #[n,3,64,64]
        # self.results = result #[n,16]

    # 采集batch内所有图片的color，其中colors[i]是第i张图片的三种主色
    def get_camouflage_colors(self, batch, results):
        colors = []
        for i in range(batch):
            result = np.array(results[i].detach())
            idxs = np.argsort(-result)
            color = [self.colors[idxs[0]],self.colors[idxs[1]],self.colors[idxs[2]]]
            #colors[i] = np.asarray(color)
            colors.append(color)
            #colors = np.asarray(colors)
        return colors

    # 提取单张PIL图片的三种主色
    def get_dominant_colors(self,img):
        result = img.convert(
            "P", palette=Image.ADAPTIVE, colors=3
        )
        # num个主要颜色的图像
        # 找到主要的颜色
        palette = result.getpalette()
        color_counts = sorted(result.getcolors(), reverse=False)
        colors = list()

        for i in range(3):
            palette_index = color_counts[i][1]
            dominant_color = palette[palette_index * 3: palette_index * 3 + 3]
            colors.append(tuple(dominant_color))
        mian_color = [list(colors[0]), list(colors[1]), list(colors[2])]
        return mian_color


    def cosine(self,x, y):
        try:
            dist = pdist(np.vstack([x, y]), 'cosine')
        except Exception:
            print('failed')
            print('x = {0},y = {1}'.format(x,y))
        else:
            return dist

    def get_dominantColorReplaced_imgs(self, img, results):
        # 提取主色
        main_color = self.get_dominant_colors(img)
        main_color = np.asarray(main_color)
        color1, color2, color3 = main_color[0], main_color[1], main_color[2]

        result = np.array(results[0].detach())
        idxs = np.argsort(-result)
        simclr_colors = [self.colors[idxs[0]], self.colors[idxs[1]], self.colors[idxs[2]]]
        simclr_colors = np.asarray(simclr_colors)
        simclr_colors1, simclr_colors2, simclr_colors3 = simclr_colors[0], simclr_colors[1], simclr_colors[2]
        # print(simclr_colors)
        img = np.asarray(img, dtype=np.double)
        for i in range(64):
            for j in range(64):
                dist1 = self.cosine(img[j][i], color1)
                dist2 = self.cosine(img[j][i], color2)
                dist3 = self.cosine(img[j][i], color3)
                if dist1 <= min(dist2, dist3):
                    img[j][i] = simclr_colors1
                elif dist2 <= min(dist1, dist3):
                    img[j][i] = simclr_colors2
                else:
                    img[j][i] = simclr_colors3

            # camouflage_img = Image.fromarray(small_img.astype('uint8')).convert('RGB')
        return img

    def get_dominantColorReplaced_imgs_two(self, img):
        main_color = self.get_dominant_colors(img)
        main_color = np.asarray(main_color)
        color1, color2, color3 = main_color[0], main_color[1], main_color[2]
        img = np.asarray(img, dtype=np.double)
        for i in range(64):
            for j in range(64):
                dist1 = self.cosine(img[j][i], color1)
                dist2 = self.cosine(img[j][i], color2)
                dist3 = self.cosine(img[j][i], color3)
                if dist1 <= min(dist2, dist3):
                    img[j][i] = color1
                elif dist2 <= min(dist1, dist3):
                    img[j][i] = color2
                else:
                    img[j][i] = color3

            # camouflage_img = Image.fromarray(small_img.astype('uint8')).convert('RGB')
        return img

    def get_colorReplaced_imgs(self, imgs, results, batch):
        camouflages = []
        colors = self.get_camouflage_colors(batch, results)
        for i in range(batch):
            img = imgs[i]
            color = colors[i]
            color1, color2, color3 = color[0], color[1], color[2]
            # print(type(color1))
            # print(color1)
            color = np.asarray(color)
            img = np.asarray(img, dtype=np.double)
            for i in range(64):
                for j in range(64):
                    dist1 = self.cosine(img[j][i], color1)
                    dist2 = self.cosine(img[j][i], color2)
                    dist3 = self.cosine(img[j][i], color3)
                    if dist1 <= min(dist2, dist3):
                        img[j][i] = color1
                    elif dist2 <= min(dist1, dist3):
                        img[j][i] = color2
                    else:
                        img[j][i] = color3

            camouflages.append(img)
            # camouflage_img = Image.fromarray(small_img.astype('uint8')).convert('RGB')
        return camouflages

