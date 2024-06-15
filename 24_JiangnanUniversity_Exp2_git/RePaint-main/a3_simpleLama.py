import numpy as np
from simple_lama_inpainting import SimpleLama
from PIL import Image

def generate_referenceImg(targetPath,maskPath):
    img_gt = Image.open(targetPath)
    img_gt = img_gt.convert("RGB")
    img_gt.save("./a3_Lama_process/target.png")

    img_mask = Image.open(maskPath)
    img_mask = img_mask.convert('L')

    #img_mask二值化
    threshold = 20
    table = []
    for i in range(256):
        if i < threshold:
            table.append(1)
        else:
            table.append(0)
    # 图片二值化
    photo = img_mask.point(table, '1')
    photo.save("./a3_Lama_process/mask.png")

    img_reshape = Image.open('./a3_Lama_process/target.png')
    mask_reshape = Image.open('./a3_Lama_process/mask.png')

    simple_lama = SimpleLama()
    result = simple_lama(img_reshape, mask_reshape)
    result.save("./a3_reference_Img/000000.png")

# generate_referenceImg('./target.png','./.maskForStanderize/rb.png')

class lama():
    # def __init__(self,targetPath = './target.png',maskPath = './gt_mask.png'):
    def __init__(self, targetPath='./a0_originImg/target.png', maskPath='./a0_originImg/gt_mask.png'):
        self.targetPath = targetPath
        self.maskPath = maskPath

    def createReferenceImg(self):
        generate_referenceImg(self.targetPath,self.maskPath)
lama1 = lama()
lama1.createReferenceImg()
'''
def reshape_gt(gt_path):
    img = Image.open(gt_path)
    img = img.convert("RGB")
    img.save("./target_reshape.png")

def reshape_mask(mask_path):
    img = Image.open(mask_path)
    img = img.convert('L')
    img.save("./mask_reshape.jpg")

    threshold = 20

    table = []
    for i in range(256):
        if i < threshold:
            table.append(1)
        else:
            table.append(0)

    # 图片二值化
    photo = img.point(table, '1')



# simple_lama = SimpleLama()
:

reshape_gt(img_path)

reshape_mask(mask_path)

img_reshape_path = "./gt/gt_reshape.png"
mask_reshape_path = "./mask/mask_reshape2.jpg"

img_reshape = Image.open(img_reshape_path)
mask_reshape = Image.open(mask_reshape_path)

simple_lama = SimpleLama()
result = simple_lama(img_reshape, mask_reshape)
result.save("./output/inpainted.png")
'''