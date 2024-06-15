from PIL import Image
import numpy as np

def generate_recMask():
    img_lt,img_rt,img_lb,img_rb = Image.new("L",(256,256),(255)),Image.new("L",(256,256),(255)),Image.new("L",(256,256),(255)),Image.new("L",(256,256),(255))
    img_lt_np = np.array(img_lt)
    img_lb_np = np.array(img_lb)
    img_rt_np = np.array(img_rt)
    img_rb_np = np.array(img_rb)
    for i in range(128):
        for j in range(128):
            img_lt_np[2 * i, 2 * j] = 0
            img_lb_np[2 * i + 1, 2 * j] = 0
            img_rt_np[2 * i, 2 * j + 1] = 0
            img_rb_np[2 * i + 1, 2 * j + 1] = 0

    img_lt = Image.fromarray(img_lt_np, mode='L')
    img_rt = Image.fromarray(img_rt_np, mode='L')
    img_lb = Image.fromarray(img_lb_np, mode='L')
    img_rb = Image.fromarray(img_rb_np, mode='L')

    img_lt.save('./a1_Full_Image_Mask/mask_lt.png')
    img_rt.save('./a1_Full_Image_Mask/mask_rt.png')
    img_lb.save('./a1_Full_Image_Mask/mask_lb.png')
    img_rb.save('./a1_Full_Image_Mask/mask_rb.png')

def generate_lapMask(gt_mask_path):
    img = Image.open(gt_mask_path)
    img = img.convert('L')
    img_np = np.array(img)

    img_lt, img_rt, img_lb, img_rb = Image.open('./a1_Full_Image_Mask/mask_lt.png'), Image.open('./a1_Full_Image_Mask/mask_rt.png'), Image.open('./a1_Full_Image_Mask/mask_lb.png'), Image.open('./a1_Full_Image_Mask/mask_rb.png')
    img_lt, img_rt, img_lb, img_rb = img_lt.convert('L'), img_rt.convert('L'), img_lb.convert('L'), img_rb.convert('L')
    img_lt_np = np.array(img_lt)
    img_lb_np = np.array(img_lb)
    img_rt_np = np.array(img_rt)
    img_rb_np = np.array(img_rb)

    gt_mask_lt, gt_mask_lb,gt_mask_rt,gt_mask_rb = Image.new("L",(256,256),(255)),Image.new("L",(256,256),(255)),Image.new("L",(256,256),(255)),Image.new("L",(256,256),(255))
    gt_mask_lt, gt_mask_lb, gt_mask_rt, gt_mask_rb = np.array(gt_mask_lt), np.array(gt_mask_lb), np.array(gt_mask_rt), np.array(gt_mask_rb)
    for i in range(256):
        for j in range(256):
            gt_mask_lt[i][j] = 0 if ((img_np[i][j] == 0) and (img_lt_np[i][j] == 0)) else 255
            gt_mask_lb[i][j] = 0 if ((img_np[i][j] == 0) and (img_lb_np[i][j] == 0)) else 255
            gt_mask_rt[i][j] = 0 if ((img_np[i][j] == 0) and (img_rt_np[i][j] == 0)) else 255
            gt_mask_rb[i][j] = 0 if ((img_np[i][j] == 0) and (img_rb_np[i][j] == 0)) else 255

    gt_mask_lt = Image.fromarray(gt_mask_lt, mode='L')
    gt_mask_lb = Image.fromarray(gt_mask_lb, mode='L')
    gt_mask_rt = Image.fromarray(gt_mask_rt, mode='L')
    gt_mask_rb = Image.fromarray(gt_mask_rb, mode='L')

    gt_mask_lt.save('./a2_lt_mask/000000.png')
    gt_mask_lb.save('./a2_rt_mask/000000.png')
    gt_mask_rt.save('./a2_lb_mask/000000.png')
    gt_mask_rb.save('./a2_rb_mask/000000.png')

def generate_Mask4Standerize():
    img_lt,img_rt,img_lb,img_rb = Image.new("L",(256,256),(255)),Image.new("L",(256,256),(255)),Image.new("L",(256,256),(255)),Image.new("L",(256,256),(255))
    img_lt_np = np.array(img_lt)
    img_lb_np = np.array(img_lb)
    img_rt_np = np.array(img_rt)
    img_rb_np = np.array(img_rb)
    for i in range(128):
        for j in range(128):
            img_lt_np[2 * i, 2 * j] = 0

            img_rt_np[2 * i, 2 * j] = 0
            img_rt_np[2 * i, 2 * j + 1] = 0

            img_lb_np[2 * i, 2 * j] = 0
            img_lb_np[2 * i, 2 * j + 1] = 0
            img_lb_np[2 * i + 1, 2 * j] = 0

            img_rb_np[2 * i, 2 * j] = 0
            img_rb_np[2 * i, 2 * j + 1] = 0
            img_rb_np[2 * i + 1, 2 * j] = 0
            img_rb_np[2 * i + 1, 2 * j + 1] = 0

    img_lt = Image.fromarray(img_lt_np, mode='L')
    img_rt = Image.fromarray(img_rt_np, mode='L')
    img_lb = Image.fromarray(img_lb_np, mode='L')
    img_rb = Image.fromarray(img_rb_np, mode='L')

    img_lt.save('./a1_Full_Standerize_Mask/mask_lt.png')
    img_rt.save('./a1_Full_Standerize_Mask/mask_rt.png')
    img_lb.save('./a1_Full_Standerize_Mask/mask_lb.png')
    img_rb.save('./a1_Full_Standerize_Mask/mask_rb.png')

def generate_lapMask_forStanderize(gt_mask_path):
    img = Image.open(gt_mask_path)
    img = img.convert('L')
    img_np = np.array(img)

    img_lt, img_rt, img_lb, img_rb = Image.open('./a1_Full_Standerize_Mask/mask_lt.png'), Image.open('./a1_Full_Standerize_Mask/mask_rt.png'), Image.open('./a1_Full_Standerize_Mask/mask_lb.png'), Image.open('./a1_Full_Standerize_Mask/mask_rb.png')
    img_lt, img_rt, img_lb, img_rb = img_lt.convert('L'), img_rt.convert('L'), img_lb.convert('L'), img_rb.convert('L')
    img_lt_np = np.array(img_lt)
    img_lb_np = np.array(img_lb)
    img_rt_np = np.array(img_rt)
    img_rb_np = np.array(img_rb)

    gt_mask_lt, gt_mask_lb,gt_mask_rt,gt_mask_rb = Image.new("L",(256,256),(255)),Image.new("L",(256,256),(255)),Image.new("L",(256,256),(255)),Image.new("L",(256,256),(255))
    gt_mask_lt, gt_mask_lb, gt_mask_rt, gt_mask_rb = np.array(gt_mask_lt), np.array(gt_mask_lb), np.array(gt_mask_rt), np.array(gt_mask_rb)
    for i in range(256):
        for j in range(256):
            gt_mask_lt[i][j] = 0 if ((img_np[i][j] == 0) and (img_lt_np[i][j] == 0)) else 255
            gt_mask_lb[i][j] = 0 if ((img_np[i][j] == 0) and (img_lb_np[i][j] == 0)) else 255
            gt_mask_rt[i][j] = 0 if ((img_np[i][j] == 0) and (img_rt_np[i][j] == 0)) else 255
            gt_mask_rb[i][j] = 0 if ((img_np[i][j] == 0) and (img_rb_np[i][j] == 0)) else 255

    gt_mask_lt = Image.fromarray(gt_mask_lt, mode='L')
    gt_mask_lb = Image.fromarray(gt_mask_lb, mode='L')
    gt_mask_rt = Image.fromarray(gt_mask_rt, mode='L')
    gt_mask_rb = Image.fromarray(gt_mask_rb, mode='L')

    gt_mask_lt.save('./a2_Target_Standerize_Mask/lt.png')
    gt_mask_lb.save('./a2_Target_Standerize_Mask/lb.png')
    gt_mask_rt.save('./a2_Target_Standerize_Mask/rt.png')
    gt_mask_rb.save('./a2_Target_Standerize_Mask/rb.png')
# generate_recMask()

# generate_lapMask('./.mask1/gt_mask.png')
# generate_Mask4Standerize()
class genertor():
    def __init__(self, gt_mask_path):
        self.mask_path = gt_mask_path
    def generate(self):
        generate_recMask()
        generate_Mask4Standerize()
        generate_lapMask(self.mask_path)
        generate_lapMask_forStanderize(self.mask_path)



gt_mask_path = './a0_originImg/gt_mask.png'
genertor = genertor(gt_mask_path)
genertor.generate()
# generate_lapMask_forStanderize(gt_mask_path)