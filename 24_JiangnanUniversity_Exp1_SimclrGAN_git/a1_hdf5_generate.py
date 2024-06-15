import os
import h5py
from PIL import Image
from a3_text2vec import txt2vec

class Generate_h5py():
    def __init__(self, img_path='./data_snowed/', text_path = './text_snowed.txt'):
        self.img_path = img_path
        self.text_path = text_path
        self.sentences = []
        f = open('./text_snowed.txt', "r", encoding='utf-8')
        line = f.readline()
        while line:
            self.sentences.append(line.split())
            line = f.readline()

    def generate(self):
        f = h5py.File("train_snowed.hdf5", 'w')
        train = f.create_group('train')
        files = os.listdir(self.img_path)

        for file in files:
            file_path = os.path.join(self.img_path + file)
            ###将img提取出来###
            img = Image.open(file_path)
            ###将文本提取出来###
            idx = file[6:8]
            for sentence in self.sentences:
                if sentence[0] == idx:
                    txt = sentence[1:]
            ###将编码提取出来###
            txt_to_vec = txt2vec()
            embedding = txt_to_vec.build_sentence_vector(sentence = txt)
            # 形状为[64,64,3]

            example = f['train'].create_group(file.split('.')[0])
            example.create_dataset('name', data=file)
            example.create_dataset('img', data=img)
            example.create_dataset('txt', data=txt)
            example.create_dataset('embeddings', data=embedding)

        f.close()

generate = Generate_h5py()
generate.generate()