from gensim.models import Word2Vec
import numpy as np
from collections import defaultdict
from gensim.models import KeyedVectors

class txt2vec():
    def __init__(self,path = './text_snowed.txt',save_name = './word2vec_snowed.txt'):
        self.path = path
        self.word_counts = defaultdict(int)
        self.save_name = save_name

    def train(self):
        f = open(self.path,"r",encoding='utf-8')
        line = f.readline()
        sentences = []
        while line:
            sentences.append(line.split()[1:])
            line = f.readline()

        for row in sentences:
            for word in row:
                self.word_counts[word] += 1

        model = Word2Vec(vector_size=1024, window=5, min_count=1, workers=0, sg=1)
        model.build_vocab(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=1000)
        model.wv.save_word2vec_format(self.save_name, binary = False)
        #model.save__forest(self.save_name)

    def load(self):
        model = KeyedVectors.load_word2vec_format(self.save_name, binary=False,unicode_errors='ignore')
        #model = Word2Vec.load(self.save_name)
        print(model['snowed'])

    # 对每个句子的所有词向量取均值，来生成一个句子的vector,SENTENCE应该是字符串列表
    def build_sentence_vector(self, sentence, size=1024):
        w2v_model = KeyedVectors.load_word2vec_format(self.save_name, binary=False,unicode_errors='ignore')
        sen_vec = np.zeros(size).reshape((1, size))
        count = 0
        for word in sentence:
            try:
                sen_vec += w2v_model[word].reshape((1, size))
                count += 1
            except KeyError:
                continue
        if count != 0:
            sen_vec /= count
        return sen_vec

# 生成词向量表
txt_to_vec = txt2vec()
txt_to_vec.train()