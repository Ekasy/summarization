# import nltk
from nltk import pos_tag
from gensim.models import KeyedVectors
import numpy as np
# from func import Preprocessing
#PATH_TO_MODEL = "models/word2vec.model"


class EmbeddingParser(object):
    def __init__(self, path_to_model="models/word2vec.model"):
        self.model = KeyedVectors.load(path_to_model, mmap='r')

    @staticmethod
    def _pos_tag_word(word):
        pt = pos_tag([word], tagset='universal')
        if not pt:
            return word
        return pt[0][0] + '_' + pt[0][1]

    def _pos_tagging(self, corpus):
        for j in range(len(corpus)):
            for i in range(len(corpus[j])):
                corpus[j][i] = self._pos_tag_word(corpus[j][i])
        return corpus

    def _word2vec(self, word):
        # 3 try to define word
        for i in range(3):
            try:
                # if vector is found, return it
                vector = self.model.get_vector(word)
                return vector.tolist()   # dim 300x1
            except KeyError:
                # if lemmatization didn't work well, try to reduce word
                word = word.split('_')[0][:-1] + '_' + word.split('_')[1]
        return np.zeros((300,)).tolist()

    def transform(self, corpus):
        # pos tagging for all corpus
        corpus = self._pos_tagging(corpus)

        # get embedding for each word in corpus
        embeddings = []
        for sentence in corpus:
            s_embeddings = []
            for word in sentence:
                s_embeddings.append(self._word2vec(word))
            embeddings.append(s_embeddings)
        return embeddings


"""
with open("data/encelad.txt", mode='r', encoding='utf-8') as f:
    document = f.read()
    f.close()

preproc = Preprocessing()
corpus = preproc.preprocess(document)

obj = EmbeddingParser(PATH_TO_MODEL)
arr = obj.transform(corpus)

print(arr[:3])"""
