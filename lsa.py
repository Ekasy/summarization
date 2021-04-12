import numpy as np
from numpy.linalg import svd

from tfidf_vectorizer import Vectorizer
from word2vec_prep import EmbeddingParser


class LSA(object):
    def __init__(self, ):
        pass

    def rate_sentences(self, document, vector_model='tf-idf', path_to_model="models/word2vec.model"):
        matrix = self._create_matrix(document, vector_model, path_to_model)
        u, sigma, v = svd(matrix.T, full_matrices=False)
        ranks = self._calc_ranks(sigma, v)
        return ranks

    def _create_matrix(self, document, vector_model, path_to_model):
        if vector_model == 'tf-idf':
            model = Vectorizer()
            vectorized_document = model.fit_transform(_texts=document)
        elif vector_model == 'word2vec':
            model = EmbeddingParser(path_to_model)
            n_vectors = model.transform(document)
            sent_count, dim = len(n_vectors), 300

            vectorized_document = np.zeros((sent_count, dim))
            for i, sent in enumerate(n_vectors):
                vectorized_document[i] = np.sum(sent, axis=0)

        return vectorized_document

    @staticmethod
    def _cosine_similarity(vectorize1, vectorize2):
        num = np.absolute(np.dot(vectorize1, vectorize2))
        denum = np.linalg.norm(vectorize1) * np.linalg.norm(vectorize2)
        return num / denum

    def _calc_ranks(self, sigma, v):
        dim = len(sigma)
        pow_sigma = tuple(s**2 if i < dim else 0.0 for i, s in enumerate(sigma))

        ranks = []
        for column in v.T:
            rank = sum(s*v**2 for s, v in zip(pow_sigma, column))
            ranks.append(np.sqrt(rank))
        return ranks
