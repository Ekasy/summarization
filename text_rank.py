import numpy as np
from tfidf_vectorizer import Vectorizer
from word2vec_prep import EmbeddingParser


class TextRank(object):
    def __init__(self, _epsilon=1e-10, _d=0.85, _delta=1e-7):
        self.epsilon = _epsilon
        self.d = _d
        self.delta = _delta

    def rate_sentences(self, document, vector_model='tf-idf', path_to_model="models/word2vec.model"):
        matrix = self._create_matrix(document, vector_model, path_to_model)
        ranks = self._calc_ranks(matrix, self.epsilon)
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

        weights = np.zeros((len(vectorized_document), len(vectorized_document)))

        for i, sentence_i in enumerate(vectorized_document):
            for j, sentence_j in enumerate(vectorized_document):
                weights[i][j] = self._cosine_similarity(sentence_i, sentence_j)

        weights = np.divide(weights, weights.sum(axis=1).reshape(-1, 1))
        return weights

    def _calc_ranks(self, matrix, epsilon):
        p_prev = np.ones((matrix.shape[0],)) / matrix.shape[0]
        loss = 1.0

        while loss > epsilon:
            p = (1 - self.delta) * np.matmul(matrix.transpose(), p_prev) + self.delta * np.ones((matrix.shape[0],)) / \
                matrix.shape[0]
            loss = np.linalg.norm(np.subtract(p, p_prev))
            p_prev = p

        return p_prev

    @staticmethod
    def _cosine_similarity(vectorize1, vectorize2):
        num = np.absolute(np.dot(vectorize1, vectorize2))
        denum = np.linalg.norm(vectorize1) * np.linalg.norm(vectorize2)
        return num / denum
