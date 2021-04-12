from utils import to_sentences
import numpy as np


class Vectorizer:
    def __init__(self, _mode='tf-idf', _lemmatize=True, _scale=True):
        self.mode = _mode
        self.lemmatize = _lemmatize
        self.scale = _scale

    def _get_unique(self) -> set:
        unique_words = set()
        for tokens in self.collection_tokenize:
            for token in tokens:
                unique_words.add(token)
        return unique_words

    def fit(self, _texts=None, _text=None) -> None:
        if _text is not None and _texts is None:
            self.texts = to_sentences(_text)
        else:
            self.texts = _texts

        self.collection_tokenize = _texts  # lemmatize(tokenize_texts(self.texts)) if lemmatize else tokenize_texts(self.texts)
        self.unique_words = self._get_unique()
        self.dictionary = {}
        for word in self.unique_words:
            self.dictionary[word] = 0
            for sentence in self.collection_tokenize:
                if word in sentence:
                    self.dictionary[word] += 1
            self.dictionary[word] /= len(self.collection_tokenize)

    def _vectorize_text(self, word2id, word2freq):
        result = np.zeros((len(self.collection_tokenize), len(word2id)))
        for text_i, text in enumerate(self.collection_tokenize):
            for token in text:
                if token in word2id:
                    result[text_i, word2id[token]] += 1

        if self.mode == 'tf-idf':
            tf = result / result.sum(axis=1).reshape((-1, 1))  # делим каждую строку на ее длину
            idf = (result > 0).astype('float32') * word2freq  # делим каждый столбец на вес слова
            tfidf = np.log(1 + tf) * idf

            if self.scale:
                result = (tfidf - tfidf.mean(axis=0)) / tfidf.std(axis=0, ddof=1)

        return result

    def transform(self):
        word_df = list(self.dictionary.items())
        word_df.sort(key=lambda t: (t[1], t[0]))
        word_doc_freq = np.array([cnt for _, cnt in word_df], dtype='float64')
        vocabulary = {word: i for i, (word, _) in enumerate(word_df)}
        self.vectorized = self._vectorize_text(vocabulary, word_doc_freq)
        return self.vectorized

    def fit_transform(self, _texts=None, _text=None):
        self.fit(_texts=_texts, _text=_text)
        return self.transform()


def cosine_similarity(vectorize1: list, vectorize2: list):
    num = np.dot(vectorize1, vectorize2)
    denum = np.linalg.norm(vectorize1) * np.linalg.norm(vectorize2)
    # ch = np.absolute((vectorize1 * vectorize2).sum())
    # zn = np.sqrt((vectorize1**2).sum()) * np.sqrt((vectorize2**2).sum())
    return num / denum


def generate_similarity(vectorize: list, texts: list):
    similarity = {}
    for row in range(len(vectorize)):
        for row2 in range(row + 1, len(vectorize)):
            if not similarity.get(row):
                similarity[row] = {}
            if not similarity.get(row2):
                similarity[row2] = {}

            similarity[row][row2] = cosine_similarity(vectorize[row], vectorize[row2])
            similarity[row2][row] = cosine_similarity(vectorize[row], vectorize[row2])

    list_similarity = []
    for key in similarity:
        summ = 0.0
        size = len(list(similarity[key]))
        for key2 in similarity[key]:
            summ += similarity[key][key2]

        list_similarity.append(
            (key, summ / size, texts[key])
        )

    return list_similarity


def ranging(list_similarity: list):
    ranging_list = list_similarity.copy()
    ranging_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return ranging_list


def generate_title(titles, top_n):
    title = ''

    top_sentences = titles[:top_n]

    for i in range(top_n):
        txt = top_sentences[i][2]
        txt = txt.replace('\n', '')
        txt = txt[1:] if txt[0] == ' ' else txt
        txt = txt[:-1] if txt[-1] == ' ' else txt
        title += txt + '. '

    return title