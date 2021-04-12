import numpy as np
import pymorphy2
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')


class SummaryGenerator(object):
    def __init__(self, order=True):
        self.order = order

    def summary(self, corpus, ranks, top_n=3):
        ranks = np.array(ranks)
        maxinds = (-ranks).argsort()[:top_n]
        if self.order:
            maxinds = np.sort(maxinds)

        summary = ''
        for ind in maxinds:
            summary += corpus[ind] + '. '

        return summary[:-1]


class Metric(object):
    def rouge(self, predict, y_true, n=1):
        predict_tokens = self._get_tokens(predict, n)
        y_true_tokens = self._get_tokens(y_true, n)
        count = 0

        for pred in predict_tokens:
            if pred in y_true_tokens:
                count += 1
                continue

        precision = count / len(predict_tokens)
        recall = count / len(y_true_tokens)
        f1 = 2 * precision * recall / (precision + recall)

        print(f'Precision: {round(precision, 4)}')
        print(f'Recall:    {round(recall, 4)}')
        print(f'F1-score:  {round(f1, 4)}')

        return precision, recall, f1

    @staticmethod
    def _get_tokens(text, n=2):
        corpus = Preprocessing(_cleaning=False).preprocess(text)
        corpus_list = []
        for sent in corpus:
            for word in sent:
                corpus_list.append(word)

        corpus_n_gram = []
        for i in range(len(corpus_list) - (n - 1)):
            buf = []
            for j in range(n):
                buf.append(corpus_list[i + j])
            corpus_n_gram.append(tuple(buf))
        return list(set(corpus_n_gram))


class Preprocessing(object):
    def __init__(self, _lemmatize=True, _cleaning=True, _tokenize=True):
        """
        set settings for next preprocess
        """
        self.lemmatize = _lemmatize
        self.cleaning = _cleaning
        self.tokenize = _tokenize

    def preprocess(self, text: str):
        # text to list of sentences
        corpus = self._text2corpus(text)

        # tokenize
        if self.tokenize:
            corpus = self._tokenize_corpus(corpus)

        # lemmatize
        if self.lemmatize:
            corpus = self._lemmatize_corpus(corpus)

        # cleaning
        if self.cleaning:
            corpus = self._clean_corpus(corpus)

        return corpus

    @staticmethod
    def _text2corpus(text: str, lower=True):
        t = text
        if lower:
            t = t.lower()
        t = t.replace('\n\n', '. ')
        t = t.replace('\n', '. ')
        t = t.replace('—', '-')
        t = t.replace('»', '"')
        t = t.replace('«', '"')
        t = t.replace('...', '.')
        t = t.replace('..', '.')
        t = t.replace('   ', ' ')
        t = t.replace('  ', ' ')
        corpus = []
        for sentence in t.split('. ')[:-1]:
            if sentence == '':
                continue
            corpus.append(sentence)
        return corpus

    @staticmethod
    def _tokenize_corpus(corpus):
        new_corpus = []
        TOKENIZE_RE = re.compile(r'[A-ЯЁа-яё]+-[A-ЯЁа-яё]+|[A-ЯЁа-яё]+', re.I)
        for sentence in corpus:
            new_corpus.append(TOKENIZE_RE.findall(sentence))
        return new_corpus

    @staticmethod
    def _lemmatize_corpus(corpus):
        morph = pymorphy2.MorphAnalyzer()
        for sentence in range(len(corpus)):
            for word in range(len(corpus[sentence])):
                corpus[sentence][word] = morph.parse(corpus[sentence][word])[0].normal_form
        return corpus

    @staticmethod
    def _clean_corpus(corpus):
        stpwrd = stopwords.words('russian') + stopwords.words('english')
        for sentence in range(len(corpus)):
            word = 0
            while word < len(corpus[sentence]):
                if corpus[sentence][word] in stpwrd:
                    # corpus[sentence].remove(corpus[sentence][word])
                    corpus[sentence] = corpus[sentence][:word] + corpus[sentence][word + 1:]
                    word -= 1
                word += 1
        return corpus

    def to_corpus(self, text):
        return self._text2corpus(text, False)


def text2corpus(text: str) -> list:
    t = text.lower()
    t = t.replace('\n\n', '.')
    t = t.replace('\n', '.')
    t = t.replace('—', '-')
    t = t.replace('...', '.')
    t = t.replace('..', '.')
    t = t.replace('  ', ' ')
    t += ' '
    return t.split('. ')[:-1]


def tokenize(txt: str) -> list:
    TOKENIZE_RE = re.compile(r'\w+', re.I)
    return TOKENIZE_RE.findall(txt.lower())


def to_sentences(text: str) -> list:
    text = ' '.join(text.split('\n'))
    text = text.replace('  ', ' ')
    return text.split('. ')[:-1]


def tokenize_texts(texts: list) -> list:
    tokens = []
    for sentence in texts:
        tokens.append(tokenize(sentence))
    return tokens


def lemmatize(tokens: list) -> list:
    morph = pymorphy2.MorphAnalyzer()

    for sentence in range(len(tokens)):
        for word in range(len(tokens[sentence])):
            tokens[sentence][word] = morph.parse(tokens[sentence][word])[0].normal_form
    return tokens


def simple_similarity(tokens: list, texts: list) -> dict:
    similarity = {}
    for i in range(len(tokens)):
        for j in range(i, len(tokens)):
            count = 0
            for word in tokens[i]:
                if word in tokens[j]:
                    count += 1
            
            if not similarity.get(i):
                similarity[i] = {}
            similarity[i][j] = count / (len(tokens[i]) * len(tokens[j]))
        similarity[i] = (sum(similarity[i].values()), texts[i])
    return similarity


def ranging(similarity: dict) -> list:
    list_similarity = list(similarity.items())
    list_similarity.sort(key=lambda i: i[1][0], reverse=True)
    return list_similarity


def generate_title(list_similarity: list, top_n: int, save_order=True) -> str:
    title = ''
    
    top_sentences = list_similarity[:top_n]
    if save_order:
        top_sentences.sort()
    
    for i in range(top_n):
        txt = top_sentences[i][1][1]
        txt = txt.replace('\n', '')
        txt = txt[1:] if txt[0] == ' ' else txt
        txt = txt[:-1] if txt[-1] == ' ' else txt
        title += txt + '. '
        
    return title
