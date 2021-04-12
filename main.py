from utils import Preprocessing, SummaryGenerator, Metric
from lsa import LSA
from text_rank import TextRank


if __name__ == "__main__":
    with open("data/encelad.txt", mode='r', encoding='utf-8') as f:
        document = f.read()
        f.close()

    preproc = Preprocessing()
    corpus = preproc.preprocess(document)

    # text_rank = TextRank()
    lsa = LSA()
    ranks = lsa.rate_sentences(corpus, vector_model='word2vec')
    # ranks = text_rank.rate_sentences(corpus, vector_model='word2vec')

    summary = SummaryGenerator().summary(preproc.to_corpus(document), ranks)

    reference = '''
    Энцелад - один из спутников Сатурна, в 2010 году удивил ученых, тем 
    что на нем обнаружили гейзеры. Необычная форма спутника и близость к Сатурну 
    способствует циклическому процессу нагреву воды и последующее замерзание до состояния льда. 
    Наличие воды, пусть даже соленой, дает надежду ученым на наличие жизни на Энцеладе.
    '''
    print(summary)

    Metric().rouge(summary, reference, n=2)
