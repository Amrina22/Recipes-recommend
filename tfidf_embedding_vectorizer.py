from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np

class TfidfEmbeddingVectorizer(object):
    def __init__(self, model):

        self.model = model
        self.word_idf_weight = None
        self.vector_size = model.wv.vector_size

    def fit(self, docs):
        text_docs = []
        for doc in docs:
            text_docs.append(" ".join(doc))

        tfidf = TfidfVectorizer()
        tfidf.fit(text_docs)
        # if a word was never seen it is given idf of the max of known idf value
        max_idf = max(tfidf.idf_)
        self.word_idf_weight = defaultdict(
            lambda: max_idf,
            [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()],
        )
        return self

    def transform(self, docs):
        doc_word_vector = self.doc_average_list(docs)
        return doc_word_vector

    def doc_average(self, doc):
        mean = []
        for word in doc:
            if word in self.model.wv.index_to_key:
                mean.append(
                    self.model.wv.get_vector(word) * self.word_idf_weight[word]
                )

        if not mean:
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean

    def doc_average_list(self, docs):
        return np.vstack([self.doc_average(doc) for doc in docs])