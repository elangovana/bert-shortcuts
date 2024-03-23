import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


class ModelNBNliClassifier:
    """
    NLI using Naive bayes
    """

    def __init__(self, min_df=None, max_words_per_class=10, classwise_vocab=True,
                 stop_words='english', ngram_range=(1, 3), analyzer='word', lowercase=True):
        self._lowercase = lowercase
        self.min_df = min_df
        self.max_words_per_class = max_words_per_class

        self._vec = None
        self._model_naivebayes = MultinomialNB()
        self._num_classes = 0
        self._feature_names = []
        self._analyser = analyzer
        self._stop_words = stop_words
        self._ngram_range = ngram_range
        self._get_vocab = self._get_vocab_common
        if classwise_vocab:
            self._get_vocab = self._get_vocab_classwise

    @property
    def nb_model(self):
        return self._model_naivebayes

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def vocab(self):
        return self._vec.vocabulary_

    def _swap(self, marker1, marker2):
        return marker2, marker1

    def _extract_span(self, item: str, marker1: str, marker2: str):
        m1 = item.find(marker1)
        m2 = item.find(marker2)
        # m1 occurs first
        if m1 > m2:
            marker1, marker2 = self._swap(marker1, marker2)

        m1_start = item.find(marker1)

        m1_end = item.rfind(marker1)
        m2_end = item.rfind(marker2)

        end = m1_end + len(marker1) if m1_end > m2_end else m2_end + len(marker2)

        return item[m1_start: end]

    def train(self, x, y):
        self._vec = CountVectorizer(stop_words=self._stop_words, vocabulary=self._get_vocab(x, y),
                                    ngram_range=self._ngram_range, analyzer=self._analyser, lowercase=self._lowercase)
        self._num_classes = len(np.unique(y))
        self._vec.fit(x)

        nb_features = self.get_features_nb(x)
        self._model_naivebayes.fit(nb_features, y)

    def get_features_nb(self, x):
        x_word_vector = self._vec.transform(x)
        extended_features = np.array(x_word_vector.toarray())

        return extended_features

    def _get_vocab_classwise(self, x, y):
        unique_labels = np.unique(y)
        result = []
        for l in unique_labels:
            xl_instances = [ix for ix, iy in zip(x, y) if iy == l]
            min_df = self.min_df or max(2, int(len(xl_instances) * .1))
            cv = CountVectorizer(stop_words=self._stop_words, max_features=self.max_words_per_class, min_df=min_df,
                                 ngram_range=self._ngram_range, lowercase=self._lowercase,
                                 analyzer=self._analyser)
            cv.fit(xl_instances)
            result.extend([w for w in cv.vocabulary_])

        result = list(set(result))
        print("Max words : ", len(result))
        return result

    def _get_vocab_common(self, x, y):

        min_df = self.min_df or max(2, int(len(x) * .1))
        cv = CountVectorizer(stop_words=self._stop_words, max_features=self.max_words_per_class, min_df=min_df,
                             ngram_range=self._ngram_range, lowercase=self._lowercase,
                             analyzer=self._analyser)
        cv.fit(x)
        result = [w for w in cv.vocabulary_]
        print("Max words : ", len(result))

        return result

    def predict(self, x):
        nb_extended_features = self.get_features_nb(x)
        result = self._model_naivebayes.predict(nb_extended_features)
        result_prob = np.max(np.array(self._model_naivebayes.predict_proba(nb_extended_features)), axis=-1)

        return result, result_prob
