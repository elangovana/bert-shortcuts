import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


class ModelNBClassifier:
    """
    Relation extractor using Naive bayes and logistics regression
    """

    def __init__(self, marker1, marker2, max_words_per_class=500):
        self.max_words_per_class = max_words_per_class
        self.marker2 = marker2
        self.marker1 = marker1
        self._vec = None
        self._model_naivebayes = MultinomialNB()
        self._num_classes = 0
        self._feature_names = []
        self._ngram_range = (1, 3)
        self._analyser = "word"

    @property
    def nb_model(self):
        return self._model_naivebayes

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def vocab(self):
        return self._vec.vocabulary_

    def train(self, x, y):
        self._vec = CountVectorizer(stop_words='english', vocabulary=self._get_vocab(x, y),
                                    ngram_range=self._ngram_range, analyzer=self._analyser)
        self._num_classes = len(np.unique(y))
        self._vec.fit(x)

        nb_features = self._get_features_nb(x)
        self._model_naivebayes.fit(nb_features, y)

    def _get_features_nb(self, x):
        x_word_vector = self._vec.transform(x)
        extended_features = np.array(x_word_vector.toarray())

        return extended_features

    def _get_vocab(self, x, y):
        unique_labels = np.unique(y)
        result = []
        for l in unique_labels:
            xl_instances = [ix for ix, iy in zip(x, y) if iy == l]
            min_df = max(2, int(len(xl_instances) * .1))
            cv = CountVectorizer(stop_words='english', max_features=self.max_words_per_class, min_df=min_df,
                                 ngram_range=self._ngram_range,
                                 analyzer=self._analyser)
            cv.fit(xl_instances)
            result.extend([w for w in cv.vocabulary_])

        result = list(set(result))

        return result

    def predict(self, x):
        nb_extended_features = self._get_features_nb(x)
        result = self._model_naivebayes.predict(nb_extended_features)
        result_prob = np.max(np.array(self._model_naivebayes.predict_proba(nb_extended_features)), axis=-1)

        return result, result_prob
