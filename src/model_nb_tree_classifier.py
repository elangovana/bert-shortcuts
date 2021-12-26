import collections
import functools
import re

import numpy as np
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


class ModelNBTreeClassifier:
    """
    Relation extractor using Naive bayes and logistics regression
    """

    def __init__(self, marker1, marker2, max_words_per_class=500):
        self.max_words_per_class = max_words_per_class
        self.marker2 = marker2
        self.marker1 = marker1
        self._vec = None
        self._model_naivebayes = MultinomialNB()
        self._model_tree = tree.DecisionTreeClassifier(max_depth=10)
        self._num_classes = 0
        self._feature_names = []
        self._ngram_range = (1, 3)
        self._analyser = "word"

    @property
    def tree_model(self):
        return self._model_tree

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

        extended_features = self._get_features_nb(x)
        self._model_naivebayes.fit(extended_features, y)

        tree_features = self._extract_features_tree(x)
        self._model_tree.fit(tree_features, y)

    def _get_features_nb(self, x):
        custom_features = self._extract_custom_features(x)
        x_word_vector = self._vec.transform(x)
        # extended_features = np.hstack((np.array(x_word_vector.toarray()), np.array(custom_features)))
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

    def _extract_features_tree(self, x):
        main_features = self._extract_custom_features(x)

        nb_predictions = self._model_naivebayes.predict(self._get_features_nb(x))

        features = [self._get_one_hot(p) + list(f) for p, f in zip(nb_predictions, main_features)]
        # features = [p + list(f) for p, f in zip(x_vector.tocoo().data, features)]

        self._feature_names = [f"l_{i}" for i in range(self._num_classes)] + ["shortest_dist", "p1_count", "p2_count",
                                                                              "pair_per_sen"]
        return features

    def _extract_custom_features(self, x):
        shortest_distance_feature = map(lambda x: self._shortest_distance(x, self.marker1, self.marker2), x)
        protein1_occurrance = map(lambda x: self._occurrance_weight(x, self.marker1), x)
        protein2_occurrance = map(lambda x: self._occurrance_weight(x, self.marker2), x)
        pair_per_sentence = map(lambda x: self._pair_per_sentence(x, self.marker1, self.marker2), x)
        main_features = list(
            zip(shortest_distance_feature, protein1_occurrance, protein2_occurrance, pair_per_sentence))
        main_features = np.array(main_features)
        return main_features

    def _get_one_hot(self, i):
        result = list(np.zeros(self._num_classes))
        result[i] = 1
        return result

    def predict(self, x):
        nb_extended_features = self._get_features_nb(x)
        # Use just NB
        result = self._model_naivebayes.predict(nb_extended_features)
        result_prob = np.max(np.array(self._model_naivebayes.predict_proba(nb_extended_features)), axis=-1)

        # Use  NB + logistic
        tree_features = self._extract_features_tree(x)
        # result = self._model_tree.predict(tree_features)

        return result, result_prob

    def _shortest_distance(self, text, p1, p2):
        words = re.split('\W+', text)

        start_i = None
        min_distance = None

        for i, w in enumerate(words):
            if w not in [p1, p2]: continue

            # Treat this reset start pointer
            # Case
            #     when new
            #     when w = start_i
            if start_i is None or w == words[start_i]:
                start_i = i

            else:
                end_i = i

                distance = end_i - start_i
                if min_distance is None or min_distance > distance:
                    min_distance = distance

                    min_distance_word = " ".join(words[start_i: end_i + 1])
                # Reset start
                start_i = end_i

        if min_distance is None:
            min_distance = 1000000

        return min_distance

    def _occurrance_weight(self, text, word):
        counter = collections.Counter(text.replace(".", " ").split(" "))
        total_wc_count = functools.reduce(lambda a, b: a + b, map(lambda x: x[1], counter.items()))
        return int(counter[word] / total_wc_count * 100)

    def _pair_per_sentence(self, text, w1, w2):
        sentences = text.split(".")
        sentences_with_pairs = list(filter(lambda s: w1 in s and w2 in s, sentences))

        return int(len(sentences_with_pairs) / len(sentences) * 10)
