import collections
import functools

import numpy as np
from nltk.stem import PorterStemmer
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from utils.shortest_span import ShortestSpan


class ModelNBTreeRelationClassifier:
    """
    Relation extractor using Naive bayes and logistics regression
    """

    def __init__(self, marker1=None, marker2=None, extract_span=False, max_words_per_class=50, min_df=None,
                 trigger_words=None, max_tree_depth=5):
        self.extract_span = extract_span
        self.min_df = min_df
        self.max_words_per_class = max_words_per_class
        self.marker2 = marker2
        self.marker1 = marker1
        self._vec = None
        self._model_naivebayes = MultinomialNB()
        self._model_tree = tree.DecisionTreeClassifier(max_depth=max_tree_depth)
        self._num_classes = 0
        self._feature_names = []
        self._ngram_range = (1, 3)
        self._analyser = "word"
        self.trigger_words = trigger_words or []
        self._stemmer = PorterStemmer()
        self._shortest_span_calc = ShortestSpan(self._stemmer)

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

    def preprocess(self, x):
        if self.marker1 and self.marker2 and self.extract_span:
            x = [self._extract_span(i, self.marker1, self.marker2) for i in x]
        return x

    def train(self, x, y):
        x = self.preprocess(x)
        self._vec = CountVectorizer(stop_words='english', vocabulary=self._get_vocab(x, y),
                                    ngram_range=self._ngram_range, analyzer=self._analyser)
        unique_labels = np.unique(y)
        assert any([not isinstance(l, int) for l in unique_labels]), "Labels must be numeric"

        self._num_classes = max(unique_labels) + 1
        self._vec.fit(x)

        nb_features = self._get_features_nb(x)
        self._model_naivebayes.fit(nb_features, y)

        tree_features = self.extract_features(x)
        self._model_tree.fit(tree_features, y)

    def _get_features_nb(self, x):
        x_word_vector = self._vec.transform(x)
        extended_features = np.array(x_word_vector.toarray())

        return extended_features

    def _get_vocab(self, x, y):
        unique_labels = np.unique(y)
        result = []
        for l in unique_labels:
            xl_instances = [ix for ix, iy in zip(x, y) if iy == l]
            min_df = self.min_df or max(2, int(len(xl_instances) * .1))
            cv = CountVectorizer(stop_words='english', max_features=self.max_words_per_class, min_df=min_df,
                                 ngram_range=self._ngram_range,
                                 analyzer=self._analyser)
            cv.fit(xl_instances)
            result.extend([w for w in cv.vocabulary_])

        result = list(set(result))

        return result

    def extract_features(self, s):
        features = {
            "LSS": self._extract_nearest_distance_trigger,
            "E1C": lambda x: self._extract_marker_occurance(self.marker1, x),
            "E2C": lambda x: self._extract_marker_occurance(self.marker2, x),
            "SPC": lambda x: self._extract_pair_count_per_sentence(self.marker1, self.marker2, x),
            "NB": lambda x: self._model_naivebayes.predict(self._get_features_nb(x))
        }
        for t in self.trigger_words:
            features[f"LSS_{t}"] = lambda x: self._extract_shortest_distance_trigger(x, t)

        feature_list = []
        feature_names = []
        for n, f in features.items():
            feature_list.append(f(s))
            feature_names.append(n)

        features = np.array(feature_list).T
        self._feature_names = feature_names
        return features

    def _extract_nearest_distance_trigger(self, sentences):
        shortest_distance_feature = map(
            lambda x: self._shortest_distance_triggerwords_markers(x, [self.marker1, self.marker2]), sentences)
        return np.array(list(shortest_distance_feature))

    def _extract_shortest_distance_trigger(self, sentences, trigger):
        shortest_distance_feature = map(
            lambda x: self._shortest_span_calc(x, [self.marker1, self.marker2, self._stemmer.stem(trigger)]), sentences)
        return np.array(list(shortest_distance_feature))

    def _extract_marker_occurance(self, marker, x):
        protein1_occurrance = map(lambda x: self._occurrance_weight(x, marker), x)
        return np.array(list(protein1_occurrance))

    def _extract_pair_count_per_sentence(self, marker1, marker2, x):
        count_pair = map(lambda x: self._pair_per_sentence(x, marker1, marker2, ), x)
        return np.array(list(count_pair))

    def _get_one_hot(self, i):
        result = list(np.zeros(self._num_classes))
        result[i] = 1
        return result

    def predict(self, x):
        x = self.preprocess(x)

        # Use  NB + logistic
        tree_features = self.extract_features(x)
        result = self._model_tree.predict(tree_features)
        result_prob = self._model_tree.predict_proba(tree_features)
        result_prob = np.max(result_prob, axis=1)

        return result, result_prob

    def _shortest_distance_triggerwords_markers(self, sentence, markers):
        if markers is None or len(markers) == 0:
            return self._shortest_span_calc(sentence, markers)

        # If has trigger words, return distance to nearest trigger word
        shortest_distance = None
        stemmed_trigger = [self._stemmer.stem(w) for w in self.trigger_words]
        for p in stemmed_trigger:
            d = self._shortest_span_calc(sentence, markers + [p])
            if shortest_distance is None or d < shortest_distance:
                shortest_distance = d

        if shortest_distance is None:
            shortest_distance = 1000000
        return shortest_distance

    def _occurrance_weight(self, text, word):
        counter = collections.Counter(text.replace(".", " ").split(" "))
        total_wc_count = functools.reduce(lambda a, b: a + b, map(lambda x: x[1], counter.items()))
        return int(counter[word] / total_wc_count * 100)

    def _pair_per_sentence(self, text, w1, w2):
        sentences = text.split(".")
        sentences_with_pairs = list(filter(lambda s: w1 in s and w2 in s, sentences))

        return int(len(sentences_with_pairs) / len(sentences) * 10)
