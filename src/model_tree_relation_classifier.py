import collections
import functools

import numpy as np
from nltk.stem import PorterStemmer
from sklearn import tree

from utils.shortest_span import ShortestSpan


class ModelTreeRelationClassifier:
    """
    Relation extractor using Naive bayes and logistics regression
    """

    def __init__(self, marker1, marker2, trigger_words=None, max_tree_depth=5):
        self.marker2 = marker2
        self.marker1 = marker1
        # self._vec = None
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
    def feature_names(self):
        return self._feature_names

    def train(self, x, y):

        tree_features = self.extract_features(x)
        self._model_tree.fit(tree_features, y)

    def extract_features(self, s):
        features = {
            "LSS": self._extract_nearest_distance_trigger,
            "E1C": lambda x: self._extract_marker_occurance(self.marker1, x),
            "E2C": lambda x: self._extract_marker_occurance(self.marker2, x),
            "SPC": lambda x: self._extract_pair_count_per_sentence(self.marker1, self.marker2, x),
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

    def predict(self, x):

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
