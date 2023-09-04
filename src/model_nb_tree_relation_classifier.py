import collections
import functools
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from nltk.stem import PorterStemmer
from sklearn import tree

from model_nb_relation_classifier import ModelNBRelationClassifier
from utils.shortest_span import ShortestSpan


class ModelNBTreeRelationClassifier:
    """
    Relation extractor using Naive bayes and logistics regression
    """

    def __init__(self, marker1=None, marker2=None, min_df=None, max_words_per_class=10, classwise_vocab=True,
                 stop_words='english', ngram_range=(1, 3), extract_span=False,
                 trigger_words=None, max_tree_depth=5):
        self.extract_span = extract_span

        self.marker2 = marker2
        self.marker1 = marker1
        self._model_tree = tree.DecisionTreeClassifier(max_depth=max_tree_depth)
        self._num_classes = 0
        self._feature_names = []
        self.trigger_words = trigger_words or []
        self._stemmer = PorterStemmer()
        self._shortest_span_calc = ShortestSpan(self._stemmer)
        if extract_span:
            self._nb = ModelNBRelationClassifier(marker1, marker2, min_df=min_df,
                                                 max_words_per_class=max_words_per_class,
                                                 classwise_vocab=classwise_vocab,
                                                 stop_words=stop_words, ngram_range=ngram_range)
        else:
            self._nb = ModelNBRelationClassifier(None, None, min_df=min_df,
                                                 max_words_per_class=max_words_per_class,
                                                 classwise_vocab=classwise_vocab,
                                                 stop_words=stop_words, ngram_range=ngram_range)
    @property
    def tree_model(self):
        return self._model_tree

    @property
    def nb_model(self):
        return self._nb.nb_model

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def vocab(self):
        return self._nb.vocab

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

        self._nb.train(x, y)

        print(f"Extracting features..for {len(x)}")
        tree_features = self.extract_features(x)

        print(f"Completed..")
        self._model_tree.fit(tree_features, y)

    def extract_features(self, s):
        features = {
            "LSS": self._extract_nearest_distance_trigger,
            "E1C": lambda x: self._extract_marker_occurance(self.marker1, x),
            "E2C": lambda x: self._extract_marker_occurance(self.marker2, x),
            "SPC": lambda x: self._extract_pair_count_per_sentence(self.marker1, self.marker2, x),
            "NB": lambda x: self._nb.predict(x)[0]
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
        with Pool(os.cpu_count()) as p:
            shortest_distance_feature = p.map(
                self._lambda_shortest_distance_triggerwords_markers, sentences)
        return np.array(list(shortest_distance_feature))

    def _lambda_shortest_distance_triggerwords_markers(self, x):
        return self._shortest_distance_triggerwords_markers(x, [self.marker1, self.marker2])

    def _extract_shortest_distance_trigger(self, sentences, trigger):
        with Pool(os.cpu_count()) as p:
            shortest_distance_feature = p.map(partial(self._lambda_shortest_span_calc, trigger=trigger), sentences)
        return np.array(list(shortest_distance_feature))

    def _lambda_shortest_span_calc(self, x, trigger):
        return self._shortest_span_calc(x, [self.marker1, self.marker2, self._stemmer.stem(trigger)])

    def _extract_marker_occurance(self, marker, x):
        with Pool(os.cpu_count()) as p:
            protein1_occurrance = p.map(partial(self._occurrance_weight, word=marker), x)
        return np.array(list(protein1_occurrance))

    def _extract_pair_count_per_sentence(self, marker1, marker2, x):
        with Pool(os.cpu_count()) as p:
            count_pair = p.map(partial(self._pair_per_sentence, w1=marker1, w2=marker2), x)
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
