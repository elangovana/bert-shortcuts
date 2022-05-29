import collections
import functools
import re

import numpy as np
from nltk.stem import PorterStemmer
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


class ModelNBTreeRelationClassifier:
    """
    Relation extractor using Naive bayes and logistics regression
    """

    def __init__(self, marker1, marker2, max_words_per_class=500, min_df=None, trigger_words=None):
        self.min_df = min_df
        self.max_words_per_class = max_words_per_class
        self.marker2 = marker2
        self.marker1 = marker1
        self._vec = None
        self._model_naivebayes = MultinomialNB()
        self._model_tree = tree.DecisionTreeClassifier(max_depth=5)
        self._num_classes = 0
        self._feature_names = []
        self._ngram_range = (1, 3)
        self._analyser = "word"
        self.trigger_words = trigger_words or []

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
        unique_labels = np.unique(y)
        assert any([not isinstance(l, int) for l in unique_labels]), "Labels must be numeric"

        self._num_classes = max(unique_labels) + 1
        self._vec.fit(x)

        nb_features = self._get_features_nb(x)
        self._model_naivebayes.fit(nb_features, y)

        tree_features = self._extract_features_tree(x)
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

    def _extract_features_tree(self, s):
        features = {
            "nearest_distance_trigger": self._extract_nearest_distance_trigger,
            f"{self.marker1}_count": lambda x: self._extract_marker_occurance(self.marker1, x),
            f"{self.marker2}_count": lambda x: self._extract_marker_occurance(self.marker2, x),
            "pair_count": lambda x: self._extract_pair_count_per_sentence(self.marker1, self.marker2, x),
            "naive_bayes": lambda x: self._model_naivebayes.predict(self._get_features_nb(x))
        }
        for t in self.trigger_words:
            features[f"shortest_distance_{t}"] = lambda x: self._extract_shortest_distance_trigger(x, t)

        feature_list = []
        feature_names = []
        for n, f in features.items():
            feature_list.append(f(s))
            feature_names.append(n)

        features = np.array(feature_list).T
        print(features.shape)
        self._feature_names = feature_names
        return features

    # def _extract_custom_features(self, x):
    #     shortest_distance_feature = map(
    #         lambda x: self._shortest_distance_triggerwords_markers(x, [self.marker1, self.marker2]), x)
    #     protein2_occurrance = map(lambda x: self._occurrance_weight(x, self.marker2), x)
    #     pair_per_sentence = map(lambda x: self._pair_per_sentence(x, self.marker1, self.marker2), x)
    #     main_features = list(
    #         zip(shortest_distance_feature, protein1_occurrance, protein2_occurrance, pair_per_sentence))
    #     main_features = np.array(main_features)
    #     return main_features

    def _extract_nearest_distance_trigger(self, sentences):
        shortest_distance_feature = map(
            lambda x: self._shortest_distance_triggerwords_markers(x, [self.marker1, self.marker2]), sentences)
        return np.array(list(shortest_distance_feature))

    def _extract_shortest_distance_trigger(self, sentences, trigger):
        shortest_distance_feature = map(
            lambda x: self._shortest_distance(x, [self.marker1, self.marker2, trigger]), sentences)
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

        # Use  NB + logistic
        tree_features = self._extract_features_tree(x)
        result = self._model_tree.predict(tree_features)
        result_prob = self._model_tree.predict_proba(tree_features)
        result_prob = np.max(result_prob, axis=1)

        return result, result_prob

    def _shortest_distance_triggerwords_markers(self, sentence, markers):
        if markers is None or len(markers) == 0:
            return self._shortest_distance(sentence, markers)

        # If has trigger words, return distance to nearest trigger word
        shortest_distance = None
        porter = PorterStemmer()
        stemmed_ptm = [porter.stem(w) for w in self.trigger_words]
        for p in stemmed_ptm:
            d = self._shortest_distance(sentence, markers + [p])
            if shortest_distance is None or d < shortest_distance:
                shortest_distance = d
        return shortest_distance

    def _shortest_distance(self, sentence, words_to_match):
        words = re.split('\W+', sentence)
        porter = PorterStemmer()

        stemmed_words = [porter.stem(w) for w in words]
        subwords = [porter.stem(w) for w in words_to_match]
        min_distance = None
        subwords_matched = []
        subwords_matched_indices = []
        min_distance_word = ""
        for i, w in enumerate(stemmed_words):
            if w not in subwords: continue

            # Treat this reset start pointer
            # Case
            #     when new
            #     when w = start_i
            if w in subwords_matched:
                # print("Matched", w, subwords_matched)
                for s_i, s in enumerate(subwords_matched):
                    if s == w:
                        del subwords_matched[s_i]
                        del subwords_matched_indices[s_i]

            subwords_matched.append(w)
            subwords_matched_indices.append(i)

            if all([s in subwords_matched for s in subwords]):
                start_i = subwords_matched_indices[0]
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
