import collections

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import re
import functools


class ModelNBLogisticClassifier:
    """
    Relation extractor using Naive bayes and logistics regression
    """

    def __init__(self, marker1, marker2):
        self.marker2 = marker2
        self.marker1 = marker1
        self._vec = CountVectorizer(stop_words='english')
        self._model_naivebayes = MultinomialNB()
        self._model_logistic = LogisticRegression()

    def train(self, x, y):
        x_vector = self._vec.fit_transform(x)
        self._model_naivebayes.fit(x_vector, y)

        features = self._extract_features(x, x_vector)
        self._model_logistic.fit(features, y)

    def _extract_features(self, x, x_vector):
        shortest_distance_feature = map(lambda x: self._shortest_distance(x, self.marker2, self.marker1), x)
        protein1_occurrance = map(lambda x: self._occurrance_weight(x, self.marker1), x)
        protein2_occurrance = map(lambda x: self._occurrance_weight(x, self.marker2), x)
        pair_per_sentence = map(lambda x: self._pair_per_sentence(x, self.marker1, self.marker2), x)
        nb_probs = self._model_naivebayes.predict(x_vector)
        features = list(zip(shortest_distance_feature, protein1_occurrance, protein2_occurrance, pair_per_sentence, nb_probs))
        print(features)
        return features

    def predict(self, x):
        x_vector = self._vec.fit(x)
        features = self._extract_features(x, x_vector)
        return self._model_logistic.predict(features)

    def _shortest_distance(self, text, p1, p2):
        words = re.split('\W+', text)

        start_i = None
        min_distance = 0

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
        return min_distance

    def _occurrance_weight(self, text, word):
        counter = collections.Counter(text.replace(".", " ").split(" "))
        total_wc_count = functools.reduce(lambda a, b: a + b, map(lambda x: x[1], counter.items()))
        return counter[word] / total_wc_count

    def _pair_per_sentence(self, text, w1, w2):
        sentences = text.split(".")
        sentences_with_pairs = list(filter(lambda s: w1 in s and w2 in s, sentences))

        return len(sentences_with_pairs) / len(sentences)
