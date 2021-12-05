import argparse
import collections
import logging
import sys

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import re
import functools
import pandas as pd
import numpy as np


class ModelNBLogisticClassifier:
    """
    Relation extractor using Naive bayes and logistics regression
    """

    def __init__(self, marker1, marker2):
        self.marker2 = marker2
        self.marker1 = marker1
        self._vec = None
        self._model_naivebayes = MultinomialNB()
        self._model_logistic = LogisticRegression()
        self._num_classes = 0

    def train(self, x, y):
        self._vec = CountVectorizer(stop_words='english', vocabulary=self._get_vocab(x, y))
        self._num_classes = len(np.unique(y))
        x_vector = self._vec.fit_transform(x)
        self._model_naivebayes.fit(x_vector, y)

        features = self._extract_features(x, x_vector)
        self._model_logistic.fit(features, y)

    def _get_vocab(self, x, y):
        unique_labels = np.unique(y)
        result = []
        for l in unique_labels:
            xl_instances = [ix for ix, iy in zip(x, y) if iy == l]
            cv = CountVectorizer(stop_words='english', max_features=100)
            cv.fit(xl_instances)
            result.extend([w for w in cv.vocabulary_])

        result = list(set(result))

        return result

    def _extract_features(self, x, x_vector):
        shortest_distance_feature = map(lambda x: self._shortest_distance(x, self.marker1, self.marker2), x)
        protein1_occurrance = map(lambda x: self._occurrance_weight(x, self.marker1), x)
        protein2_occurrance = map(lambda x: self._occurrance_weight(x, self.marker2), x)
        pair_per_sentence = map(lambda x: self._pair_per_sentence(x, self.marker1, self.marker2), x)
        nb_predictions = self._model_naivebayes.predict(x_vector)
        features = list(
            zip(shortest_distance_feature, protein1_occurrance, protein2_occurrance, pair_per_sentence))
        features = [self._get_one_hot(p) + list(f) for p, f in zip(nb_predictions, features)]
        return features

    def _get_one_hot(self, i):
        result = list(np.zeros(self._num_classes))
        result[i] = 1
        return result

    def predict(self, x):
        x_vector = self._vec.transform(x)
        features = self._extract_features(x, x_vector)

        # Use just NB
        result = self._model_naivebayes.predict(x_vector)

        # Use  NB + logistic
        # result = self._model_logistic.predict(features)

        return result

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
        return counter[word] / total_wc_count

    def _pair_per_sentence(self, text, w1, w2):
        sentences = text.split(".")
        sentences_with_pairs = list(filter(lambda s: w1 in s and w2 in s, sentences))

        return len(sentences_with_pairs) / len(sentences)


def run_main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--trainfile",
                        help="The input ppi multiclass train file", required=True)

    parser.add_argument("--testfile",
                        help="The input ppi multiclass test file", required=True)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print(args.__dict__)

    compute(args.trainfile, args.testfile)


def compute(trainfile, testfile):
    df_train = pd.read_json(trainfile, orient="records")
    df_test = pd.read_json(testfile, orient="records")
    m = ModelNBLogisticClassifier("PROTPART1", "PROTPART0")
    m.train(df_train["x"], df_train["y"])
    actual = m.predict(df_test["x"])
    pos_f1 = sklearn.metrics.f1_score(df_test["y"], actual, labels=[1, 2, 3, 4, 5, 6], average='micro',
                                      sample_weight=None, zero_division='warn')
    all_f1 = sklearn.metrics.f1_score(df_test["y"], actual, average='micro', sample_weight=None, zero_division='warn')
    print(sklearn.metrics.classification_report(df_test["y"],
                                                actual,
                                                output_dict=False,
                                                labels=[1, 2, 3, 4, 5, 6]))

    # print(sklearn.metrics.classification_report(df_test["y"],
    #                                             actual,
    #                                             output_dict=False))
    print("Pos labels", pos_f1, "All labels", all_f1)


if __name__ == "__main__":
    run_main()
