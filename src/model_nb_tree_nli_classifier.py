import numpy as np
from nltk.stem import PorterStemmer
from sklearn import tree

from model_nb_nli_classifier import ModelNBNliClassifier
from utils.shortest_span import ShortestSpan


class ModelNBTreeNliClassifier:
    """
    Relation extractor using Naive bayes and logistics regression
    """

    def __init__(self, min_df=None, max_words_per_class=10, classwise_vocab=True,
                 stop_words='english', ngram_range=(1, 3), extract_span=False,
                 trigger_words=None, max_tree_depth=5):
        self.extract_span = extract_span

        self._model_tree = tree.DecisionTreeClassifier(max_depth=max_tree_depth)
        self._num_classes = 0
        self._feature_names = []
        self.trigger_words = trigger_words or []
        self._stemmer = PorterStemmer()
        self._shortest_span_calc = ShortestSpan(self._stemmer)
        self._nb = ModelNBNliClassifier(min_df=min_df,
                                        max_words_per_class=max_words_per_class,
                                        classwise_vocab=classwise_vocab,
                                        stop_words=stop_words, ngram_range=ngram_range)
        self.features = {
            "HYL": lambda x: self._extract_num_words(x, "hypothesis"),
            "PRL": lambda x: self._extract_num_words(x, "premise"),
            "HNEG": lambda x: self._has_neg(x, "hypothesis"),
            "PNEG": lambda x: self._has_neg(x, "premise"),
            "NBC": lambda x: [p_l == "contradiction" for p_l in self._nb.predict([i["prem_hyp"] for i in x])[0]],
            "NBE": lambda x: [p_l == "entailment" for p_l in self._nb.predict([i["prem_hyp"] for i in x])[0]],
            "NBN": lambda x: [p_l == "neutral" for p_l in self._nb.predict([i["prem_hyp"] for i in x])[0]],

            "LOV": lambda x: self._num_words_overlap_hyp_prem(x)

        }

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

    def preprocess(self, x):
        for item in x:
            item["prem_hyp"] = "{},{}".format(item["premise"], item["hypothesis"])
        return x

    def train(self, x, y):
        """
        Expect each item in x to be a dict, {"hypothesis", "premise"}
        """
        x = self.preprocess(x)

        self._nb.train([i["prem_hyp"] for i in x], y)

        print(f"Extracting features..for {len(x)}")
        tree_features = self.extract_features(x)

        print(f"Completed..")
        self._model_tree.fit(tree_features, y)

    def extract_features(self, s):

        feature_list = []
        feature_names = []
        for n, f in self.features.items():
            feature_list.append(f(s))
            feature_names.append(n)

        features = np.array(feature_list).T
        self._feature_names = feature_names
        return features

    def _extract_num_words(self, list_of_items_x, key):
        return [len(l[key]) for l in list_of_items_x]

    def _has_neg(self, list_of_items_x, key):
        result = []
        for l in list_of_items_x:
            item = l[key].lower().split(" ")
            has_neg = 0
            if "not" in item:
                has_neg = 1
            elif "no" in item:
                has_neg = 1
            result.append(has_neg)
        return result

    def predict(self, x):
        x = self.preprocess(x)

        # Use  NB + logistic
        tree_features = self.extract_features(x)
        result = self._model_tree.predict(tree_features)
        result_prob = self._model_tree.predict_proba(tree_features)
        result_prob = np.max(result_prob, axis=1)

        return result, result_prob

    def _num_words_overlap_hyp_prem(self, x):
        result = []
        for xi in x:
            words_overlap = set(xi["premise"].lower().split(" ")) \
                .intersection(set(xi["hypothesis"].lower().split(" ")))

            result.append(len(words_overlap))
        return result
