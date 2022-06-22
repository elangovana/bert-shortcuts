import re

from nltk import PorterStemmer


class ShortestSpan:
    def __init__(self, stemmer=None):
        self.stemmer = stemmer or PorterStemmer()

    def __call__(self, sentence, words_to_match):
        words = re.split('\W+', sentence)

        stemmed_words = [self.stemmer.stem(w) for w in words]
        subwords = [self.stemmer.stem(w) for w in words_to_match]
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
