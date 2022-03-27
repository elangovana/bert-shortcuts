import os.path
from unittest import TestCase

from transformers import BertForSequenceClassification, BertTokenizer, BertConfig

from src.utils.predict_entailment import PredictEntailment


class TestPredictEntailment(TestCase):

    def test__call(self):
        # Arrange
        vocab_file = os.path.join(os.path.dirname(__file__), "data", "vocab.txt")
        config_bert = BertConfig()
        model = BertForSequenceClassification(config=config_bert)
        tokenizer = BertTokenizer(vocab_file=vocab_file)
        sentences_a = ["sentence1", "sentence2"]
        sentences_b = ["sentence1", "sentence2"]

        sut = PredictEntailment(model, tokenizer)

        # Act
        actual_prob, actual_gt = sut(sentences_a, sentences_b)

        # Assert
        print(actual_prob)
        self.assertEqual(len(actual_prob), len(sentences_a))
