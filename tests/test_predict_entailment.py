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
        self.assertEqual(len(actual_prob), len(sentences_a))

    def test_write_to_file(self):
        # Arrange
        vocab_file = os.path.join(os.path.dirname(__file__), "data", "vocab.txt")
        config_bert = BertConfig()
        model = BertForSequenceClassification(config=config_bert)
        tokenizer = BertTokenizer(vocab_file=vocab_file)
        sentences_a = ["sentence1", "sentence2"]
        sentences_b = ["sentence1", "sentence2"]

        sut = PredictEntailment(model, tokenizer)

        temp_output = os.path.join(os.path.dirname(__file__), "out.csv")

        # Act
        sut(sentences_a, sentences_b, transform_result_func=lambda a, b, c: sut.write_to_file(temp_output, a, b, c))

        # Assert
        self.assertTrue(os.path.isfile(temp_output))
