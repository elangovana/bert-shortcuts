from unittest import TestCase

from model_nb_logistic_classifier import ModelNBLogisticClassifier


class TestNBLogisticClassifier(TestCase):

    def test_predict(self):
        # Arrange
        sut = ModelNBLogisticClassifier("m1", "m2")
        x = [
            "M1 is located in M2",
            "M1 is not located in M2"
        ]
        y = [
            True,
            False
        ]

        sut.train(x, y)

        # Act
        actual = sut.predict(x)

        self.assertEqual(len(actual), len(y))
