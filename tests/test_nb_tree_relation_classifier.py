from unittest import TestCase

from model_nb_tree_relation_classifier import ModelNBTreeRelationClassifier


class TestNBTreeRelationClassifier(TestCase):

    def test_predict(self):
        # Arrange
        sut = ModelNBTreeRelationClassifier("M1", "M2", min_df=1)
        x = [
            "M1 is located in M2",
            "M1 is not located in M2"
        ]
        y = [
            1,
            0
        ]

        sut.train(x, y)

        # Act
        actual = sut.predict(x)

        self.assertEqual(len(actual), len(y))
