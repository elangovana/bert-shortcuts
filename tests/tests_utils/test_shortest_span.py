from unittest import TestCase

from utils.shortest_span import ShortestSpan


class TestShortestSpan(TestCase):

    def test__call__(self):
        sut = ShortestSpan()
        input_sentence = "rivaroxaban and _CHEMICAL_ , are potent, oral direct inhibitors of _GENE_-bound, clot-associated or free FXa	"
        marker_1 = "_CHEMICAL_"
        marker_2 = "_GENE_"
        trigger = "inhibitors"

        expected = 7

        # Act
        actual = sut(input_sentence, [marker_1, marker_2, trigger])

        # Assert
        self.assertEqual(expected, actual)
