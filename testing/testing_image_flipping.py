import unittest
import numpy as np

from data_Processing.data_augmentation import *


class MyTestCase(unittest.TestCase):
    def test_flipping(self):
        input_matrix = []
        initial_matrix = np.random.randint(254, size=(299, 299))
        input_matrix.append(initial_matrix)
        test_labels = np.random.randint(4, size=1)

        rotated90_matrix, test_labels = flip_image(input_matrix, test_labels)
        # Tests if the image top right pixel will be equal to the flipped image's top left pixel after flipping.
        self.assertEqual(input_matrix[0][0][298], rotated90_matrix[0][0][0])


if __name__ == '__main__':
    unittest.main()
