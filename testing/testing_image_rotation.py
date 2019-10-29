import unittest

from data_Processing.data_augmentation import *


class DataAugmentationTest(unittest.TestCase):
    def test_rotation90(self):
        input_array = []
        initial_matrix = np.random.randint(254, size=(299, 299))
        input_array.append(initial_matrix)
        test_labels = np.random.randint(4, size=1)

        rotated90_matrix, test_labels = flip_image_90(input_array, test_labels)
        self.assertEqual(input_array[0][0][0], rotated90_matrix[0][298][0])

        return rotated90_matrix, input_array, test_labels,

    def test_rotation180(self):
        rotated90_matrix, input_array, test_labels = self.test_rotation90()
        rotated180_matrix, test_labels = flip_image_90(rotated90_matrix, test_labels)

        self.assertEqual(input_array[0][0][0], rotated180_matrix[0][298][298])

        return rotated180_matrix, input_array, test_labels

    def test_rotation270(self):
        rotated180_matrix, input_array, test_labels = self.test_rotation180()
        rotated270_matrix, test_labels = flip_image_90(rotated180_matrix, test_labels)

        self.assertEqual(input_array[0][0][0], rotated270_matrix[0][0][298])

    if __name__ == '__main__':
        unittest.main()
