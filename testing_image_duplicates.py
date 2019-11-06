import unittest
import numpy as np
from collections import Counter
from data_Processing.binary_pre_processing import *
from sorting_hub import *


class MyTestCase(unittest.TestCase):
    def test_image_duplicates(self):
        training_data, validation_data, testing_data = process_data(five_diagnosis_paths)
        array = []
        for image, label in training_data:
            image = image.numpy()
            new_image = image.flatten()
            array.append(new_image)
        for image, label in validation_data:
            image = image.numpy()
            new_image = image.flatten()
            array.append(new_image)
        for image, label in testing_data:
            image = image.numpy()
            new_image = image.flatten()
            array.append(new_image)

        duplicate_counter = 0
        outer = 0
        inner = 1
        while outer <= len(array) - 1:
            while inner <= len(array):
                if np.all(array[outer] == array[inner]):
                    duplicate_counter += 1
                    inner += 1
            outer += 1

        self.assertTrue(duplicate_counter == 0)


if __name__ == '__main__':
    unittest.main()
