import unittest
import numpy as np
from collections import Counter
from data_Processing.binary_pre_processing import *
from sorting_hub import *


class MyTestCase(unittest.TestCase):
    def test_image_duplicates(self):
        one, two, three = process_data(five_diagnosis_paths)
        array = []
        for image, label in one:
            image = image.numpy()
            new_image = image.flatten()
            array.append(new_image)
        for image, label in two:
            image = image.numpy()
            new_image = image.flatten()
            array.append(new_image)
        for image, label in three:
            image = image.numpy()
            new_image = image.flatten()
            array.append(new_image)

        duplicate_counter = 0
        counter = 0
        inner = 1
        while counter <= len(array) - 1:
            while inner <= len(array):
                if np.all(array[counter] == array[inner]):
                    print("DUPLICATE!")
                    duplicate_counter += 1
                    inner += 1

            counter += 1

        print(duplicate_counter)


if __name__ == '__main__':
    unittest.main()
