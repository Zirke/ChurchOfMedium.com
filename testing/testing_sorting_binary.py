import unittest

from data_Processing.binary_pre_processing import *
from sorting_hub import *


class MalignantCalTest(unittest.TestCase):
    def test_sortingOfMalignantCalBinary(self):
        parsed_training_data, parsed_val_data, parsed_testing_data = process_data(malignant_cal_split_paths)
        print(parsed_training_data)
        # self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
