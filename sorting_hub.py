from data_Processing.sort_tfrecords import *

"""
This file is where all the paths for different tfrecords are stored. 

This is also where all sorted tf records are created. The sorting algorithms create binary classification splits 
for all categories displayed below. To see the exact split in the data go to data_Processing and the algorithms are
in the files. 

All files are fairly high in volume and take a bit of space, each is written above a given statement. 
"""
def make_dirs(filepath):
    import os
    import errno
    try:
        os.makedirs(filepath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def make_sorted_tfrecords():
    make_dirs('sorted_tfrecords')
    make_dirs('sorted_tfrecords/benign_cal')
    make_dirs('sorted_tfrecords/benign_mass')
    make_dirs('sorted_tfrecords/malignant_cal')
    make_dirs('sorted_tfrecords/malignant_mass')
    make_dirs('sorted_tfrecords/negative_binary')
    make_dirs('sorted_tfrecords/five_diagnosis')



# Different sorting of tfrecords.
sorting_algorithms = ['negative_bi', 'benign_cal_split', 'benign_mass_split',
                      'malignant_cal_split', 'malignant_mass_split', 'five_diagnosis']

# The two lines below create a new file with negative/others with even distribution and loads it into a dataset.
negative_bi_file_paths = ['sorted_tfrecords/negative_binary/negative_binary_training',
                          'sorted_tfrecords/negative_binary/negative_binary_val.tfrecord',
                          'sorted_tfrecords/negative_binary/negative_binary_test.tfrecord']

benign_cal_split_paths = ['sorted_tfrecords/benign_cal/benign_cal_split_training',
                          'sorted_tfrecords/benign_cal/benign_cal_split_val.tfrecord',
                          'sorted_tfrecords/benign_cal/benign_cal_split_test.tfrecord']

benign_mass_split_paths = ['sorted_tfrecords/benign_mass/benign_mass_split_training',
                           'sorted_tfrecords/benign_mass/benign_mass_split_val.tfrecord',
                           'sorted_tfrecords/benign_mass/benign_mass_split_test.tfrecord']

malignant_cal_split_paths = ['sorted_tfrecords/malignant_cal/malignant_cal_split_training',
                             'sorted_tfrecords/malignant_cal/malignant_cal_split_val.tfrecord',
                             'sorted_tfrecords/malignant_cal/malignant_cal_split_test.tfrecord']

malignant_mass_split_paths = ['sorted_tfrecords/malignant_mass/malignant_mass_split_training',
                              'sorted_tfrecords/malignant_mass/malignant_mass_split_val.tfrecord',
                              'sorted_tfrecords/malignant_mass/malignant_mass_split_test.tfrecord']

five_diagnosis_paths = ['sorted_tfrecords/five_diagnosis/five_diagnosis_split_training.tfrecord',
                        'sorted_tfrecords/five_diagnosis/five_diagnosis_split_val.tfrecord',
                        'sorted_tfrecords/five_diagnosis/five_diagnosis_split_test.tfrecord', ]


# make the right directories
make_sorted_tfrecords()

# if __name__ == '__main__':
# binary_classification(negative_bi_file_paths, sorting_algorithms[0])

# Creates 3 files of BENIGN CALCIFICATION, train, val, test. Roughly ~1.4gb data.
# binary_classification(benign_cal_split_paths, sorting_algorithms[1])

# Creates 3 files of BENIGN MASS, train, val, test. Roughly ~1.35gb data.
# binary_classification(benign_mass_split_paths, sorting_algorithms[2])

# Creates 3 files of MALIGNANT CALCIFICATION, train, val, test. Roughly ~1.15gb data.
# binary_classification(malignant_cal_split_paths, sorting_algorithms[3])

# Creates 3 files of MALIGNANT MASS, train, val, test. Roughly ~1.2gb data.
#binary_classification(malignant_mass_split_paths, sorting_algorithms[4])

#binary_classification(five_diagnosis_paths, sorting_algorithms[5])


