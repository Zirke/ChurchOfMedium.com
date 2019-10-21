from fastcompile import get_dataset
import random
import tensorflow as tf

# def get_array_from_dataset(dataset):
#     image_arr, label_arr, count = [],[], 0
#     for image,label in dataset:
#         image_arr.append(image)
#         label_arr.append(label)
#         count += 1
#     return image_arr,label_arr,count
#
# def len_dataset(dataset):
#     count = 0
#     for x in dataset:
#         count+=1
#     return count

def shuffle(image_array, label_array, max_len_dataset):
    n_image_array,n_label_array = [],[]
    indices = []
    i = 0
    #initialise indices to contain the indices of the old dataset
    while i < max_len_dataset:
        indices.append(i)
        i+=1
    #Randomize old indices
    for x in range(5):
        random.shuffle(indices)
    #create arrays based on indices array
    for x in indices:
        n_image_array.append(image_array[x])
        n_label_array.append(label_array[x])
    #convert to Dataset object
    #dataset = tf.data.Dataset.from_tensor_slices((n_image_array, n_label_array))
    return n_image_array,n_label_array