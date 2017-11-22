from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import os
import time
import scipy.misc
from sklearn.model_selection import KFold


def Train_test_split_cv(train_data, train_label):

    use_cv = 5
    kf = KFold(n_splits=use_cv, shuffle=True)
    kf.get_n_splits(train_data)
    train_test_fold = kf.split(train_data)

    print type(train_test_fold)

    return train_test_fold


def ParseImageFile(file_path):

    file_list, label_list = [], []

    f = open(file_path)
    f = f.readlines()

    for line in f:
        filename, label = line.strip().split(" ")
        file_list.append(filename), label_list.append(label)

    return file_list, label_list

def ParseTestImageFile(file_path):

    file_list = []

    f = open(file_path)
    f = f.readlines()

    for line in f:
        filename = line.strip()
        file_list.append(filename)

    return file_list


def LoadImage2Array(file_path, data_path, image_size = 32, use_gray = False, use_keras = True):



    file_list, label_list = ParseImageFile(file_path)

    current_path = os.getcwd()

    print current_path

    os.chdir(data_path)

    image_list = []

    count = 0

    t1 = time.time()

    for item in file_list:


        if count%100 == 0:
            print("processing image ", count, ' total ', len(file_list))

        item =  np.array(Image.open(item))
        item = scipy.misc.imresize(item, (image_size,image_size))

        if use_gray:
            item = np.dot(item[...,:3], [0.299, 0.587, 0.114])


        # item = np.dot(item[...,:3], [0.299, 0.587, 0.114])
        if not use_keras:
            item = item.flatten()
        #print item.shape, type(item[0])
        item = item.astype(np.float32)
        #print item.shape

        image_list.append(item)
        count = count + 1


    image_list = np.array(image_list)
    print(image_list.shape, image_list[0].shape)

    t2 = time.time()

    print(t2 - t1)

    label_list = map(lambda x: np.int32(x), label_list)
    label_list = np.array(label_list)



    nb_classes = 5

    if not use_keras:

        temp = np.zeros((len(label_list), nb_classes))
        temp[np.arange(len(label_list)), label_list] = 1
        label_list = temp

    print('there is label')
    print(label_list)

    os.chdir(current_path)

    return image_list, label_list

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    data_shuffle -= np.mean(data_shuffle, axis = 0)
    data_shuffle /= np.std(data_shuffle, axis = 0)

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def LoadTestImage2Array(file_path, data_path, image_size = 32, use_gray = False, use_keras = True):



    file_list = ParseTestImageFile(file_path)

    current_path = os.getcwd()

    os.chdir(data_path)

    image_list = []

    count = 0

    t1 = time.time()

    for item in file_list:


        if count%100 == 0:
            print("processing image ", count, ' total ', len(file_list))

        item =  np.array(Image.open(item))
        item = scipy.misc.imresize(item, (image_size,image_size))

        if use_gray:
            item = np.dot(item[...,:3], [0.299, 0.587, 0.114])


        # item = np.dot(item[...,:3], [0.299, 0.587, 0.114])
        if not use_keras:
            item = item.flatten()
        #print item.shape, type(item[0])
        item = item.astype(np.float32)
        #print item.shape

        image_list.append(item)
        count = count + 1


    image_list = np.array(image_list)

    print(image_list.shape, image_list[0].shape)
    t2 = time.time()
    print(t2 - t1)

    os.chdir(current_path)

    return image_list

