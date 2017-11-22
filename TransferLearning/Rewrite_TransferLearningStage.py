from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import hashlib
import json
import os
import re
import struct
import sys
import tarfile
import time

import numpy as np
from six.moves import urllib
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys

from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.metrics.python.ops import metric_ops
from sklearn.metrics import classification_report

FLAGS = tf.app.flags.FLAGS
LABELS_FILENAME = "output_labels.json"

# comment out for less info during the training runs.
tf.logging.set_verbosity(tf.logging.INFO)

# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and
# their sizes. If you want to adapt this script to work with another model,
# you will need to update these to reflect the values in the network
# you're using.
# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

LABEL_DICT = {'2':'roses','0':'daisy','4':'tulips','3':'sunflowers','1':'dandelion'}

def Parse_train_val_file(file_dir):

    result_dict = {}

    file_path = open(file_dir)
    file_path = file_path.read().splitlines()

    for line in file_path:

        file_name, file_label = line.split(" ")
        file_name = file_name.split("/")[-1]

        if LABEL_DICT[file_label] not in result_dict.keys():

            result_dict[LABEL_DICT[file_label]] = {}
            if 'train' in file_dir:
                result_dict[LABEL_DICT[file_label]]['training'] = []
                result_dict[LABEL_DICT[file_label]]['training'].append(file_name)
            else:
                result_dict[LABEL_DICT[file_label]]['validation'] = []
                result_dict[LABEL_DICT[file_label]]['validation'].append(file_name)
                result_dict[LABEL_DICT[file_label]]['testing'] = []
                result_dict[LABEL_DICT[file_label]]['testing'].append(file_name)

            result_dict[LABEL_DICT[file_label]]['dir'] = LABEL_DICT[file_label]

        else:
            if 'train' in file_dir:
                result_dict[LABEL_DICT[file_label]]['training'].append(file_name)
            else:
                result_dict[LABEL_DICT[file_label]]['validation'].append(file_name)
                result_dict[LABEL_DICT[file_label]]['testing'].append(file_name)

    for key, value in result_dict.items():
        print(key, value)

    return result_dict

def make_image_lists_from_file(train_dict, val_dict):

    for item in LABEL_DICT.values():
        train_dict[item].update(val_dict[item])

    print('*'*100)

    print(train_dict.keys())

    for key, value in train_dict.items():
        print(value.keys())
        for inter_key, inter_value in value.items():
            print(inter_key, inter_value)


    return train_dict


def get_image_path(image_lists, label_name, index, image_dir, category):
    """"Returns a path to an image for a label at the given index.
    Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Int offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of set to pull images from - training, testing, or
    validation.
    Returns:
    File system path string to an image that meets the requested parameters.
    """
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                     label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)

    return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category):
    """"Returns a path to a bottleneck file for a label at the given index.
    Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    category: Name string of set to pull images from - training, testing, or
    validation.
    Returns:
    File system path string to an image that meets the requested parameters.
    """
    return get_image_path(image_lists, label_name, index, bottleneck_dir,
                        category) + '.txt'


def create_inception_graph(dest_dir):
    """"Creates a graph from saved GraphDef file and returns a Graph object.
    Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
    """
    with tf.Session() as sess:
        model_filename = os.path.join(
            dest_dir, 'classify_image_graph_def.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:

          graph_def = tf.GraphDef()
          graph_def.ParseFromString(f.read())
          bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
              tf.import_graph_def(graph_def, name='', return_elements=[
                  BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                  RESIZED_INPUT_TENSOR_NAME]))
    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer.
    Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    bottleneck_tensor: Layer before the final softmax.
    Returns:
    Numpy array of bottleneck values.
    """
    bottleneck_values = sess.run(
      bottleneck_tensor,
      {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)

    return bottleneck_values


def maybe_download_and_extract(dest_dir='/Users/yanyunliu/PycharmProjects/TensorFlowTutorial/'):
    """Download and extract model tar file.
    If the pretrained model we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a directory.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_dir, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                           (filename,
                            float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL,
                                                 filepath,
                                                 _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    tarfile.open(filepath, 'r:gz').extractall(dest_dir)


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.
    Args:
    dir_name: Path string to the folder we want to create.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def write_list_of_floats_to_file(list_of_floats, file_path):
    """Writes a given list of floats to a binary file.
    Args:
    list_of_floats: List of floats we want to write to a file.
    file_path: Path to a file where list of floats will be stored.
    """

    s = struct.pack('d' * BOTTLENECK_TENSOR_SIZE, *list_of_floats)
    with open(file_path, 'wb') as f:
        f.write(s)


def read_list_of_floats_from_file(file_path):
    """Reads list of floats from a given file.
    Args:
    file_path: Path to a file where list of floats was stored.
    Returns:
    Array of bottleneck values (list of floats).
    """

    with open(file_path, 'rb') as f:
        s = struct.unpack('d' * BOTTLENECK_TENSOR_SIZE, f.read())
    return list(s)


bottleneck_path_2_bottleneck_values = {}


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             bottleneck_tensor):
    """Retrieves or calculates bottleneck values for an image.
    If a cached version of the bottleneck data exists on-disk, return that,
    otherwise calculate the data and save it to disk for future use.
    Args:
    sess: The current active TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be modulo-ed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string  of the subfolders containing the training
    images.
    category: Name string of which  set to pull images from: training, testing,
    or validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: The tensor to feed loaded jpeg data into.
    bottleneck_tensor: The output tensor for the bottleneck values.
    Returns:
    Numpy array of values produced by the bottleneck layer for the image.
    """
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                        bottleneck_dir, category)
    if not os.path.exists(bottleneck_path):
        print('Creating bottleneck at ' + bottleneck_path)
        image_path = get_image_path(image_lists, label_name, index, image_dir,
                                    category)
        if not gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottleneck_values = run_bottleneck_on_image(sess, image_data,
                                                    jpeg_data_tensor,
                                                    bottleneck_tensor)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)

    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, bottleneck_tensor):
    """Ensures all the training, testing, and validation bottlenecks are cached.
    Because we're likely to read the same image multiple times (if there are no
    distortions applied during training) it can speed things up a lot if we
    calculate the bottleneck layer values once for each image during
    preprocessing, and then just read those cached values repeatedly during
    training. Here we go through all the images we've found, calculate those
    values, and save them off.
    Args:
    sess: The current active TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    image_dir: Root folder string of the subfolders containing the training
    images.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: Input tensor for jpeg data from file.
    bottleneck_tensor: The penultimate output layer of the graph.
    Returns:
    Nothing.
    """
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(sess, image_lists, label_name, index,
                                         image_dir, category, bottleneck_dir,
                                         jpeg_data_tensor, bottleneck_tensor)
                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    print(str(how_many_bottlenecks) + ' bottleneck files created.')


def get_all_cached_bottlenecks(
    sess, image_lists, category, bottleneck_dir, image_dir, jpeg_data_tensor,
    bottleneck_tensor):

    bottlenecks = []
    ground_truths = []
    label_names = list(image_lists.keys())
    for label_index in range(len(label_names)):
        label_name = label_names[label_index]
        for image_index in range(len(image_lists[label_name][category])):
            bottleneck = get_or_create_bottleneck(
              sess, image_lists, label_name, image_index, image_dir, category,
              bottleneck_dir, jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(len(label_names), dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)

    return bottlenecks, ground_truths


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


def add_final_training_ops(
    class_count, mode, final_tensor_name,
    bottleneck_input, ground_truth_input):
    """Adds a new softmax and fully-connected layer for training.
    We need to retrain the top layer to identify our new classes, so this
    function adds the right operations to the graph, along with some variables
    to hold the weights, and then sets up all the gradients for the backward
    pass.
    The set up for the softmax and fully-connected layers is based on:
    https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
    Args:
    class_count: Integer of how many categories of things we're trying to
    recognize.
    final_tensor_name: Name string for the new final node that produces
    results.
    bottleneck_tensor: The output of the main CNN graph.
    Returns:
    The tensors for the training and cross entropy results, and tensors for the
    bottleneck input and ground truth input.
    """

  # Organizing the following ops as `final_training_ops` so they're easier
  # to see in TensorBoard
    train_step = None
    cross_entropy_mean = None

    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            layer_weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count],
                  stddev=0.001), name='final_weights')
            variable_summaries(layer_weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram(layer_name + '/pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    tf.summary.histogram(final_tensor_name + '/activations', final_tensor)

    if mode in [ModeKeys.EVAL, ModeKeys.TRAIN]:
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
              logits=logits, labels=ground_truth_input)
            with tf.name_scope('total'):
                cross_entropy_mean = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('cross entropy', cross_entropy_mean)

        with tf.name_scope('train'):
            train_step = tf.train.GradientDescentOptimizer(
                learning_rate).minimize(
                  cross_entropy_mean,
                  global_step=tf.contrib.framework.get_global_step())

    return (train_step, cross_entropy_mean, final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results.
    Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.
    Returns:
    Nothing.
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(result_tensor, 1), \
                                    tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step


def make_model_fn(class_count, final_tensor_name):

    def _make_model(bottleneck_input, ground_truth_input, mode, params):

        prediction_dict = {}
        train_step = None
        cross_entropy = None

        # Add the new layer that we'll be training.
        (train_step, cross_entropy,
         final_tensor) = add_final_training_ops(
            class_count, mode, final_tensor_name,
            bottleneck_input, ground_truth_input)


        if mode == ModeKeys.EVAL:
            prediction_dict['loss'] = cross_entropy
            # Create the operations we need to evaluate accuracy
            acc = add_evaluation_step(final_tensor, ground_truth_input)
            prediction_dict['accuracy'] = acc

        if mode == ModeKeys.INFER:
            predclass = tf.argmax(final_tensor, 1)
            prediction_dict["class_vector"] = final_tensor
            prediction_dict["index"] = predclass

        return prediction_dict, cross_entropy, train_step

    return _make_model

METRICS = {
    'loss': metric_spec.MetricSpec(
        metric_fn=metric_ops.streaming_mean,
        prediction_key='loss'
    ),
    'accuracy': metric_spec.MetricSpec(
        metric_fn=metric_ops.streaming_mean,
        prediction_key='accuracy'
    )
}


def make_image_predictions(
    classifier, jpeg_data_tensor, bottleneck_tensor, path_list, labels_list):
    """Use the learned model to make predictions."""

    if not labels_list:
        output_labels_file = os.path.join(model_dir, LABELS_FILENAME)
    if gfile.Exists(output_labels_file):
        with open(output_labels_file, 'r') as lfile:
            labels_string = lfile.read()
            labels_list = json.loads(labels_string)
            print("labels list: %s" % labels_list)
    else:
        print("Labels list %s not found" % output_labels_file)
        return None

    sess = tf.Session()
    bottlenecks = []
    print("Predicting for images: %s" % path_list)
    for img_path in path_list:
        # get bottleneck for an image path. Don't cache the bottleneck values here.
        if not gfile.Exists(img_path):
            tf.logging.fatal('File does not exist %s', img_path)
        image_data = gfile.FastGFile(img_path, 'rb').read()
        bottleneck_values = run_bottleneck_on_image(sess, image_data,
                                                    jpeg_data_tensor,
                                                    bottleneck_tensor)
        bottlenecks.append(bottleneck_values)
    prediction_input = np.array(bottlenecks)
    predictions = classifier.predict(x=prediction_input, as_iterable=True)

    prediction_result = []
    class_vector_list = []

    print(predictions)
    print("Predictions:")

    for num_index, p in enumerate(predictions):
        print("---------", num_index, path_list[num_index])
        for k in p.keys():
            print("%s is: %s " % (k, p[k]))
            if k == "index":
                print("index label is: %s" % labels_list[p[k]])
                prediction_result.append((labels_list[p[k]], path_list[num_index]))
            elif k == "class_vector":
                print("class vector is:", p[k], path_list[num_index])
                vector_list = list(p[k])
                vector_list.append(path_list[num_index])
                class_vector_list.append(vector_list)


    return prediction_result, class_vector_list



def get_prediction_images(img_dir):
    """Grab images from the prediction directory."""
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    if not gfile.Exists(img_dir):
        print("Image directory '" + img_dir + "' not found.")
        return None
    print("Looking for images in '" + img_dir + "'")
    for extension in extensions:
        file_glob = os.path.join(img_dir, '*.' + extension)
        file_list.extend(glob.glob(file_glob))
    if not file_list:
        print('No image files found')
    return file_list

def Convert_prediction_to_array(prediction_DF, test_file):

    inverse_label_dict = dict(map(lambda (x,y): (y,x), LABEL_DICT.items()))
    print(inverse_label_dict)

    prediction_DF = pd.read_csv(prediction_DF, sep=" ")
    prediction_DF['data'] = prediction_DF['data'].apply(lambda x: inverse_label_dict[x])
    prediction_DF['label'] = prediction_DF['label'].apply(lambda x: x.split("/")[-1])


    print(prediction_DF['data'])

    test_file = pd.read_csv(test_file, header= None, names=['label'])

    test_file['label'] = test_file['label'].apply(lambda x: x.split("/")[-1])

    test_file = test_file.merge(prediction_DF)

    test_file['data'] = test_file['data'].astype(int)

    np.savetxt('TF_prediction_array_v2.txt',test_file['data'].values,fmt='%1.0f')

    print(test_file)
    #pass


def Check_test_accuracy(truth_label, prediction):

    truth_label = np.loadtxt(truth_label).ravel()
    prediction = np.loadtxt(prediction).ravel()

    print(accuracy_score(truth_label, prediction))


def main():

    train_dict = Parse_train_val_file('/Users/yanyunliu/PycharmProjects/TensorFlowTutorial/data/train.txt')
    val_dict = Parse_train_val_file('/Users/yanyunliu/PycharmProjects/TensorFlowTutorial/data/val.txt')
    make_image_lists_from_file(train_dict, val_dict)

    print("Using model directory %s" % model_dir)

    # Set up the pre-trained graph.
    maybe_download_and_extract(dest_dir=incp_model_dir)
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph(incp_model_dir))

    sess = tf.Session()
    labels_list = None


    # Look at the folder structure, and create lists of all the images.
    image_lists = make_image_lists_from_file(train_dict, val_dict)

    class_count = len(image_lists.keys())

    print(class_count)

    cache_bottlenecks(
        sess, image_lists, image_dir, bottleneck_dir,
        jpeg_data_tensor, bottleneck_tensor)

    model_fn = make_model_fn(class_count, final_tensor_name)
    model_params = {}
    classifier = tf.contrib.learn.Estimator(
        model_fn=model_fn, params=model_params, model_dir=model_dir)



    train_bottlenecks, train_ground_truth = get_all_cached_bottlenecks(
        sess, image_lists, 'training',
        bottleneck_dir, image_dir, jpeg_data_tensor,
        bottleneck_tensor)
    train_bottlenecks = np.array(train_bottlenecks)
    print('*'*100)
    print(train_bottlenecks.shape)
    train_ground_truth = np.array(train_ground_truth)

    # then run the training, unless doing prediction only
    print("Starting training for %s steps max" % num_steps)
    classifier.fit(
        x=train_bottlenecks.astype(np.float32),
        y=train_ground_truth, batch_size=50,
        max_steps=num_steps)


    val_bottlenecks, val_ground_truth = get_all_cached_bottlenecks(
        sess, image_lists, 'validation',
        bottleneck_dir, image_dir, jpeg_data_tensor,
        bottleneck_tensor)
    val_bottlenecks = np.array(val_bottlenecks)

    print('*' * 100)
    print(val_bottlenecks.shape)

    val_ground_truth = np.array(val_ground_truth)

    print("evaluating....")
    print(classifier.evaluate(
        val_bottlenecks.astype(np.float32), val_ground_truth, metrics=METRICS))

    print("\nPredicting...")
    img_list = get_prediction_images(prediction_img_dir)
    if not img_list:
        print("No images found in %s" % prediction_img_dir)
    else:
        prediction_output, class_vector_list = make_image_predictions(
            classifier, jpeg_data_tensor, bottleneck_tensor, img_list, labels_list)
        print(prediction_output)

        prediction_result = pd.DataFrame.from_records(prediction_output, columns=['data', 'label'])
        prediction_result.to_csv('/Users/yanyunliu/PycharmProjects/TensorFlowTutorial/MSBD6000BProject2/TransferLearning/transfer_learning_predict',
                                 sep=' ', index=False)

        class_vectorDf = pd.DataFrame(class_vector_list)
        class_vectorDf.to_csv('/Users/yanyunliu/PycharmProjects/TensorFlowTutorial/MSBD6000BProject2/TransferLearning/transfer_learning_class_vectorDf',
                              index=False)



if __name__ == '__main__':

    image_dir = '/Users/yanyunliu/PycharmProjects/TensorFlowTutorial/data/flower_photos'
    model_dir = '/Users/yanyunliu/PycharmProjects/TensorFlowTutorial/MSBD6000BProject2/TransferLearning/TFModel'
    incp_model_dir = '/Users/yanyunliu/PycharmProjects/TensorFlowTutorial/MSBD6000BProject2/TransferLearning'
    bottleneck_dir = '/Users/yanyunliu/PycharmProjects/TensorFlowTutorial/MSBD6000BProject2/TransferLearning'
    final_tensor_name = 'final_result'
    prediction_img_dir = '/Users/yanyunliu/PycharmProjects/TensorFlowTutorial/data/flower_photos/test'
    # prediction_img_dir = '/Users/yanyunliu/PycharmProjects/TensorFlowTutorial/data/flower_photos/test'
    num_steps = 15000
    learning_rate = 0.01


    main()

    Convert_prediction_to_array('/Users/yanyunliu/PycharmProjects/TensorFlowTutorial/MSBD6000BProject2/TransferLearning/transfer_learning_predict',
                                '/Users/yanyunliu/PycharmProjects/TensorFlowTutorial/data/test.txt')

    Check_test_accuracy('/Users/yanyunliu/PycharmProjects/TensorFlowTutorial/sample_truth_labelonly', '/Users/yanyunliu/PycharmProjects/TensorFlowTutorial/MSBD6000BProject2/TransferLearning/TF_prediction_array_v2.txt')



