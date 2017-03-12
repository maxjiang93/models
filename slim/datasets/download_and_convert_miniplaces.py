# Copyright 2017 Chiyu 'Max' Jiang
# ==============================================================================
r"""Downloads and converts Miniplaces data to TFRecords of TF-Example protos.

This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import csv

import tensorflow as tf

from datasets import dataset_utils

# The URL where the Miniplaces data can be downloaded.
_DATA_URL = 'http://dl.caffe.berkeleyvision.org/mit_mini_places/data.tar.gz'

# The URL where the Development Toolkit can be downloaded.
_TOOL_URL = 'http://dl.caffe.berkeleyvision.org/mit_mini_places/development_kit.tar.gz'

# Seed for repeatability
_RANDOM_SEED = 0

class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_dataset_filename(dataset_dir, split_name):
    output_filename = 'miniplaces_%s.tfrecord' % split_name
    return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, labels, dataset_dir):
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      filenames: A list of absolute paths to png or jpg images.
      class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
      dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'validation']

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            output_filename = _get_dataset_filename(dataset_dir, split_name)

            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                start_ndx = 0
                end_ndx = len(filenames)
                for i in range(start_ndx, end_ndx):
                    sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                    sys.stdout.flush()

                    # Read the filename:
                    image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
                    height, width = image_reader.read_image_dims(sess, image_data)

                    # class_name = classname[labels[i]]
                    class_id = labels[i]

                    example = dataset_utils.image_to_tfexample(
                        image_data, 'jpg', height, width, class_id)
                    tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def _clean_up_temporary_files(dataset_dir):
    """Removes temporary files used to create the dataset.

    Args:
      dataset_dir: The directory where the temporary files are stored.
    """
    filename = _DATA_URL.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)
    tf.gfile.Remove(filepath)

    tmp_dir1 = os.path.join(dataset_dir, 'development_kit')
    tmp_dir2 = os.path.join(dataset_dir, 'images')
    tmp_dir3 = os.path.join(dataset_dir, 'objects')
    tf.gfile.DeleteRecursively(tmp_dir1)
    tf.gfile.DeleteRecursively(tmp_dir2)
    tf.gfile.DeleteRecursively(tmp_dir3)


def _dataset_exists(dataset_dir):
    for split_name in ['train', 'validation']:
        output_filename = _get_dataset_filename(dataset_dir, split_name)
        if not tf.gfile.Exists(output_filename):
            return False
    return True


def get_train_val(dataset_dir):
    train_txt = os.path.join(dataset_dir, 'development_kit/data/train.txt')
    val_txt =  os.path.join(dataset_dir, 'development_kit/data/val.txt')
    # training data
    train_reader = csv.reader(open(train_txt), delimiter=" ")
    training_filenames, training_labels = [], []
    for row in train_reader:
        training_filenames.append(dataset_dir+'/images/'+row[0])
        training_labels.append(int(row[1]))
    # validation data
    valid_reader = csv.reader(open(val_txt), delimiter=" ")
    validation_filenames, validation_labels = [], []
    for row in valid_reader:
        validation_filenames.append(dataset_dir+'/images/'+row[0])
        validation_labels.append(int(row[1]))
    return training_filenames, training_labels, validation_filenames, validation_labels


def get_class_names(dataset_dir):
    class_names_txt = os.path.join(dataset_dir, 'development_kit/data/categories.txt')
    class_name = []
    class_reader = csv.reader(open(class_names_txt), delimiter=" ")
    for row in class_reader:
        class_name.append(row[0])
    return class_name


def run(dataset_dir):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    if _dataset_exists(dataset_dir):
      print('Dataset files already exist. Exiting without re-creating them.')
      return

    dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
    dataset_utils.download_and_uncompress_tarball(_TOOL_URL, dataset_dir)
    training_filenames, training_labels, validation_filenames, validation_labels = get_train_val(dataset_dir)
    class_names = get_class_names(dataset_dir)

    # Shuffle training and validation
    random.seed(_RANDOM_SEED)
    ztrain = zip(training_filenames, training_labels)
    zval = zip(validation_filenames, validation_labels)
    random.shuffle(ztrain)
    random.shuffle(zval)
    training_filenames, training_labels = zip(*ztrain)
    validation_filenames, validation_labels = zip(*zval)

    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames, training_labels, dataset_dir)
    _convert_dataset('validation', validation_filenames, validation_labels, dataset_dir)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    _clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the Miniplaces dataset!')
