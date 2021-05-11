import tensorflow as tf
from datasets.utils import unpickle
import sys
import os

data_path = '/home/dingyf/lwy/Deep-Mutual-Learning-master/data/cifar-100-python'

tfrecord_path = '/home/dingyf/lwy/Deep-Mutual-Learning-master/data/cifar100-tfrecord'

split = ['test', 'train']

_IMAGE_HEIGHT = 32
_IMAGE_WIDTH = 32
_IMAGE_CHANNELS = 3


def int64_feature(values):
    """Returns a TF-Feature of int64s.

    Args:
      values: A scalar or list of values.

    Returns:
      a TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
      values: A string.

    Returns:
      a TF-Feature.
    """
    if isinstance(values, str):
      values = values.encode()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, class_id, height, width, image_format):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/label': int64_feature(class_id),
        # 'image/filename': bytes_feature(filename),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/format': bytes_feature(image_format),
    }))


def _add_to_tfrecord(images, labels, tfrecord_writer, split_name):
    """Loads images and writes files to a TFRecord.

    Args:
      image_dir: The image directory where the raw images are stored.
      list_filename: The list file of images.
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    # num_images = len(tf.gfile.FastGFile(list_filename, 'r').readlines())
    num_images = len(images)

    shape = (_IMAGE_HEIGHT, _IMAGE_WIDTH, _IMAGE_CHANNELS)

    with tf.Graph().as_default():
        image = tf.placeholder(dtype=tf.uint8, shape=shape)
        encoded_png = tf.image.encode_png(image)
        j = 0
        with tf.Session('') as sess:
            # for line in tf.gfile.FastGFile(list_filename, 'r').readlines():
            for idx, image_data in enumerate(images):
                sys.stdout.write('\r>> Converting %s image %d/%d' % (split_name, j + 1, num_images))
                sys.stdout.flush()
                j += 1
                # imagename, label = line.split(' ')
                label = labels[idx]
                # file_path = os.path.join(image_dir, imagename)
                # image_data = misc.imread(file_path)
                # image_data = misc.imresize(image_data, [_IMAGE_HEIGHT, _IMAGE_WIDTH])
                # image_data = line
                png_string = sess.run(encoded_png, feed_dict={image: image_data})
                example = image_to_tfexample(png_string, label, _IMAGE_HEIGHT, _IMAGE_WIDTH, 'png')
                tfrecord_writer.write(example.SerializeToString())


for s in split:
    data = unpickle(data_path + '/' + s)
    x = data[b'data']
    labels = data[b'fine_labels']
    x_input = x.reshape(x.shape[0], 3, 32, 32)
    image = x_input.transpose(0, 2, 3, 1)
    # tf_filename = '{}/{}.tfrecord'.format(tfrecord_path, s)
    tf_filename = os.path.join(tfrecord_path, '{}.tfrecord'.format(s))
    # input(tf_filename)
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        _add_to_tfrecord(image, labels, tfrecord_writer, s)
    print(tf_filename + 'Done!')
    # if not os.path.exists('./trans/tran.tfrecords'):
    #     generate_tfrecord(image, label, './trans/', 'tran.tfrecords')
    #     # generate_tfrecord(x_test, y_test, './trans/', 'test.tfrecords')


