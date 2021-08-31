import tensorflow as tf

"""
TODO:
  - add support for merged datasets
"""
# '/media/finkbeinernas/robodata/tau_agg_group/data/model_data/tfrecords/test.tfrecord'

class Datagen:
    def __init__(self):
        self.dataset = None
        self.it = None

    def tfrec_parse(self, example, batch=False):
        features = {'filename': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.int64),
                    'image': tf.io.FixedLenFeature([], tf.string),
                    'height': tf.io.FixedLenFeature([], tf.int64),
                    'width': tf.io.FixedLenFeature([], tf.int64)
                    }

        parsed = tf.io.parse_example(example, features) if batch else tf.io.parse_single_example(example, features)

        image = tf.io.decode_raw(parsed['image'], tf.int16)
        if batch:
            shape = tf.stack([parsed['height'], parsed['width']])
            image = tf.map_fn()
        else:
            image = tf.reshape(image, tuple(([-1] if batch else []) + [parsed['height'], parsed['width']]))

        label = parsed['label']

        return image, label

    def tfrec_parse_single(self, example):
        return self.tfrec_parse(example, batch=False)

    def tfrec_parse_batch(self, example):
        return self.tfrec_parse(example, batch=True)

    def tfrec_load_single(self, tfrec_path):
        dataset = tf.data.TFRecordDataset(tfrec_path)
        dataset = dataset.map(self.tfrec_parse_single)

        return iter(dataset)

    def tfrec_load_batch(self, tfrec_path, batch_size=None, shuffle_buffer=None):
        dataset = tf.data.TFRecordDataset(tfrec_path)
        dataset = dataset.repeat()
        if shuffle_buffer is not None: dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size if batch_size is not None else 1, drop_remainder=True)
        dataset = dataset.map(self.tfrec_parse_batch)

        return iter(dataset)

f = '/media/finkbeinernas/robodata/tau_agg_group/data/model_data/tfrecords/test.tfrecord'




