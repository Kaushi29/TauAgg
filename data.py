import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import skimage.io

import tensorflow as tf

"""
TODO:
  - add progress bar
  - parameterize features, dtype, etc
  - 
"""

ROBODATA_PATH = '/media/finkbeinernas/robodata'
DATA_PATH = os.path.join(ROBODATA_PATH, 'tau_agg_group', 'data', 'model_data')

DATA_LABELS = 'AAV_Lenti CRY2tau'.split()
data_folders = list(map(lambda l: os.path.join(DATA_PATH, l), DATA_LABELS))
tfrec_folder = os.path.join(DATA_PATH, 'tfrecords')

class TFRec:
    def __init__(self, tfrec_folder, data_folders, data_lbls=None):
        assert os.path.exists(tfrec_folder), 'Invalid tfrec folder'
        assert False not in [os.path.exists(data_folder) for data_folder in data_folders], 'Invalid data folders'
        assert data_lbls is None or len(data_lbls) == len(data_folders)

        self.tfrec_folder = tfrec_folder
        self.data_folders = data_folders
        if data_lbls is None: self.data_lbls = list(range(len(data_folders)))
        else: self.data_lbls = data_lbls

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def load_image(self, filename):
        assert os.path.exists(filename), 'Invalid image path'

        return skimage.io.imread(filename)

    def tfrec_write(self, tfrec_file):
        try:
            with tf.io.TFRecordWriter(os.path.join(self.tfrec_folder, tfrec_file)) as writer:
                for data_folder, data_lbl in zip(self.data_folders, self.data_lbls):
                    for filename in next(os.walk(data_folder))[2]:
                        image = self.load_image(os.path.join(data_folder, filename))
                        image_dims = image.shape
                        assert len(image_dims) == 2
                        assert image.dtype == np.uint16

                        feature = {'filename': self._bytes_feature(tf.compat.as_bytes(filename)),
                                   'label': self._int64_feature(data_lbl),
                                   'image': self._bytes_feature(tf.compat.as_bytes(image.tobytes())),
                                   'height': self._int64_feature(image_dims[0]),
                                   'width': self._int64_feature(image_dims[1])
                        }

                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        writer.write(example.SerializeToString())
            return True
        except Exception as e:
            print(e)
            return False



if __name__ == '__main__':
    t = TFRec(tfrec_folder, data_folders)
    if t.tfrec_write('test.tfrecord'):
        print('Success! Record file written')

##############
"""
class Record:2

    def __init__(self, images_dir_A, tfrecord_dir, lbl):
        self.p = param.Param()
        self.images_dir_A = images_dir_A
        # Add dummy folder for batch two, different tree.
        self.impaths_A = glob.glob(os.path.join(self.images_dir_A, '*.tif'))

        self.tfrecord_dir = tfrecord_dir

        self.impaths = np.array(self.impaths_A)
        self.lbls = np.array([lbl for i in range(len(self.impaths_A))])
        assert len(self.impaths) == len(self.lbls), 'Length of images and labels do not match.'

    def load_image(self, im_path):
        img = imageio.imread(im_path)
        # assume it's the correct size, otherwise resize here
        img = img.astype(np.float32)
        return img

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def tiff2record(self, tf_data_name, filepaths, labels):
        
        assert len(filepaths) == len(labels), 'len of filepaths and labels do not match {} {}'.format(len(filepaths),
                                                                                                      len(labels))
        with tf.io.TFRecordWriter(os.path.join(self.tfrecord_dir, tf_data_name)) as writer:
            for i in range(len(filepaths)):
                # one less in range for matching pairs
                if not i % 100:
                    print('Train data:', i)  # Python 3 has default end = '\n' which flushes the buffer
                #                sys.stdout.flush()
                filename = str(filepaths[i])

                img = self.load_image(filename)

                label = labels[i]
                filename = str(filename)
                filename = str.encode(filename)
                ratio = 0.0

                feature = {'label': self._int64_feature(label),
                           'ratio': self._float_feature(ratio),
                           'image': self._bytes_feature(tf.compat.as_bytes(img.tostring())),
                           'filename': self._bytes_feature(filename)}
                # feature = {'image': self._bytes_feature(tf.compat.as_bytes(img.tostring())),
                #            'label': self._int64_feature(label)}

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        print('Saved to ' + os.path.join(self.tfrecord_dir, tf_data_name))

        sys.stdout.flush()


if __name__ == '__main__':
    p = param.Param()
    lbl = 1
    # if lbl ==1:
    #   lblstr = 'positive'
    # elif lbl==0:
    #    lblstr= 'negative'
    use_dir = '/media/finkbeinernas/robodata/Thinking_Microscope/Neurites50-10/AblatedImages/*/*'

    Rec = Record(use_dir, p.tfrecord_dir, lbl)
    savedeploy = os.path.join(p.tfrecord_dir, use_dir.split('/')[-4] + '_' + use_dir.split('/')[-1] + '.tfrecord')
    Rec.tiff2record(savedeploy, Rec.impaths, Rec.lbls)
"""