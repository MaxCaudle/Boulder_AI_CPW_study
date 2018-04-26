import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("..")
from tensorflow_things.models.research.object_detection.utils import ops as utils_ops
from tensorflow_things.models.research.object_detection.utils import label_map_util
from tensorflow_things.models.research.object_detection.utils import visualization_utils as vis_util

def make_model():
    print('making model')
    # What model to download.
    MODEL_NAME = '../tensorflow_things/inference_graph_faster_small_lr_2_31313'
    # MODEL_FILE = MODEL_NAME + '.tar.gz'
    # DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'



    detection_graph = tf.Graph()

    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    return detection_graph
