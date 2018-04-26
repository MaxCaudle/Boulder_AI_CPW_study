import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from make_model import make_model


sys.path.append("..")
from tensorflow_things.models.research.object_detection.utils import ops as utils_ops
from tensorflow_things.models.research.object_detection.utils import label_map_util
from tensorflow_things.models.research.object_detection.utils import visualization_utils as vis_util


def cat_index(label_map_path = '../tensorflow_things/object-detection.pbtxt',
              NUM_CLASSES = 33):
    # List of the strings that is used to add correct label for each box.
    # PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    toc = time.time()
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                    max_num_classes=NUM_CLASSES,
                                                    use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    tic = time.time()
    return category_index, tic - toc

def load_image_into_numpy_array(image):
  toc = time.time()
  (im_width, im_height) = image.size
  tic = time.time()

  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8), tic - toc

def run_inference_for_single_image(image, graph):
  toc = time.time()
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  tic = time.time()

  return output_dict, tic - toc

def make_image(image_path, detection_graph):
  toc = time.time()
  category_index, cat_time = cat_index()
  image = Image.open(image_path)
  width, height = image.size
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np, np_time = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict, inf_time = run_inference_for_single_image(image_np, detection_graph)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=(width//50))
  IMAGE_SIZE = (8,8)
  save_path = image_path.split('/')[-1]
  save_path = 'static/detected_img/' + save_path.split('.')[0]+'.jpg'
  plt.figure(figsize=IMAGE_SIZE)
  plt.imshow(image_np)
  plt.savefig(save_path)
  tic = time.time()
  good_scores = np.argwhere(output_dict['detection_scores'] >0.5).ravel()
  keepers_list = [(output_dict['detection_classes'][good_score],
                   output_dict['detection_scores'][good_score])
                   for good_score in good_scores]
  return save_path, keepers_list

if __name__ == "__main__":
    graph = make_model()
    make_image('static/img/Beagle-1.jpg', graph)
