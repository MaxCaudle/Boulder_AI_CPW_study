'''
    this is just fully stolen from the object_detection_tutorial notebook.
    I made some functions where they just had a script, what with them having it
    in a Jupy and what not. I also don't need to print the images, so fuck that

    I did add the terrible terrible make_df_with_classes and make_dicts functions
    ok, i actually made a fair number of things in this bad boy. Namely, the
    ones that make it a genuine bad boy: make_matrix, make_precesion, and
    compare_thresholds

    wheeeeeeee
'''


import numpy as np
import pandas as pd
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pickle

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from models.research.object_detection.utils import ops as utils_ops
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util


def make_category_index(LABEL_MAP, NUM_CLASSES):
    label_map = label_map_util.load_labelmap(LABEL_MAP)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                        max_num_classes=NUM_CLASSES,
                                                        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return category_index


def idk_things_are_supposed_to_be_in_functions_i_guess_also_why_are_we_yelling(
                                        test_images,
                                        PATH_TO_CKPT,
                                        NUM_CLASSES,
                                        LABEL_MAP):

    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    category_index = make_category_index(LABEL_MAP, NUM_CLASSES)

    PATH_TO_TEST_IMAGES_DIR = test_images
    TEST_IMAGE_PATHS = []
    for file in os.listdir(PATH_TO_TEST_IMAGES_DIR):
        TEST_IMAGE_PATHS.append(os.path.join(PATH_TO_TEST_IMAGES_DIR, file))
    return TEST_IMAGE_PATHS, detection_graph, category_index


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
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
  return output_dict


def print_image(image_path, IMAGE_SIZE, detection_graph, category_index):
  image = Image.open(image_path)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
  # IMAGE_SIZE == tuple | Size, in inches, of the output images.
  plt.figure(figsize=IMAGE_SIZE)
  plt.imshow(image_np)


def make_dicts(test_images,PATH_TO_CKPT, NUM_CLASSES, LABEL_MAP, new_dict = False):
    if new_dict:

        TEST_IMAGE_PATHS, detection_graph, category_index = \
        idk_things_are_supposed_to_be_in_functions_i_guess_also_why_are_we_yelling(
                                        test_images,
                                        PATH_TO_CKPT,
                                        NUM_CLASSES,
                                        LABEL_MAP)
        dicts = []
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            dicts.append((image_path, run_inference_for_single_image(image, detection_graph)))
        with open('detected_dicts.pkl', 'wb') as f:
            pickle.dump(dicts, f)

    else:
        category_index = make_category_index(LABEL_MAP, NUM_CLASSES)
        with open('detected_dicts.pkl', 'rb') as f:
            dicts = pickle.load(f)

    return dicts, category_index


def make_df_with_classes(dicts, category_index, threshold):
    df = pd.DataFrame(columns=['image_path', 'detections'])

    for image_dict in dicts:
        image_path = image_dict[0]
        scores = image_dict[1]['detection_scores']
        classes = image_dict[1]['detection_classes']

        keeper_indices = np.argwhere(scores.astype(float) > threshold)
        keeper_classes_int = classes[keeper_indices].ravel().tolist()
        keeper_classes = [category_index[x]['name'] for x in keeper_classes_int]
        df = df.append({'image_path': image_path, 'detections': keeper_classes}, ignore_index=True)

    return df


def make_matrix(path_to_df, df_pred):
    test_df = pd.read_csv(path_to_df)
    test_df = test_df[test_df['CommonName'] != 'Setup'][test_df['ObsID'] == 2][['FileName', 'CommonName']]
    test_df['CommonName'] = test_df['CommonName'].str.lower()
    test_df['CommonName'].replace(['none', 'false trigger'], ['none','none'], inplace = True)
    test_df['image_path'] = 'image_annotations/images/' + test_df['FileName']

    comparison_df = test_df.join(df_pred.set_index('image_path'),
                                 on='image_path')
    comparison_df.dropna(inplace = True)
    comparison_df['tp'] = comparison_df.apply(lambda row:
                                              int(row.CommonName in
                                                  row.detections),
                                              axis=1)
    comparison_df['fp'] = comparison_df.apply(lambda row:
                                              int(len(row.detections) -
                                                      row.tp),
                                              axis=1)
    comparison_df['fn'] = comparison_df.apply(lambda row:
                                              int(
                                              (row.CommonName not in
                                               row.detections)
                                               &
                                               (row.CommonName != 'none')),
                                              axis=1)
    comparison_df['tn'] = comparison_df.apply(lambda row:
                                             int((row.CommonName == 'none')
                                              & len(row.detections)==0), axis=1)
    return comparison_df


def make_precesion(comparison_df):
    TP = sum(comparison_df['tp'])
    FP = sum(comparison_df['fp'])
    return TP / (TP + FP)


def compare_thresholds(dicts, category_index, path_to_df, range_object):
    precision = -1
    best_threshold = 0
    keeper_df = None

    for threshold in range_object:
        df = make_df_with_classes(dicts, category_index, threshold)
        comparison_df = make_matrix(path_to_df, df)
        this_precision = make_precesion(comparison_df)
        if this_precision > precision:
            best_threshold = threshold
            precision = this_precision
            keeper_df = comparison_df

    return precision, best_threshold, keeper_df

if __name__ == '__main__':
    dicts, category_index = make_dicts(test_images='image_annotations/images/',
                   PATH_TO_CKPT='inference_graph/frozen_inference_graph.pb',
         NUM_CLASSES=5, LABEL_MAP='object-detection.pbtxt', new_dict=False)

    precision, best_threshold, keeper_df = compare_thresholds(dicts,
                                                    category_index,
                                                    'test_csv/kb_photos.csv',
                                                    range(0,1,10))

    # if you want to print your images out...
    # for image_path in TEST_IMAGE_PATHS:
    #     print_image(image_path, (12,8), detection_graph, category_index)
