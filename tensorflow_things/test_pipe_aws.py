'''
    this is just fully stolen from the object_detection_tutorial notebook.
    I made some functions where they just had a script, what with them having it
    in a Jupy and what not. I also don't need to print the images, so fuck that

    I did add the terrible terrible make_df_with_classes and make_dicts functions
    ok, i actually made a fair number of things in this bad boy. Namely, the
    ones that make it a genuine bad boy: make_matrix, make_precesion, and
    compare_thresholds

    wheeeeeeee

    This is for pulling data in from an s3 bucket.
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
import csv

from collections import defaultdict
from io import StringIO
# from matplotlib import pyplot as plt
from PIL import Image

#from pull_from_s3 import get_bucket
from models.research.object_detection.utils import ops as utils_ops
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util


def make_category_index(LABEL_MAP, NUM_CLASSES):
    ''' DOCSTRING
        makes the category_index used to pull the name of the detected
        object from the numerical detection class
        ----------
        INPUTS:
        LABEL_MAP: a .pbtxt file (generated with xml_to_csv.py if you're using
                  my code)
        NUM_CLASSES: Int, the number of classes your model will predict
        __________
        RETURNS:
        category_index
    '''
    label_map = label_map_util.load_labelmap(LABEL_MAP)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                        max_num_classes=NUM_CLASSES,
                                                        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return category_index


def idk_things_are_supposed_to_be_in_functions_i_guess_also_why_are_we_yelling(
                                        PATH_TO_CKPT,
                                        NUM_CLASSES,
                                        LABEL_MAP):
    '''DOCSTRING
        this creates the detection graph and category index (see above function)
        Pretty much just a bunch of TensorFlow objects and methods
        -----------
        INPUTS
        PATH_TO_CKPT: path to the tensorflow graph made from
                      export_inference_graph
        NUM_CLASSES: the number of classes your model predicts
        LABEL_MAP: the label map (a .pbtxt file). This is made in xml_to_csv if
                    you are using my code
        RETURNS
        -----------
        detection_graph: TensorFlow object
        category_index: an index of the class # and class label
    '''
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    category_index = make_category_index(LABEL_MAP, NUM_CLASSES)

    return detection_graph, category_index


def load_image_into_numpy_array(image):
    ''' DOCSTRING
        Loads the image into a numpy array
        -------------
        INPUT:
        image: the image to be converted
        _____________
        RETURNS:
        numpy array representation of the image
    '''
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image,sess):
    '''DOCTRING
    This predicts the bounding boxes and class type for an image. This is
    pretty much the same as the object_detection_tutorial.ipynb, but I moved
    the 2 opening with statements into above a for loop, s the graph isn't
    re-loaded everytime
    -------------
    INPUT:
    image: the image you want to predict after it has been loaded into a numpy
           array and had it's dimensions expanded
    graph: the detection_graph you loaded from a checkpoint, after it has gone
           through the long function name above
    -------------
    RETURNS:
    output_dict: 3 keys, detection_boxes, detection_scores, detection_classes
    '''
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
    '''DOCSTRING
       Displays the image using matplotlib with bounding boxes and the class in
       the image.

       #todo: come back and resize the detecion class displayed on the image
       --------------
       INPUTS:
       image_path: the path to the image you want to load
       IMAGE_SIZE: the desired output size
       detection_graph: the detection graph of the model you want to use to
                        predict on the image
       category_index: the category_index that will be used to go from class id
                       to class name
       --------------
       Returns:
       None, but it does print a pretty image
    '''
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


def make_dicts(test_images, PATH_TO_CKPT, NUM_CLASSES, LABEL_MAP,
               new_dict=False, s3=None):
    ''' DOCSTRING
        This makes the output_dictionary for images from an s3 bucket, one at a
        time. It pickles the dicts every 1000 images, and again at the end of the
        evaluation. It can be used on deeply nested images, but not on images
        dispersed in different folders.
        -----------
        INPUTS
        test_images: a list of boto object summaries, one for each image
        PATH_TO_CKPT: filepath to the checkpoint/frozen_inference_graph
        NUM_CLASSES: the number of classes you want to evaluate for
        LABEL_MAP: a .pbtxt file of the image ids and class name
        new_dict: if you want to load existing images or predict new ones
        s3: the s3 client you want to use to access the bucket.
        ----------
        Returns:
        dicts: a list of the dictionaries
        category_index: a pointer for category id to category name
    '''
    if new_dict:

        detection_graph, category_index = \
        idk_things_are_supposed_to_be_in_functions_i_guess_also_why_are_we_yelling(
                                        PATH_TO_CKPT,
                                        NUM_CLASSES,
                                        LABEL_MAP)
        dicts = []
        with detection_graph.as_default():
            with tf.Session() as sess:
                for i, image in enumerate(test_images[1:]):
                    try:
                        print('predicting image #', i)
                        image_key = image.key
                        image_bucket = image.bucket_name
                        s3.Bucket(image_bucket).download_file(image_key, 'test_image_from_aws.jpg')
                        print('    loaded image')
                        image = Image.open('    test_image_from_aws.jpg')
                        image_np = load_image_into_numpy_array(image)
                        image_np_expanded = np.expand_dims(image_np, axis=0)
                        print('    here')
                        dictionary = run_inference_for_single_image(image, sess)
                        print('    made dict')
                        dicts.append((image_key.split('/')[-1], dictionary))
                    except:
                        pass
                    if not i % 1000:
                        with open('detected_dicts.pkl', 'wb') as f:
                            pickle.dump(dicts, f)

        with open('detected_dicts.pkl', 'wb') as f:
            pickle.dump(dicts, f)

    else:
        category_index = make_category_index(LABEL_MAP, NUM_CLASSES)
        with open('detected_dicts.pkl', 'rb') as f:
            dicts = pickle.load(f)

    return dicts, category_index


def make_df_with_classes(dicts, category_index, threshold):
    ''' DOCSTRING:
        takes a list of dictionaries, and turns it into a df of predicted objects
        Used to test for different threshold values
        ------------
        INPUTS:
        dicts: a list of output_dictionaries
        category_index: a category_index object
        threshold: the threshold above which you will consider an obect detected
        ------------
        RETURNS:
        df: a dataframe with: image_path, and detections
    '''
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


def make_matrix_animal(path_to_df, df_pred):
    ''' DOCSTRING
        This compares two dataframes, a test and a predicted df. It calculates
        False Negative, False Postive, True negative, True Positive for each
        image. Honestly, your definition of the above might vary from mine.
        I also have to alter a lot of the columns.
        -------------
        INPUTS:
        path_to_df: a path to the csv containing true class values for each image
        df_pred: the predicted dataframe
    '''
    my_dict = {}
    with open(path_to_df, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            my_dict[row[1]] = row[0]

    test_df = pd.read_csv(path_to_df)
    test_df = test_df[test_df['CommonName'] != 'Setup'][test_df['ObsID'] == 2][['FileName', 'CommonName']]
    test_df['CommonName'] = test_df['CommonName'].str.lower()
    test_df['CommonName'].replace(['none', 'false trigger', 'setup',
                                   'timelapse', 'unknown', 'check', 'no flash'],
                                  ['','','','','','',''], inplace = True)
    test_df.dropna(inplace = True)
    test_df['CommonName'].replace(my_dict, inplace = True)
    test_df['image_path'] = 'test_images/' + test_df['FileName']
    print(test_df)
    comparison_df = test_df.join(df_pred.set_index('image_path'),
                                 on='image_path')
    comparison_df.dropna(inplace = True)
    print(df_pred)
    print(comparison_df)
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
                                             int(len(row.detections)==0) &
                                             (row['CommonName'] == 'none'),
                                             axis=1)
    return comparison_df


def make_precesion(comparison_df):
    ''' DOCSTRING
        Calculates the precesion from a comparison df
        --------------
        INPUTS:
        comparison_df: the df to calculate precesion from
        --------------
        RETURNS:
        precesion
    '''
    TP = sum(comparison_df['tp'])
    FP = sum(comparison_df['fp'])
    return TP/(TP + FP)


def compare_thresholds(dicts, category_index, path_to_df, range_object):
    ''' DOCSTRING
        Finds the best threshold from a given list of dictionaries
        ------------
        INPUTS
        dicts: a list of dictionaries (output_dicts)
        category_index: the category index that will be used to go from
                        category id to category name
        path_to_df: a path to the dataframe you will use to test against
        range_object: a range of thesholds to test from
        ------------
        RETURNS
        precision: the best precision found
        best_threshold: the threshold used to find this precision
        keeper_df: the best df it could create
    '''
    precision = -1
    best_threshold = 0
    keeper_df = None

    for threshold in range_object:
        print('THRESHOLD: ', threshold, '\n\n')
        df = make_df_with_classes(dicts, category_index, threshold)
        comparison_df = make_matrix_detection(path_to_df, df)
        print(comparison_df)
        this_precision = make_precesion(comparison_df)
        if this_precision > precision:
            best_threshold = threshold
            precision = this_precision
            keeper_df = comparison_df
            print('new best | precesion: ', this_precision, 'Threshold: ',best_threshold)
    keeper_df.to_csv('best-results.csv')
    return precision, best_threshold, keeper_df

def make_matrix_detection(path_to_df, df_pred):
    ''' DOCSTRING
        This compares two dataframes, a test and a predicted df. It calculates
        False Negative, False Postive, True negative, True Positive for each
        image. Honestly, your definition of the above might vary from mine.
        I also have to alter a lot of the columns.
        -------------
        INPUTS:
        path_to_df: a path to the csv containing true class values for each image
        df_pred: the predicted dataframe
    '''
    my_dict = {}
    with open(path_to_df, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            my_dict[row[1]] = row[0]

    test_df = pd.read_csv(path_to_df)

    test_df = test_df[test_df['CommonName'] != 'Setup'][test_df['ObsID'] == 2][['FileName', 'CommonName']]
    test_df['CommonName'] = test_df['CommonName'].str.lower()
    test_df['CommonName'].replace(['none', 'false trigger', 'setup',
                                   'timelapse', 'unknown', 'check', 'no flash'],
                                  [0,0,0,0,0,0,0], inplace = True)
    test_df.dropna(inplace = True)
    test_df['CommonName'] = np.where(test_df['CommonName']==0, 0, 1)
    test_df['image_path'] = test_df['FileName']
    comparison_df = test_df.join(df_pred.set_index('image_path'),
                                 on='image_path')
    comparison_df.dropna(inplace = True)
    comparison_df['tp'] = comparison_df.apply(lambda row:
                                              int((row.CommonName ==1) &
                                              (len(row.detections) > 0)),
                                              axis=1)
    comparison_df['fp'] = comparison_df.apply(lambda row:
                                              ((len(row.detections) -
                                              row.CommonName) + np.absolute(len(row.detections) -
                                              row.CommonName)) / 2,
                                              axis=1)
    comparison_df['fn'] = comparison_df.apply(lambda row:
                                              int((row.CommonName > 0) &
                                              (len(row.detections) == 0)),
                                              axis=1)
    comparison_df['tn'] = comparison_df.apply(lambda row:
                                             int((row.CommonName ==0) &
                                             (len(row.detections) == 0)),
                                             axis=1)
    return comparison_df

if __name__ == '__main__':
    #bucket, s3, objects = get_bucket('cpwphotos', aws_access_key, aws_secret_key, 'kb/')
    objects = None
    s3 = None
    dicts, category_index = make_dicts(test_images=objects,
                   PATH_TO_CKPT='inference_graph_faster_small_lr_15781/frozen_inference_graph.pb',
         NUM_CLASSES=3, LABEL_MAP='object-detection.pbtxt', new_dict=False, s3=s3)
    precision, best_threshold, keeper_df = compare_thresholds(dicts,
                                                    category_index,
                                                    'test_csv_file/kb_photos.csv',
                                                    np.arange(0,1,.1))
    recall = sum(keeper_df['tp']) / (sum(keeper_df['tp']) + sum(keeper_df['fn']))
    accuracy = (sum(keeper_df['tp']) + sum(keeper_df['tn'])) / len(keeper_df)

    # if you want to print your images out...
    # for image_path in TEST_IMAGE_PATHS:
    #     print_image(image_path, (12,8), detection_graph, category_index)
