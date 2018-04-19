'''
I mainly used his xml parse function for an outline of how to make the xml_to_df
and extract_from_member functions
See datitran's excellent git repo on training a racoon detector
https://github.com/datitran/raccoon_dataset
'''


import os
import glob
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET


def make_class_label(path):
    ''' DOCSTRING
        This will make a dictionary for the different class labels. It's ok. You
        just give it a path to the a folder containing different folders and it
        loops through them and makes a dictionary of them and a class number.
        -------------
        INPUTS
        path: the sys_path to the folder that contains the xml files. Mine was
              organized so that different folders contained different classes.
              The different files also started with a class identifier
        --------------
        RETURNS
        class_label: a dictionary of the class label and class id number
    '''
    class_label = {}
    i=1
    for folder in os.listdir(path):
        if folder == '.DS_Store':
            continue
        backslash = '/' if path[-1] != '/' else ''
        first_file = os.listdir(path + backslash + folder)[0]
        image_net_id = first_file.split('_')[0]
        class_label[folder] = i
        i+=1
    new_dict = {}
    class_label.update(new_dict)
    return class_label


def xml_to_df(path, class_label, directory):
    ''' DOCSTRING
        This converts all xml files in a path to dataframe that will eventually
        become saved to a csv. It also makes sure that the image actually exists
        ------------
        INPUTS
        path: path to xml files
        class_label: the class_label dictionary
        directory: the actual folder we are looking in
        ------------
        RETURNS
        xml_df: a dataframe with the info from all the xml files in the directory
    '''
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = extract_from_member(member, path, directory, root, class_label)
            if value:
                xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def extract_from_member(member, path, directory, root, class_label):
    ''' DOCSTRING
        This gets the info from a single file in the directory. This is where we
        check to make sure the image actually exists
        -----------
        INPUTS
        member: the xml file to get info from
        path: the path to the directory
        directory: the directory we are looking at
        root: an xml.etree.ElementTree object
        class_label: the dictionary with the class labels
        ----------
        RETURNS
        value: the row to append to the df for the folder
    '''
    img_path = os.path.dirname(os.path.dirname(path))+'/images/'+directory+'/'
    this_file_path = root.find('filename').text.split('.')[0]
    this_file_path = this_file_path + '.JPEG'
    this_file_location = os.path.join(img_path, this_file_path)
    if not os.path.isfile(this_file_location):
        value = False
    else:
        label = member.find('name').text
        if label[0] != 'n':
            print(label)
            label = 'n02084071'
        value = (os.path.join(img_path, this_file_path), #filename
                 int(root.find('size')[0].text), #width
                 int(root.find('size')[1].text), #height
                 class_label[label], # class
                 int(member.find('bndbox')[0].text), # xmin
                 int(member.find('bndbox')[1].text), # ymin
                 int(member.find('bndbox')[2].text), # xmax
                 int(member.find('bndbox')[3].text) # ymax
                 )
    return value


def df_split(xml_df, test_size=0.2):
    ''' DOCSTRING
        This function splits the dataframe into a train and test group, it also
        shuffles the data points. We call this for every directory to make sure
        the classes are adequately represented in the test and train dfs
        ------------
        INPUTS:
        xml_df: the df to split
        test_size: ratio of test/all points, defaults to 20%  of the set
        -----------
        RETURNS
        test_df: the test df of xml info
        train_df: the train df of xml info
    '''
    observations = len(xml_df)
    test_indices = np.random.choice(observations,
                                    round(observations * test_size),
                                    replace = False)
    train_indicies = np.isin(np.arange(observations), test_indices, invert = True)
    test_df = xml_df.iloc[test_indices].sample(frac=1)
    train_df = xml_df.iloc[train_indicies].sample(frac=1)
    return test_df, train_df


def concate_save_dfs(path, test_df, train_df):
    ''' DOCSTRING
        This concates all the dfs and saves them. I do this iteratively to make
        sure the different classes are represented in both test and train dfs
        ----------
        INPUTS
        path: the path to the test and train csv files. the file names are
              hardcoded because youre not allowed to name your csvs something
              dumb
        test_df: the test dataframe
        train_df: the training dataframe
        ---------
        RETURNS
        new_test_df: the newly concatenated test df
        new_train_df: the newly concatenated train df
    '''
    if 'train.csv' in os.listdir('image_annotations') \
    and 'test.csv' in os.listdir('image_annotations'):
        existing_test = pd.read_csv(os.path.join(path,'test.csv'))
        existing_train = pd.read_csv(os.path.join(path,'train.csv'))
        new_test_df = pd.concat([existing_test, test_df])
        new_train_df = pd.concat([existing_train, train_df])
    else:
        new_test_df = test_df
        new_train_df = train_df
    new_test_df.to_csv(path+'/test.csv', index = None)
    #shuffle the code with df.sample(frac=1) frac returns the full df
    new_train_df.sample(frac=1).to_csv(path+'/train.csv', index = None)

    return new_test_df, new_train_df

def make_object_detection_map(class_label, path_to_obj_dec):
    ''' DOCSTRING
        This makes a detection map that is used later on when you actually
        detect things.
        ---------
        INPUTS:
        class_label: the class label dictionary made earlier in this script
        path_to_obj_dec: where you want to save this
        ---------
        RETURNS:
        NONE
    '''
    strings = ''
    for key, value in class_label.items():
        string = \
        '''item {{
        \r        id: {}
        \r        name: '{}'
        \r      }}\n\n'''.format(value, key)
        strings += string
    with open(path_to_obj_dec, 'w') as write_file:
        write_file.write(strings)


def main(path, path_to_obj_dec):
    ''' DOCSTRING
        This just runs the whoooooooole script.
        ----------
        INPUTS:
        path: the path to the folder containing folders of xml files
        path_to_obj_dec: where you want to save the object-detection file
        ----------
        RETURNS:
        new_test_df: a test_df
        new_train_df: a train_df
        class_label: a class_label dictionary
    '''
    class_label = make_class_label(path)
    make_object_detection_map(class_label, path_to_obj_dec)
    for directory in os.listdir(path):
        folder_path = os.path.join(path, directory)
        xml_df = xml_to_df(folder_path, class_label, directory)
        if len(xml_df) > 0:
            test_df, train_df = df_split(xml_df, 0.2)
            new_test_df, new_train_df = concate_save_dfs(os.path.dirname(path), test_df, train_df)
    return new_test_df, new_train_df, class_label


if __name__ == "__main__":
    new_test_df, new_train_df, class_label = main('image_annotations/annotations', 'object-detection.pbtxt')


















    #
