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
    class_label = {}
    for i, folder in enumerate(os.listdir(path), 1):
        backslash = '/' if path[-1] != '/' else ''
        first_file = os.listdir(path + backslash + folder)[0]
        image_net_id = first_file.split('_')[0]
        class_label[image_net_id] = [folder, i]
    new_dict = {}
    for key, value in class_label.items():
        new_dict[value[0]] = [key, value[1]]
    class_label.update(new_dict)
    return class_label


def xml_to_df(path, class_label, directory):
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
    img_path = os.path.dirname(os.path.dirname(path))+'/images/'+directory+'/'
    this_file_path = root.find('filename').text.split('.')[0]
    this_file_path = this_file_path + '.JPEG'
    this_file_location = os.path.join(img_path, this_file_path)
    if not os.path.isfile(this_file_location):
        value = False
    else:
        value = (os.path.join(img_path, this_file_path), #filename
                 int(root.find('size')[0].text), #width
                 int(root.find('size')[1].text), #height
                 class_label[member[0].text][1], # class
                 int(member.find('bndbox')[0].text), # xmin
                 int(member.find('bndbox')[1].text), # ymin
                 int(member.find('bndbox')[2].text), # xmax
                 int(member.find('bndbox')[3].text) # ymax
                 )
    return value


def df_split(xml_df, test_size):
    observations = len(xml_df)
    test_indices = np.random.choice(observations,
                                    round(observations * test_size),
                                    replace = False)
    train_indicies = np.isin(np.arange(observations), test_indices, invert = True)
    test_df = xml_df.iloc[test_indices]
    train_df = xml_df.iloc[train_indicies]
    return test_df, train_df


def concate_save_dfs(path, test_df, train_df):
    if 'train.csv' in os.listdir('GalvanizeMax_ImageSet') \
    and 'test.csv' in os.listdir('GalvanizeMax_ImageSet'):
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
    strings = ''
    for value in class_label.values():
        string = \
        '''item {{
        \r        id: {}
        \r        name: '{}'
        \r      }}\n\n'''.format(value[1], value[0])
        strings += string
    with open(path_to_obj_dec, 'w') as write_file:
        write_file.write(strings)

def main(path, path_to_obj_dec):
    class_label = make_class_label(path)
    make_object_detection_map(class_label, path_to_obj_dec)
    for directory in os.listdir(path):
        folder_path = os.path.join(path, directory)
        xml_df = xml_to_df(folder_path, class_label, directory)
        test_df, train_df = df_split(xml_df, 0.2)
        new_test_df, new_train_df = concate_save_dfs(os.path.dirname(path), test_df, train_df)
    return new_test_df, new_train_df


if __name__ == "__main__":
    new_test_df, new_train_df = main('GalvanizeMax_ImageSet/annotationsXML', 'object-detection.pbtxt')


















    #
