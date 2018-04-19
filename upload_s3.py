import pandas as pd
import os
import boto3


def recurse_folders_upload(path, connection, bucket_name):
    ''' DOCSTRING
      Uploads all files from a given directory path to a given bucket
      -------------
      Input:
      directory: the directory path the files live in
      bucket_name: desired upload bucket
      connection: valid boto3-->s3 connection
      ------------
      Returns:
      Nothing, put it does ding when its done
    '''
    i = 0
    for folder in os.listdir(path):
        if os.path.isdir(path + '/' + folder):
            path1 = path + '/' + folder
            recurse_folders_upload(path1, connection, bucket_name)
        elif folder[-4:] == '.JPG':
            file_path = path + '/' + folder
            file_name = path[75:] + '/' + folder
            if i < 5:
                print(file_path)
                print('uploading ' + folder + '...')
            connection.upload_file(file_path,
                                   bucket_name,
                                   file_name)
            i += 1
        else:
            print(path)
            print(folder)
    print('Done! \a')


def upload_from_pandas(df, col_name, location,
                       bucket_name, connection, s3_folder,
                       verbose=False, test_size=5):
    ''' DOCSTRING
       loops through a Pandas df and pulls the name of every file out
       of a given column. Then uploads this file to the given bucket
       -----------------
       Input:
       df: the pd df to get file names from
       col_name: the name of the file containing the file names
       location: where the files live
       bucket_name: target bucket to upload files to
       connection: the live connection to an AWS s3 bucket
       verbose: T/F (default), prints uploading filename ... if true
       -----------------
       Returns:
       nothing, but it does beep when it's done
    '''
    i = 0
    for file_name in df[col_name]:
        if i == 0 and not verbose:
            print('Ensuring file loop...')
            i += 1
        if i < test_size and not verbose:
            print('uploading ' + file_name + '... (test)')
            i += 1
        if verbose:
            print('uploading', file_name, '...')
        filepath = location + file_name
        s3_location = s3_folder + file_name
        load_things(s3_location, bucket_name,
                    filepath, connection)
    print('All files uploaded\a')


def load_things(name, bucket_name, filepath, connection):
    ''' DOCSTRING
        this is really a one line function, and I realize I didn't need to make
        it, but I do think it made the code a little more readable. Mainly, it
        makes it easier to remember the arguments and where things go.
        -----------
        INPUTS:
        name : desired filepath / name in S3
        bucket_name : name of the bucket to uplaod the file to
        filepath : the path to the file
        connection : the connection to an s3 bucket (the client)
    '''

    connection.upload_file(filepath, bucket_name, name)


def make_connection(accesskey, secretkey):
    ''' DOCSTRING
        This makes a connection, again not many lines of code, but it does help
        make it easier to see where to put things. I mainly use this if I am
        uploading to s3 on a different computer than my own and have to
        manually enter my creds.
        ----------------
        INPUT
        accesskey : The access key from aws
        secretkey : The secret access key from aws
        ----------------
        RETURNS
        The connected client
    '''

    boto3_connection = boto3.resource('s3')
    s3_client = boto3.client('s3',
                             aws_access_key_id=accesskey,
                             aws_secret_access_key=secretkey)

    return s3_client


if __name__ == "__main__":
    boto3_connection = boto3.resource('s3')
    s3_client = boto3.client('s3')
    location = '/Volumes/Seagate Backup Plus Drive/Colorado Parks and ' +\
               'Wildlife/KevinBlechas/SPK_Cameras/'
    df = pd.read_csv(location + 'Photos with Species Identified ' +
                     'table as of summer 2017.csv')
    upload_from_pandas(df, 'FileName',
                       location + 'CatalogedPics/All/',
                       'cpwphotos', 's3_client',
                       verbose=True)
