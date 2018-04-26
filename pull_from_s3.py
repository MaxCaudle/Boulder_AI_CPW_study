import botocore
import boto3


def get_bucket(bucket_name, aws_access_key, aws_secret_key, filter= None):
    s3 = boto3.resource('s3',
                             aws_access_key_id=aws_access_key,
                             aws_secret_access_key=aws_secret_key)
    bucket = s3.Bucket(bucket_name)
    exists = True
    try:
        s3.meta.client.head_bucket(Bucket=bucket_name)
    except botocore.exceptions.ClientError as e:
        # If a client error is thrown, then check that it was a 404 error.
        # If it was a 404 error, then the bucket does not exist.
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            exists = False
    objects = [i for i in bucket.objects.filter(Prefix=filter)]
    return bucket, s3, objects

if __name__ == '__main__':
    bucket, client, objects = get_bucket('cpwphotos', aws_access_key, aws_secret_key, 'kb/')
