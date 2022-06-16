import boto
from boto.s3.connection import S3Connection
from boto.sts import STSConnection
import boto3
import csv
import pandas as pd
import numpy as np
import io 
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests
from io import BytesIO
import csv


# Prompt for MFA time-based one-time password (TOTP)
mfa_TOTP = input("Enter the MFA code: ")

sts=boto3.client('sts')    
tempCredentials = sts.get_session_token(
    DurationSeconds=900,  # set session time in seconds
    SerialNumber="arn:aws:iam::302145289873:mfa/mai.xueqiao@carro.co", 
    TokenCode=mfa_TOTP
)


print("\nAccessKeyId: "+tempCredentials['Credentials']['AccessKeyId'])
print("\nSecretAccessKey: "+tempCredentials['Credentials']['SecretAccessKey'])
print("\nSessionToken: "+tempCredentials['Credentials']['SessionToken'])

# Use the temporary credentials to list the contents of an S3 bucket
mfa_session = boto3.session.Session(
    aws_access_key_id=tempCredentials['Credentials']['AccessKeyId'], 
    aws_secret_access_key=tempCredentials['Credentials']['SecretAccessKey'], 
    aws_session_token=tempCredentials['Credentials']['SessionToken'])

s3 = mfa_session.client('s3') 
bucket = 'carro-cv-project'
file_name = 'dashboard_errors/listing_inventories_images_202204271828.csv'

dtype={'url': 'string'}

obj = s3.get_object(Bucket=bucket, Key=file_name) #2
url_df = pd.read_csv(io.BytesIO(obj['Body'].read()), low_memory=False)["url"][:3]
#sg_df = initial_df.loc[initial_df['country']=='singapore']
#sg_urls = sg_df['url']
#for url in sg_df:
    # print(url)
    # res = requests.get(url)
    # img = Image.open(BytesIO(res.content))
#    img.show()
# urls = initial_df['url'][:3]

print(url_df.head())
