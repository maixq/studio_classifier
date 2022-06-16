import boto
from boto.s3.connection import S3Connection
from boto.sts import STSConnection
import boto3
import csv
import pandas as pd
import numpy as np
import io 

# Prompt for MFA time-based one-time password (TOTP)
mfa_TOTP = input("Enter the MFA code: ")

sts=boto3.client('sts')    
tempCredentials = sts.get_session_token(
    DurationSeconds=900,  # set session time in seconds
    SerialNumber="arn:aws:iam::302145289873:mfa/mai.xueqiao@carro.co", #"&region-arn;iam::302145289873:mfa/zhili.ng@carro.co",
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
url_df =  pd.read_csv(io.BytesIO(obj['Body'].read()), low_memory=False)["url"]
# url_df = pd.DataFrame(url_df)

###########################################################################

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import efficientnet.keras as efn
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, GlobalAveragePooling2D 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests
from io import BytesIO
import csv
import preprocess_data

################################# MODEL ####################################

car_checkpoint = 'car_weights.h5'
stu_checkpoint = 'studio_weights.h5'

def load_car_weight(car_weights):

    efnb0 = efn.EfficientNetB0(input_shape = (224, 224, 3), include_top=False, weights='imagenet')

    model = Sequential()
    model.add(efnb0)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))

    # Add a final sigmoid layer with 1 node for classification output
    model.add(Dense(1, activation="sigmoid"))

    #model compiling
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    model.load_weights(car_weights)

    return model

# binary car classifier
def car_classifier(input_img, model):

    class_lookup = {1: 'car', 0: 'non-car'}
    x = preprocess_data.load_url(input_img)
    result = model.predict([x])[0][0]
    predicted_probability = [1-result,result]
    predicted_class = class_lookup[predicted_probability.index(1-max(predicted_probability))]

    return x, predicted_class, predicted_probability

def load_stu_weight(stu_weights):

    efnb0 = efn.EfficientNetB0(input_shape = (224, 224, 3), include_top=False, weights='imagenet')
    model = Sequential()
    model.add(efnb0)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))

    # Add a final sigmoid layer with 1 node for classification output
    model.add(Dense(1, activation="sigmoid"))

    #model compiling
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    model.load_weights(stu_weights)

    return model

# binary studio classifier
def stu_classifier(x, model):

    class_lookup = {0: 'outdoor', 1: 'studio'}
    result = model.predict([x])[0][0]
    predicted_probability = [1-result,result]
    predicted_class = class_lookup[predicted_probability.index(max(predicted_probability))]

    return predicted_class, predicted_probability

#load model weights
def load_model_weights(car_weight_path, stu_weight_path):
    car_model = load_car_weight(car_weight_path)
    stu_model = load_stu_weight(stu_weight_path)
    return car_model, stu_model

def main(car_model, stu_model, input_img):

    x, pclass, _ = car_classifier(input_img, car_model)
    print('Predicted {} !!'.format(pclass))
    if pclass == 'car':
        stu_class, _ = stu_classifier(x, stu_model)
        print('Predicted {} !!'.format(stu_class))
        return stu_class

    return pclass

################################# PREDICT ####################################

if __name__ == '__main__':
    car_model, stu_model = load_model_weights(car_checkpoint, stu_checkpoint)  
    print('Loaded the weightttsss !!')
    
    import csv
    result_dict = {
        'filename':[],
        'prediction':[]
    }
    print(url_df)
    for i, input in url_df[29000:100000].iteritems():
        result_dict['filename'].append(input)
        try:
            print('Count',i)
            prediction = main(car_model, stu_model, input)
            # print('Image : {} ==> Prediction {}'.format(input, prediction))
            result_dict['prediction'].append(prediction)
    
        except:
            # url_df.at[i,'class']= 'error'
            result_dict['prediction'].append('error')
            pass

        if i%2==0:
            output_df = pd.DataFrame(result_dict)
            output_df.to_csv('test_batch0.csv',index= False)
