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
from aws import urls

if __name__ == '__main__':

################################# MODEL ####################################

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

    model.load_weights('efficientnetb0_best_weights.h5')


    ########################################## PREDICT ###########################################

    def pred_car(x):
        class_lookup = {1: 'car', 0: 'non-car'}
        file = open('log_file.txt', 'a+')
        result = model.predict([x])[0][0]
        result_verbose = model.predict([x])
        predicted_probability = [1-result,result]
        predicted_class = class_lookup[predicted_probability.index(1-max(predicted_probability))]
        file.writelines(f'{predicted_class}, Probabilities = car:{predicted_probability[0]}, non-car:{predicted_probability[1]}\n')
        
        return predicted_class, predicted_probability


    # data_path = 'listing_inventories_images_202204271828_thai.csv'
    # df = pd.read_csv(data_path)
    # urls = df['url'][170:180]

    height, width = 224, 224
    class_lookup = {1: 'car', 0: 'non-car'}

    car, non_car, error_indexes = list(), list(), list()

    for i, url in enumerate(urls):
        print('url {}'.format(url))
        try:
            response = requests.get(url)
            print('response {}'.format(response))
            img = Image.open(BytesIO(response.content))
            img =img.resize((height,width),resample=0)
            x = image.img_to_array(img)
            # Reshape
            x = x.reshape((1,) + x.shape)
            x /= 255.
            predicted_class, predicted_probability = pred_car(x)
            if predicted_class == 'car':
                car.append(url)
                print('Image is a car',url)
                print(f'Probabilities = car:{predicted_probability[0]}, non-car:{predicted_probability[1]}')
            if predicted_class == 'non-car':
                non_car.append(url)
                print('Not a car', url)
                print(f'Probabilities = car:{predicted_probability[0]}, non-car:{predicted_probability[1]}')
        except:
            error_indexes.append(i)
            print(i, 'index is an error')
            pass

    print('--------------------')
    print(car)

    dict = {'car': car}
    out_df = pd.DataFrame(dict) 
    out_df.to_csv('car.csv',index=False) 

## nohup python binary_classification.py > binary.log &