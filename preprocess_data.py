import pandas as pd
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests
from io import BytesIO
import csv

# data_path = 'listing_inventories_images_202204271828_thai.csv'

# df = pd.read_csv(data_path)
# url_list = df['url']

height, width = 224, 224
class_lookup = {1: 'car', 0: 'non-car'}

def load_url(url):
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

        return x
        # predicted_class, predicted_probability = pred_car(x)

        # if predicted_class == 'car':
        #     car.append(url)
        #     print('Image is a car',url)
        #     print(f'Probabilities = car:{predicted_probability[0]}, non-car:{predicted_probability[1]}')
        #     urllib.request.urlretrieve(url, 'data/car2/' + file_name)

        # if predicted_class == 'non-car':
        #     non_car.append(url)
        #     print('Not a car', url)
        #     print(f'Probabilities = car:{predicted_probability[0]}, non-car:{predicted_probability[1]}')
        #     urllib.request.urlretrieve(url, 'data/noncar2/' + file_name)

    except:
        pass


# for i, url in enumerate(url_list[:5]):
#     file_name = str(id[i])+ '-' + url_list[i].split('/')[-1]
#     print('Count: ' ,i)
#     print('url {}'.format(url))
#     try:
#         response = requests.get(url)
#         print('response {}'.format(response))
#         img = Image.open(BytesIO(response.content))
#         img =img.resize((height,width),resample=0)
#         x = image.img_to_array(img)
#         # Reshape
#         x = x.reshape((1,) + x.shape)
#         x /= 255.
#         predicted_class, predicted_probability = pred_car(x)

#         if predicted_class == 'car':
#             car.append(url)
#             print('Image is a car',url)
#             print(f'Probabilities = car:{predicted_probability[0]}, non-car:{predicted_probability[1]}')
#             urllib.request.urlretrieve(url, 'data/car2/' + file_name)

#         if predicted_class == 'non-car':
#             non_car.append(url)
#             print('Not a car', url)
#             print(f'Probabilities = car:{predicted_probability[0]}, non-car:{predicted_probability[1]}')
#             urllib.request.urlretrieve(url, 'data/noncar2/' + file_name)

#     except:
#         pass