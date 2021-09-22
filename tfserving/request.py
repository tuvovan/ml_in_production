import json
import requests
import argparse
import numpy as np

from PIL import Image
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', default='dog.jpeg', type=str)

args = parser.parse_args()

KERAS_REST_API_URL = 'http://localhost:8501/v1/models/my_classifier:predict'
IMAGE_PATH = args.input
image = Image.open(IMAGE_PATH)


def prepare_image(image, target=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image


image = prepare_image(image)


def make_prediction(instances):
    data = json.dumps({"signature_name": "serving_default",
                      "instances": instances.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(
        KERAS_REST_API_URL, data=data, headers=headers)

    predictions = json.loads(json_response.text)
    return predictions


preds = make_prediction(image)['predictions']
results = imagenet_utils.decode_predictions(np.array(preds))
data = {'success': False}
data['predictions'] = []

for (imagenetID, label, prob) in results[0]:
    r = {'label': label, 'probability': float(prob)}
    data['predictions'].append(r)

for (i, result) in enumerate(data['predictions']):
    print('{}. {}: {:.4f}'.format(i+1, result['label'], result['probability']))
