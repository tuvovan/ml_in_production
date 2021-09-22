import requests
import argparse

parser = argparse.ArgumentParser()
   
parser.add_argument('-i', '--input', default=None, type=str)

args = parser.parse_args()

KERAS_REST_API_URL = 'http://0.0.0.0:5000/predict'
IMAGE_PATH = args.input

image = open(IMAGE_PATH, 'rb').read()
payload = {'image':image}

r =requests.post(KERAS_REST_API_URL, files=payload).json()


if r['success'] :
    for (i, result) in enumerate(r['predictions']):
        print('{}. {}: {:.4f}'.format(i+1, result['label'], result['probability']))

else:
    print('Request failed!')
