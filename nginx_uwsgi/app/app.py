"""
api.py
~~~~~~

This module defines a simple REST API for an imaginary Machine Learning
(ML) model. It will be used for testing Docker and Kubernetes.

This can be tested locally on the command line, using `python api.py`
to start the service and then in another terminal window using,

```
curl http://localhost:5000/score \
--request POST \
--header "Content-Type: application/json" \
--data '{"X": [1, 2]}'
```

To test the API.
"""
from flask import Flask, jsonify, make_response, request, Response
from typing import Iterable
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils

from PIL import Image
import numpy as np
import io
import flask
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


app = Flask(__name__)

model = None


def load_model():
    global model
    model = ResNet50(weights='imagenet')


def prepare_image(image, target):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image


@app.route('/')
def hello():
    return 'Hello Tu!'


@app.route('/predict', methods=['POST'])
def score():
    """Score data using an imaginary machine learning model.

    This API endpoint expects a JSON payload with a field called `X`
    containing an iterable sequence of features to send to the model.
    This data is parsed into Python dict and made available via
    `request.json`

    If `X` cannot be found in the parsed JSON data, then an exception
    will be raised. Otherwise, it will return a JSON payload with the
    `score` field containing the model's prediction.
    """
    load_model()
    try:
        data = {'success': False}

        if flask.request.method == 'POST':
            if flask.request.files.get('image'):
                image = flask.request.files['image'].read()
                image = Image.open(io.BytesIO(image))

                image = prepare_image(image, target=(224, 224))

                preds = model.predict(image)

                results = imagenet_utils.decode_predictions(preds)
                data['predictions'] = []

                for (imagenetID, label, prob) in results[0]:
                    r = {'label': label, 'probability': float(prob)}
                    data['predictions'].append(r)

                data['success'] = True

        return flask.jsonify(data)
    except KeyError:
        raise RuntimeError('"X" cannot be be found in JSON payload.')


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
# from flask import Flask
# app = Flask(__name__)


# @app.route('/')
# def hello():
#     return 'Hello, myapp!'
