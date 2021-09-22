# restAPI demo

demo for deploying a deep learning model using restAPI and redis

## Overview

- RestAPI works on the web server end, to recieve the request from use and to show the results returned from model server.
- Redis works as a message queue, handle the multiple requests from user. Main functions would be:
  - store the request messages from web server
  - store the results that were returned from the model server

## How to run

Run

```bash
python scale_keras_server.py
```

Open another terminal, run

```
redis-server
```

_**note**_: make sure to install the redis and redis-server before running any of them.

then we are good to go. Lets make a request and see how the model see this cute dog.

![Image of golden](dog.jpeg#center)

```bash
curl -X POST -F image=@dog.jpeg http://localhost:80/predict
```

The result returned from the model server:

```json
{
  "predictions": [
    { "label": "golden_retriever", "probability": 0.943586528301239 },
    { "label": "Labrador_retriever", "probability": 0.0440649688243866 },
    { "label": "cocker_spaniel", "probability": 0.006853845901787281 },
    { "label": "clumber", "probability": 0.0016925691161304712 },
    { "label": "Sussex_spaniel", "probability": 0.0005000767414458096 }
  ],
  "success": true
}
```

or just:

```bash
python request.py -i dog.jpeg
```

then the result will be

```python
1. golden_retriever: 0.9436
2. Labrador_retriever: 0.0441
3. cocker_spaniel: 0.0069
4. clumber: 0.0017
5. Sussex_spaniel: 0.0005
```
