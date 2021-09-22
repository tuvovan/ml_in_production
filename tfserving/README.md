# tfserving_demo

demo for deploying a deep learning model using tfserving

The demo is to show how to use the tensorflow serving to deploy a deep learning model in production.

## step 1

Tensorflow serving uses tensorflow weight files, not keras so the first step would be transform the keras weight file to the tensorflow type.

In this tutorial, we will use the ResNet50, pretrained on ImageNet dataset.

Run

```bash
python create_tf_model.py
```

to generate a folder named `my_classifier/` which is the tensorflow typed weight file.

## step 2

Pull the tensorflow-serving from docker.

```bash
docker pull tensorflow/serving
```

in case you have GPU

```bash
docker pull tensorflow/serving:latest-gpu
```

## step 3

```bash
docker run -p 8501:8501 --name tfserving_classifier \
--mount type=bind,source=/Users/tf-server/my_classifier/,target=/models/my_classifier \
-e MODEL_NAME=my_classifier -t tensorflow/serving
```

## make prediction

Run `python request.py` and get the response from the tensorflow server.

The result should be like this as we run the prediction on the following cute picture
![Image of Yaktocat](dog.jpeg#center)

```python
1. golden_retriever: 0.9436
2. Labrador_retriever: 0.0441
3. cocker_spaniel: 0.0069
4. clumber: 0.0017
5. Sussex_spaniel: 0.0005
```
