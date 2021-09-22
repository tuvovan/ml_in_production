# Deploying Machine Learning Models on Kubernetes

A common pattern for deploying Machine Learning (ML) models into production environments - e.g. ML models trained using the SciKit Learn or Keras packages (for Python), that are ready to provide predictions on new data - is to expose these ML as RESTful API microservices, hosted from within [Docker](https://www.docker.com) containers. These can then deployed to a cloud environment for handling everything required for maintaining continuous availability - e.g. fault-tolerance, auto-scaling, load balancing and rolling service updates.

The configuration details for a continuously available cloud deployment are specific to the targeted cloud provider(s) - e.g. the deployment process and topology for Amazon Web Services is not the same as that for Microsoft Azure, which in-turn is not the same as that for Google Cloud Platform. This constitutes knowledge that needs to be acquired for every cloud provider. Furthermore, it is difficult (some would say near impossible) to test entire deployment strategies locally, which makes issues such as networking hard to debug.

[Kubernetes](https://kubernetes.io) is a container orchestration platform that seeks to address these issues. Briefly, it provides a mechanism for defining **entire** microservice-based application deployment topologies and their service-level requirements for maintaining continuous availability. It is agnostic to the targeted cloud provider, can be run on-premises and even locally on your laptop - all that's required is a cluster of virtual machines running Kubernetes - i.e. a Kubernetes cluster.

This README is designed to be read in conjunction with the code in this repository, that contains the Python modules, Docker configuration files and Kubernetes instructions for demonstrating how a simple Python ML model can be turned into a production-grade RESTful model-scoring (or prediction) API service, using Docker and Kubernetes - both locally and with Google Cloud Platform (GCP). It is not a comprehensive guide to Kubernetes, Docker or ML - think of it more as a 'ML on Kubernetes 101' for demonstrating capability and allowing newcomers to Kubernetes (e.g. data scientists who are more focused on building models as opposed to deploying them), to get up-and-running quickly and become familiar with the basic concepts and patterns.

We will demonstrate ML model deployment using two different approaches: a first principles approach using Docker and Kubernetes; and then a deployment using the [Seldon-Core](https://www.seldon.io) Kubernetes native framework for streamlining the deployment of ML services. The former will help to appreciate the latter, which constitutes a powerful framework for deploying and performance-monitoring many complex ML model pipelines.

This work was initially committed in 2018 and has since formed the basis of [Bodywork](https://github.com/bodywork-ml/bodywork-core) - an open-source MLOps tool for deploying machine learning projects developed in Python, to Kubernetes. Bodywork automates a lot of the steps that this project has demonstrated to the many machine learning engineers that have used it over the years - take a look at the [documentation](https://bodywork.readthedocs.io/en/latest/).

## Containerising a Simple ML Model Scoring Service using Flask and Docker

We start by demonstrating how to achieve this basic competence using the simple Python ML model scoring REST API contained in the `api.py` module, together with the `Dockerfile`, both within the `py-flask-ml-score-api` directory, whose core contents are as follows,

```bash
py-flask-ml-score-api/
 | Dockerfile
 | Pipfile
 | Pipfile.lock
 | api.py
```

If you're already feeling lost then these files are discussed in the points below, otherwise feel free to skip to the next section.

### Defining the Flask Service in the `api.py` Module

This is a Python module that uses the [Flask](http://flask.pocoo.org) framework for defining a web service (`app`), with a function (`score`), that executes in response to a HTTP request to a specific URL (or 'route'), thanks to being wrapped by the `app.route` function. For reference, the relevant code is reproduced below,

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils

from PIL import Image
import numpy as np
import io
import flask
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


from typing import Iterable

from flask import Flask, jsonify, make_response, request, Response

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
```

If running locally - e.g. by starting the web service using `python run api.py` - we would be able reach our function (or 'endpoint') at `http://localhost:5000/predict`. This function takes data sent to it as JSON (that has been automatically de-serialised as a Python dict made available as the `request` variable in our function definition), and returns a response (automatically serialised as JSON).

In our example function, we expect an array of features, `X`, that we pass to a ML model, which in our example returns those same features back to the caller - i.e. our chosen ML model is the identity function, which we have chosen for purely demonstrative purposes. We could just as easily have loaded a pickled SciKit-Learn or Keras model and passed the data to the approproate `predict` method, returning a score for the feature-data as JSON - see [here](https://github.com/AlexIoannides/ml-workflow-automation/blob/master/deploy/py-sklearn-flask-ml-service/api.py) for an example of this in action.

### Defining the Docker Image with the `Dockerfile`

 A `Dockerfile` is essentially the configuration file used by Docker, that allows you to define the contents and configure the operation of a Docker container, when operational. This static data, when not executed as a container, is referred to as the 'image'. For reference, the `Dockerfile` is reproduced below,

```docker
    FROM python:3.6-slim
    WORKDIR /usr/src/app
    COPY . .
    RUN pip install pipenv
    RUN pipenv install
    EXPOSE 5000
    CMD ["pipenv", "run", "python", "api.py"]
```

In our example `Dockerfile` we:

 - start by using a pre-configured Docker image (`python:3.6-slim`) that has a version of the [Alpine Linux](https://www.alpinelinux.org/community/) distribution with Python already installed;
 - then copy the contents of the `py-flask-ml-score-api` local directory to a directory on the image called `/usr/src/app`;
 - then use `pip` to install the [Pipenv](https://pipenv.readthedocs.io/en/latest/) package for Python dependency management (see the appendix at the bottom for more information on how we use Pipenv);
 - then use Pipenv to install the dependencies described in `Pipfile.lock` into a virtual environment on the image;
 - configure port 5000 to be exposed to the 'outside world' on the running container; and finally,
 - to start our Flask RESTful web service - `api.py`. Note, that here we are relying on Flask's internal [WSGI](https://en.wikipedia.org/wiki/Web_Server_Gateway_Interface) server, whereas in a production setting we would recommend on configuring a more robust option (e.g. Gunicorn), [as discussed here](https://pythonspeed.com/articles/gunicorn-in-docker/).

 Building this custom image and asking the Docker daemon to run it (remember that a running image is a 'container'), will expose our RESTful ML model scoring service on port 5000 as if it were running on a dedicated virtual machine. Refer to the official [Docker documentation](https://docs.docker.com/get-started/) for a more comprehensive discussion of these core concepts.

### Building a Docker Image for the ML Scoring Service

We assume that [Docker is running locally](https://www.docker.com) (both Docker client and daemon), that the client is logged into an account on [DockerHub](https://hub.docker.com) and that there is a terminal open in the this project's root directory. To build the image described in the `Dockerfile` run,

```bash
docker build --tag alexioannides/test-ml-score-api py-flask-ml-score-api
```

Where 'alexioannides' refers to the name of the DockerHub account that we will push the image to, once we have tested it. 

#### Testing

To test that the image can be used to create a Docker container that functions as we expect it to use,

```bash
docker run --rm --name test-api -p 5000:5000 -d alexioannides/test-ml-score-api
```

Where we have mapped port 5000 from the Docker container - i.e. the port our ML model scoring service is listening to - to port 5000 on our host machine (localhost). Then check that the container is listed as running using,

```bash
docker ps
```

And then test the exposed API endpoint using,

Lets see how our model sees our dog:
![Image of Yaktocat](dog.jpeg#center)

```bash
curl -X POST -F image=@dog.jpeg 'http://localhost:5000/predict'
```

Where you should expect a response along the lines of,

```json
{
  "predictions": [
    {
      "label": "golden_retriever", 
      "probability": 0.9435867667198181
    }, 
    {
      "label": "Labrador_retriever", 
      "probability": 0.04406480863690376
    }, 
    {
      "label": "cocker_spaniel", 
      "probability": 0.006853834725916386
    }, 
    {
      "label": "clumber", 
      "probability": 0.0016925663221627474
    }, 
    {
      "label": "Sussex_spaniel", 
      "probability": 0.0005000759847462177
    }
  ], 
  "success": true
}
```

All our test model does is return the input data - i.e. it is the identity function. Only a few lines of additional code are required to modify this service to load a SciKit Learn model from disk and pass new data to it's 'predict' method for generating predictions - see [here](https://github.com/AlexIoannides/ml-workflow-automation/blob/master/deploy/py-sklearn-flask-ml-service/api.py) for an example. Now that the container has been confirmed as operational, we can stop it,

```bash
docker stop test-api
```

#### Pushing the Image to the DockerHub Registry

In order for a remote Docker host or Kubernetes cluster to have access to the image we've created, we need to publish it to an image registry. All cloud computing providers that offer managed Docker-based services will provide private image registries, but we will use the public image registry at DockerHub, for convenience. To push our new image to DockerHub (where my account ID is 'alexioannides') use,

```bash
docker push alexioannides/test-ml-score-api
```

Where we can now see that our chosen naming convention for the image is intrinsically linked to our target image registry (you will need to insert your own account ID where required). Once the upload is finished, log onto DockerHub to confirm that the upload has been successful via the [DockerHub UI](https://hub.docker.com/u/alexioannides).

## Installing Kubernetes for Local Development and Testing

There are two options for installing a single-node Kubernetes cluster that is suitable for local development and testing: via the [Docker Desktop](https://www.docker.com/products/docker-desktop) client, or via [Minikube](https://github.com/kubernetes/minikube).

### Installing Kubernetes via Docker Desktop

If you have been using Docker on a Mac, then the chances are that you will have been doing this via the Docker Desktop application. If not (e.g. if you installed Docker Engine via Homebrew), then Docker Desktop can be downloaded [here](https://www.docker.com/products/docker-desktop). Docker Desktop now comes bundled with Kubernetes, which can be activated by going to `Preferences -> Kubernetes` and selecting `Enable Kubernetes`. It will take a while for Docker Desktop to download the Docker images required to run Kubernetes, so be patient. After it has finished, go to `Preferences -> Advanced` and ensure that at least 2 CPUs and 4 GiB have been allocated to the Docker Engine, which are the the minimum resources required to deploy a single Seldon ML component.

To interact with the Kubernetes cluster you will need the `kubectl` Command Line Interface (CLI) tool, which will need to be downloaded separately. The easiest way to do this on a Mac is via Homebrew - i.e with `brew install kubernetes-cli`. Once you have `kubectl` installed and a Kubernetes cluster up-and-running, test that everything is working as expected by running,

```bash
kubectl cluster-info
```

Which ought to return something along the lines of,

```bash
Kubernetes master is running at https://kubernetes.docker.internal:6443
KubeDNS is running at https://kubernetes.docker.internal:6443/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy

To further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.
```

### Installing Kubernetes via Minikube

On Mac OS X, the steps required to get up-and-running with Minikube are as follows:

- make sure the [Homebrew](https://brew.sh) package manager for OS X is installed; then,
- install VirtualBox using, `brew cask install virtualbox` (you may need to approve installation via OS X System Preferences); and then,
- install Minikube using, `brew cask install minikube`.

To start the test cluster run,

```bash
minikube start --memory 4096
```

Where we have specified the minimum amount of memory required to deploy a single Seldon ML component. Be patient - Minikube may take a while to start. To test that the cluster is operational run,

```bash
kubectl cluster-info
```

Where `kubectl` is the standard Command Line Interface (CLI) client for interacting with the Kubernetes API (which was installed as part of Minikube, but is also available separately).

### Deploying the Containerised ML Model Scoring Service to Kubernetes

To launch our test model scoring service on Kubernetes, we will start by deploying the containerised service within a Kubernetes [Pod](https://kubernetes.io/docs/concepts/workloads/pods/pod-overview/), whose rollout is managed by a [Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/), which in in-turn creates a [ReplicaSet](https://kubernetes.io/docs/concepts/workloads/controllers/replicaset/) - a Kubernetes resource that ensures a minimum number of pods (or replicas), running our service are operational at any given time. This is achieved with,

```bash
kubectl create deployment test-ml-score-api --image=alexioannides/test-ml-score-api:latest
```

To check on the status of the deployment run,

```bash
kubectl rollout status deployment test-ml-score-api
```

And to see the pods that is has created run,

```bash
kubectl get pods
```

It is possible to use [port forwarding](https://en.wikipedia.org/wiki/Port_forwarding) to test an individual container without exposing it to the public internet. To use this, open a separate terminal and run (for example),

```bash
kubectl port-forward test-ml-score-api-szd4j 5000:5000
```

Where `test-ml-score-api-szd4j` is the precise name of the pod currently active on the cluster, as determined from the `kubectl get pods` command. Then from your original terminal, to repeat our test request against the same container running on Kubernetes run,

```bash
curl -X POST -F image=@dog.jpeg 'http://localhost:5000/predict'
```

To expose the container as a (load balanced) [service](https://kubernetes.io/docs/concepts/services-networking/service/) to the outside world, we have to create a Kubernetes service that references it. This is achieved with the following command,

```bash
kubectl expose deployment test-ml-score-api --port 5000 --type=LoadBalancer --name test-ml-score-api-lb
```

If you are using Docker Desktop, then this will automatically emulate a load balancer at `http://localhost:5000`. To find where Minikube has exposed its emulated load balancer run,

```bash
minikube service list
```

Now we test our new service - for example (with Docker Desktop),

```bash
curl -X POST -F image=@dog.jpeg 'http://localhost:5000/predict'
```

Note, neither Docker Desktop or Minikube setup a real-life load balancer (which is what would happen if we made this request on a cloud platform). To tear-down the load balancer, deployment and pod, run the following commands in sequence,

```bash
kubectl delete deployment test-ml-score-api
kubectl delete service test-ml-score-api-lb
```
