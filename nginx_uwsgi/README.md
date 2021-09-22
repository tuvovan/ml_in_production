# fastAPI demo

demo for deploying a deep learning model using RestAPI, nginx, uwsgi, flask and docker.

## Overview

Lets look at the overall architecture, image is borrowed from [hackernoon](https://hackernoon.com/a-guide-to-scaling-machine-learning-models-in-production-aa8831163846)

![Image of Yaktocat](architecture.png#center)

- **RestAPI** vs **Flask** work on the web server end, to recieve the request from use and to show the results returned from model server.
- **nginx**: the highly stable web server, which provides benefits such as load-balancing, SSL configuration, etc.
- **uWSGI**: a highly configurable WSGI server (Web Server Gateway Interface) that allows forking multiple workers to serve multiple requests at a time.
- **Docker**: to dockerize and run as a standalone application, reduce pain in the ass.

## How to run

Run

```bash
docker compose up
```

To validate the docker running, type:

```bash
docker ps -a
```

The results should be as follow:

```bash
CONTAINER ID   IMAGE                                 COMMAND                  CREATED         STATUS                        PORTS                                                                                                                                  NAMES
83f65933234e   4bb0cd2c996e                          "uwsgi /home/project…"   9 seconds ago   Up 6 seconds                                                                                                                                                         app
c4b2b766ea1f   d31986a4b4ae                          "nginx -g 'daemon of…"   9 seconds ago   Up 5 seconds                  0.0.0.0:80->80/tcp, :::80->80/tcp                                                                                                      server_nginx
037c3357a178   gcr.io/k8s-minikube/kicbase:v0.0.26   "/usr/local/bin/entr…"   18 hours ago    Exited (255) 38 minutes ago   127.0.0.1:55004->22/tcp, 127.0.0.1:55003->2376/tcp, 127.0.0.1:55002->5000/tcp, 127.0.0.1:55001->8443/tcp, 127.0.0.1:55000->32443/tcp   minikube
```

then we are good to go. Lets make a request and see how the model see this funny dog.

![Image of Yaktocat](dog.jpeg#center)

```bash
curl -X POST -F image=@doge.jpeg http://localhost:80/predict
```

The result returned from the model server:

```json
{
  "predictions": [
    { "label": "pug", "probability": 0.6822489500045776 },
    { "label": "bull_mastiff", "probability": 0.08646740764379501 },
    { "label": "tub", "probability": 0.023030536249279976 },
    { "label": "tennis_ball", "probability": 0.02032969892024994 },
    { "label": "Brabancon_griffon", "probability": 0.016338394954800606 }
  ],
  "success": true
}
```

this time the model correctly predicted the pug 😆
