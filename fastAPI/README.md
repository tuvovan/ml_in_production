# fastAPI demo

demo for deploying a deep learning model using fastAPI, resdis and docker.

## Overview

- FastAPI works on the web server end, to recieve the request from use and to show the results returned from model server.
- Redis works as a message queue, handle the multiple requests from user. Main functions would be:
  - store the request messages from web server
  - store the results that were returned from the model server
- Docker: to dockerize and run as a standalone application, reduce pain in the ass.

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
CONTAINER ID   IMAGE                                 COMMAND                  CREATED          STATUS                       PORTS                                                                                                                                  NAMES
f160a36f6579   5d1ea8317949                          "/start.sh"              25 seconds ago   Up 19 seconds                0.0.0.0:80->80/tcp, :::80->80/tcp                                                                                                      fastapi_webserver_1
3dfd27b11594   92b9a0355525                          "python /app/main.py"    25 seconds ago   Up 20 seconds                                                                                                                                                       fastapi_modelserver_1
bda69d5bf828   576c0aa0d36f                          "docker-entrypoint.s…"   25 seconds ago   Up 23 seconds                6379/tcp                                                                                                                               fastapi_redis_1
037c3357a178   gcr.io/k8s-minikube/kicbase:v0.0.26   "/usr/local/bin/entr…"   18 hours ago     Exited (255) 8 minutes ago   127.0.0.1:55004->22/tcp, 127.0.0.1:55003->2376/tcp, 127.0.0.1:55002->5000/tcp, 127.0.0.1:55001->8443/tcp, 127.0.0.1:55000->32443/tcp   minikube
```

then we are good to go. Lets make a request and see how the model see this funny dog.

![Image of Yaktocat](doge.jpeg#center)

```bash
curl -X POST -F img_file=@doge.jpeg http://localhost:80/predict
```

The result returned from the model server:

```json
{
  "success": true,
  "predictions": [
    { "label": "dingo", "probability": 0.523766815662384 },
    { "label": "Pembroke", "probability": 0.17858819663524628 },
    { "label": "basenji", "probability": 0.15123720467090607 },
    { "label": "Eskimo_dog", "probability": 0.031266018748283386 },
    { "label": "bath_towel", "probability": 0.01143248938024044 }
  ]
}
```

or we can just go to

[localhost:80/docs](localhost:80/docs)

click try it out, upload the image and see.

![FastAPI UI](fastapiUI.png#center)

seems like the ResNet50 doesn't have a Shiba dog class but Dingo dog 😆
