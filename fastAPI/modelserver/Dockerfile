FROM python:3.7-slim-buster
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

RUN python -c "from tensorflow.keras.applications import ResNet50; ResNet50(weights='imagenet')"

COPY . /app
CMD [ "python", "/app/main.py" ]