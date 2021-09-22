import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input

resnetmodel = ResNet50(
    weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))

resnetmodel.save('resnet50.h5')

tf.keras.backend.set_learning_phase(0)
model = tf.keras.models.load_model('./resnet50.h5')
export_path = '../my_classifier/1'

tf.saved_model.save(model, export_path)
