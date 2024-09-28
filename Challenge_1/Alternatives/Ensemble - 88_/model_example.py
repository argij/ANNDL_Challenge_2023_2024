import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

class model:
    def __init__(self, path):
        self.model_1 = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel', 'first'))
        self.model_2 = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel','second'))
        self.model_3 = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel', 'third'))

    def predict(self, X):
        
        X = preprocess_input(X)

        out_1 = tf.argmax(self.model_1.predict(X), axis=-1)
        out_2 = tf.argmax(self.model_2.predict(X), axis=-1)
        out_3 = tf.argmax(self.model_3.predict(X), axis=-1)

        out = tf.math.round((out_1 + out_2 + out_3)/3)

        return out