import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array

from flask import request
from flask import jsonify
from flask import Flask

import tensorflow as tf

app = Flask(__name__)

def get_model():
	global model
	model = load_model('cnn_model2.h5')
	print("* Model Loaded!")

def preprocess_image(image, target_size):
	if image.mode != "RGB":
		image = image.convert("RGB")
	image = image.resize(target_size)
	image = img_to_array(image)
	image = np.expand_dims(image, axis = 0)

	return image

def pred(num):
	if num == 1:
		return "INFECTED"
	return "NOT INFECTED"
		
print("* Loading Keras Model...")
graph = tf.get_default_graph()
get_model()

@app.route("/predict", methods=["POST"])	
def predict():
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(decoded))
	processed_image = preprocess_image(image, target_size=(64, 64))

	global graph
	with graph.as_default():
		prediction = model.predict(processed_image)

		response = {
			'prediction': pred(prediction[0][0])
		}

		return jsonify(response) 