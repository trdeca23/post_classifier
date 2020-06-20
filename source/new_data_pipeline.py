from numpy import argmax

class new_data_pipeline:

	def __init__(self, cleaning_function, preprocessing_function, decode_label_function):
		self.cleaning_function = cleaning_function
		self.preprocessing_function = preprocessing_function
		self.decode_label_function = decode_label_function

	def clean(self, data):
		return self.cleaning_function(data)

	def preprocess(self, data):
		return self.preprocessing_function(data.text)

	def decode_label(self, label):
		return self.decode_label_function(label)

	def model_predict(self, observation, model):
		return argmax(model.predict(observation), axis=-1)
		
	def get_prediction(self, observation, model):
		observation = self.preprocess(self.clean(observation))
		model_prediction = self.model_predict(observation, model)
		decoded_prediction = self.decode_label(model_prediction)
		return decoded_prediction