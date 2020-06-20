from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense,Embedding,LSTM,Input,RepeatVector,TimeDistributed,Bidirectional,GRU
import keras
import pickle
from numpy import argmax, max as npmax
import sklearn


#local
from variables import max_text_len, tokenizer_loc, batch_size, epochs, label_encoder_loc


learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.75, min_lr=0.00001)


class trained_model:
	"""
	:param x_train: class 'numpy.ndarray', with .shape (n, max_text_len)
	:param y_train: of type pandas.core.series.Series, with .shape (n,)
	:param x_test: class 'numpy.ndarray', with shape (m, max_text_len)
	:param y_test: of type pandas.core.series.Series, with shape (m,)
	:param embedding_matrix: a class 'numpy.ndarray' of .shape (rows, max_text_len) containing an embedding matrix, where rows = min(max_features, vocabulary size)
	:param model_type: either 'classifier' or 'autoencoder'
	:attribute label_encoder: class 'sklearn.preprocessing.label.LabelEncoder' containing integer codes for the levels in y_train.unique()
	:attribute model: either a classifier (class 'keras.engine.sequential.Sequential') or an autoencoder, depending on the value of `model_type`
	:attribute history: class 'keras.callbacks.callbacks.History' containing training metrics for each epoch
	"""

	def __init__(self, x_train, y_train, x_test, y_test, embedding_matrix, model_type='classifier'):
		self.train(x_train, y_train, x_test, y_test, embedding_matrix, model_type)



	def train(self, x_train, y_train, x_test, y_test, embedding_matrix, model_type):
		"""
	    Train model
	    :param x_train:
	    :param y_train:
		:param x_test:
		:param y_test:
	    :param embedding_matrix:
	    :param model_type:
	    """

		y_train, y_test = self.preprocess(x_train, y_train, x_test, y_test, model_type)

		self.make_model(embedding_matrix, x_train, y_train, x_test, y_test, model_type=model_type)
		
		self.history = self.fit(x_train, y_train, x_test, y_test, batch_size, epochs, learning_rate_reduction, model_type)

		print()



	def preprocess(self, x_train, y_train, x_test, y_test, model_type, label_encoder_loc=label_encoder_loc):

		if model_type is 'classifier':
			# encode class values as integers
			label_encoder = LabelEncoder()
			label_encoder.fit(y_train)
			encoded_y_train = label_encoder.transform(y_train)
			encoded_y_test = label_encoder.transform(y_test)

			# convert integers to dummy variables (i.e. one hot encoded)
			y_train = np_utils.to_categorical(encoded_y_train)
			y_test = np_utils.to_categorical(encoded_y_test)

			self.label_encoder = label_encoder

			#save labelencoder to use it later for making new predictions later
			with open(label_encoder_loc, 'wb') as handle:
				pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

		elif model_type is 'autoencoder':
			#Because this is for an auto-encoder, the labels will be the same as the features
			y_train = x_train
			y_test = x_test

			#Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
			y_train = y_train.reshape(*y_train.shape, 1)
			y_test = y_test.reshape(*y_test.shape, 1)

		print('Train features shape:', x_train.shape)
		print('Test features shape:', x_test.shape)
		print('Train labels shape:', y_train.shape)
		print('Test labels shape:', y_test.shape)

		return y_train, y_test



	def fit(self, x_train, y_train, x_test, y_test, batch_size, epochs, learning_rate_reduction, model_type):
		#fit model to data
		history = self.model.fit(\
			x_train,\
			y_train,\
			batch_size = batch_size,\
			validation_data = (x_test,y_test),\
			epochs = epochs,\
			callbacks = [learning_rate_reduction])

		#analysis broad results
		print('Accuracy of the model on Training Data is - ' , self.model.evaluate(x_train,y_train)[1]*100)
		print('Accuracy of the model on Testing Data is - ' , self.model.evaluate(x_test,y_test)[1]*100)
		print('First five observations for the test set:', [self.decode_label_autoencoder(p) for p in x_test[:5,:]])

		if model_type is "classifier":
			print('First five labels for test set:', self.decode_label_classifier(y_test[:5,:], self.label_encoder))
			print('First five predictions for test set:', self.decode_label_classifier(self.model.predict_classes(x_test[:5,:]), self.label_encoder))
		elif model_type is "autoencoder":
			print('First five labels for the test set:', [self.decode_label_autoencoder(p) for p in y_test[:5,:,:]])
			five_predictions = argmax(self.model.predict(x_test[:5,:]), axis=-1)
			print('First five predictions for the test set:', [self.decode_label_autoencoder(p) for p in five_predictions])

		return history



	def make_model(self, *args, model_type, **kwargs):
		if model_type is 'classifier':
			self.make_classifier(*args, **kwargs)
		elif model_type is 'autoencoder':
			self.make_autoencoder(*args, **kwargs)


	
	def make_classifier(self, embedding_matrix, x_train, y_train, x_test, y_test):
		#Defining Neural Network
		model = Sequential()
		#Non-trainable embeddidng layer
		model.add(Embedding(embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=max_text_len, trainable=False))
		#LSTM 
		model.add(LSTM(units=128 , return_sequences = True , recurrent_dropout = 0.25 , dropout = 0.25))
		model.add(LSTM(units=64 , recurrent_dropout = 0.1 , dropout = 0.1))
		model.add(Dense(units = 32 , activation = 'relu'))
		model.add(Dense(units = y_train.shape[1], activation='softmax'))
		model.compile(optimizer=keras.optimizers.Adam(lr = 0.01), loss='categorical_crossentropy', metrics=['accuracy'])

		print(model.summary())

		self.model = model
		


	def make_autoencoder(self, embedding_matrix, x_train, y_train, x_test, y_test):
		#Defining Neural Network
		#define encoder
		visible = Input(shape=(max_text_len,))
		#Non-trainable embedding layer
		encoder = Embedding(embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=max_text_len, trainable=False)(visible)
		encoder = Bidirectional(GRU(embedding_matrix.shape[1]))(encoder)
		#encoder = LSTM(units=embedding_matrix.shape[1], activation='relu')(encoder) #perhaps try Bidirectional GRU here

		#define reconstruct decoder
		decoder1 = RepeatVector(max_text_len)(encoder)
		decoder1 = Bidirectional(GRU(embedding_matrix.shape[1],return_sequences=True))(decoder1)
		#decoder1 = LSTM(embedding_matrix.shape[1], activation='relu', return_sequences=True)(decoder1) #perhaps try Bidirectional GRU here
		decoder1 = TimeDistributed(Dense(embedding_matrix.shape[0], activation='softmax'))(decoder1)

		model = Model(inputs=visible, outputs=decoder1)
		model.compile(optimizer=keras.optimizers.Adam(lr = 0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

		print(model.summary())

		self.model = model



	@staticmethod
	def decode_label_classifier(label, label_encoder=label_encoder_loc):
		"""
    	Decode labels according to label_encoder
		:param label: either a class 'numpy.ndarray' of one hot labels with .shape (n, number of levels), or class 'numpy.ndarray' of .shape (n,)
    	:param label_encoder: either class 'sklearn.preprocessing.label.LabelEncoder' or the location for a pickled file containing such an object
    	:return: a class 'numpy.ndarray' of .shape (n,), containing string labels corresponding with the classes_ attribute of the sklearn.preprocessing.label.LabelEncoder
    	"""
		if (not isinstance(label_encoder, sklearn.preprocessing.label.LabelEncoder)): #if no sklearn LabelEncoder is provided as an argument:
			with open(label_encoder_loc, 'rb') as handle:
				label_encoder = pickle.load(handle) #then read in pickled LabelEncoder
		if (label.ndim == 2): #to decode categorical i.e., one hot labels
			decode = label_encoder.inverse_transform(argmax(label, axis=1))
		elif(label.ndim == 1): #to decode class levels
			decode = label_encoder.inverse_transform(label)
		return decode



	@staticmethod
	def decode_label_autoencoder(label, tokenizer_loc=tokenizer_loc):
		"""
    	Decode labels according to tokenizer
		:param label: class 'numpy.ndarray', .ndim of 1, containing tokens and padding
		:param tokenizer_loc: location of pickled file containing object of class 'keras_preprocessing.text.Tokenizer'
    	:return: text corresponding with the word_index attribute of the 'keras_preprocessing.text.Tokenizer'
    	"""
		with open(tokenizer_loc, 'rb') as handle:
			tokenizer = pickle.load(handle)

		y_id_to_word = {value: key for key, value in tokenizer.word_index.items()}
		y_id_to_word[0] = '<PAD>'
		text = ' '.join([y_id_to_word[npmax(x)] for x in label])
		return text

