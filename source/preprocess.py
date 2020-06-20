from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle


#local
from variables import max_features, max_text_len, tokenizer_loc


class preprocessed_data:
	"""
	:param train: of type pandas.core.series.Series, with .shape (n,)
	:param test: of type pandas.core.series.Series, with shape (m,)
	:attribute train: class 'numpy.ndarray', .ndim of 2, shape (n, max_text_len)
	:attribute test: class 'numpy.ndarray', .ndim of 2, shape (m, max_text_len)
	:attribute tokenizer: class 'keras_preprocessing.text.Tokenizer'
	"""
	def __init__(self, train, test):
		self.preprocess(train, test)


	def preprocess(self, train, test):
		"""
	    Preprocess x
	    :param train: List of text fields in training set
	    :param test: List of text fields in test set
	    :return: Tuple of (Preprocessed train with padding, Preprocessed test with padding, tokenizer built from train data, function to pre-process new data)
	    """
		train, test, tokenizer = self.tokenize(train, test)

		train = self.pad(train)
		test = self.pad(test)

		max_train_sequence_length = train.shape[1]
		max_test_sequence_length = test.shape[1]
		train_vocab_size = len(tokenizer.word_index)

		print("Max train sentence length (after setting maximum text length as {} words):".format(max_text_len), max_train_sequence_length)
		print("Max test sentence length (after setting maximum text length as {} words):".format(max_text_len), max_test_sequence_length)
		print("Train Vocabulary size (though the maximum allowed words is set at {}):".format(max_features), train_vocab_size)
		print()

		#save tokenizer to use it later for making new predictions later
		with open(tokenizer_loc, 'wb') as handle:
			pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

		self.train, self.test, self.tokenizer_loc = train, test, tokenizer_loc


	@staticmethod
	def tokenize(train, test):
		"""
	    Tokenize x
	    :param train: of type pandas.core.series.Series
	    :param test: List of sentences/strings in test set to be tokenized
	    :return: Tuple of (tokenized train data, tokenized test data, tokenizer used to tokenize train)
	    """
		tokenizer=Tokenizer(num_words=max_features)
		tokenizer.fit_on_texts(train)
		train=tokenizer.texts_to_sequences(train)
		test=tokenizer.texts_to_sequences(test)

		return train, test, tokenizer


	@staticmethod
	def pad(x):
		"""
	    Pad x
	    :param x: List of sequences.
	    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
	    :return: Padded numpy array of sequences
	    """
		# TODO: Implement
		padding=pad_sequences(x,padding='post',maxlen=max_text_len)

		return padding



	def preprocess_new_observations(self, x, tokenizer_loc=tokenizer_loc):
		"""
		:param x: of type pandas.core.series.Series, with shape (p,)
	    :param tokenizer: location of pickled file containing object of class 'keras_preprocessing.text.Tokenizer'
	    :return: class 'numpy.ndarray', .ndim of 2, shape (p, max_text_len)
		"""
		#load tokenizer used for data used to train model
		with open(tokenizer_loc, 'rb') as handle:
			tokenizer = pickle.load(handle)

		x = tokenizer.texts_to_sequences(x)

		x = self.pad(x)
		
		return(x)


