import numpy as np
from preprocess import max_features
import pickle


#local
from variables import embeddings_loc, tokenizer_loc


def load_embeddings():
	"""
    Return embedding_matrix for tokenizer
    :param tokenizer_loc: location of pickled file containing object of class 'keras_preprocessing.text.Tokenizer' that has been fit on data
    :param max_features: Use only the top max_features from 'keras_preprocessing.text.Tokenizer'
    :return: embedding_matrix
    """

	embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(embeddings_loc))

	all_embs = np.stack(list(embeddings_index.values()))
	emb_mean,emb_std = all_embs.mean(), all_embs.std()
	embed_size = all_embs.shape[1]

	with open(tokenizer_loc, 'rb') as handle:
		tokenizer = pickle.load(handle)

	word_index = tokenizer.word_index
	nb_words = int(np.nanmin([max_features, len(word_index)]))
	#change below line if computing normal stats is too slow
	embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

	for word, i in word_index.items():
		if i >= max_features: continue
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None: embedding_matrix[i] = embedding_vector

	print("Size of embedding matrix:", embedding_matrix.shape)
	print()

	return embedding_matrix



def get_coefs(word, *arr): 
	return word, np.asarray(arr, dtype='float32')
