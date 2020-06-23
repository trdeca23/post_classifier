"""
This script builds a pipeline to classify text using supervised learning

!! train.py `epochs` parameter currently set at 1 to decrease running time during testing; be sure to change this after testing

python3 environment defined by the kaggle/python docker image: https://github.com/kaggle/docker-python 
comes with the necessary analytics libraries installed but might be overkill

"""

#cd to source
#create output and generated files

import os
from sklearn.model_selection import train_test_split
from numpy import save
import pickle
from pathlib import Path


os.chdir('source')


#local
from variables import model_loc, model_type, pipeline_loc
from load_data import load_data
from clean import cleaned_data
from preprocess import preprocessed_data
from load_embeddings import load_embeddings
from train import trained_model
from new_data_pipeline import new_data_pipeline


Path("../output").mkdir(parents=True, exist_ok=True) #prepare directory for output
Path("generated").mkdir(parents=True, exist_ok=True) #prepare directory for saving binary objects, e.g., model


#load data
pandas_df = load_data()
print('DATASET LOADED')


#clean data
this_cleaned_data = cleaned_data(pandas_df)
print('DATASET CLEANED')


#split data into training and testing
x_train,x_test,y_train,y_test = train_test_split(this_cleaned_data.data.text, this_cleaned_data.data.subject, random_state = 0)
print('Training set has the following labels:', y_train.unique())
print('DATASET SEPARATED INTO TRAIN AND TEST SETS')


#pre-process feature data, not targets/labels
this_preprocessed_data = preprocessed_data(x_train, x_test)
print('DATASET PRE-PROCESSED')


#load embeddings
embedding_matrix = load_embeddings()
print('DATASET EMBEDDINGS LOADED')


#train model, print out some performance metrics, and return: model object, performance history, function for decoding labels
this_trained_model = trained_model(
	x_train=this_preprocessed_data.train, 
	y_train=y_train,\
	x_test=this_preprocessed_data.test, 
	y_test=y_test, 
	embedding_matrix=embedding_matrix,
	model_type=model_type
	)
print('DATASET TRAINED')


#save model - calling `save('my_model')` creates a SavedModel folder `my_model`.
this_trained_model.model.save(model_loc)
print('MODEL SAVED')


#build new data pipeline (for making predictions on new unprocessed data) and pickle it
if model_type is "classifier":
	this_trained_model.decode_label = this_trained_model.decode_label_classifier
elif model_type is "autoencoder":
	this_trained_model.decode_label = this_trained_model.decode_label_autoencoder
pipeline = new_data_pipeline(
	cleaning_function=this_cleaned_data.clean_new_observations,\
	preprocessing_function=this_preprocessed_data.preprocess_new_observations,\
	decode_label_function=this_trained_model.decode_label\
	) #add .model to this_trained_model.model
with open(pipeline_loc, 'wb') as handle:
	pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('PREDICTION PIPELINE SAVED')


if model_type is "classifier":
#command line can be used to make prediction/s with a .txt file and a model
#...in the following manner: python predict.py my_text.txt my_model
#...and returns a prediction.txt file containing the predicted classification/s
	print('CLASSIFICATIONS CAN BE MADE FROM THE COMMAND LINE AS FOLLOWS:')
	print('python source/classify.py [point_to_observation_to_classify].txt [point_to_model]')
	print('The .txt should contain text from one or more observations to be classified, with new lines separating observations.')
	print('PREDICTION PIPELINE COMPLETE')
	print('Have a nice day!')

elif model_type is "autoencoder":
#command line can be used to make encoding/s with a .txt file and a model
#...in the following manner: python encode.py my_text.txt my_model
#...and returns an encoding.txt file containing the vectors representing text
	print('ENCODINGS CAN BE MADE FROM THE COMMAND LINE AS FOLLOWS:')
	print('python source/predict.py [point_to_observation_to_encode].txt [point_to_model]')
	print('The .txt should contain text from one or more observations to be encoded, with new lines separating observations.')
	print('PREDICTION PIPELINE COMPLETE')
	print('Have a nice day!')




#Sources:
#https://www.kaggle.com/madz2000/nlp-using-glove-embeddings-99-8-accuracy
#https://machinelearningmastery.com/lstm-autoencoders/
#https://www.kaggle.com/harishreddy18/english-to-french-translation
#https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
#https://www.kaggle.com/tanulsingh077/deep-learning-for-nlp-zero-to-transformers-bert?
#https://towardsdatascience.com/applying-machine-learning-to-classify-an-unsupervised-text-document-e7bb6265f52


