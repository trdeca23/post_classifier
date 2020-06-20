Natural Language Processing for Classification

These scripts will build a model (i.e., CLASSIFIER) that predicts whether text is more likely to be from a post related to: computers, recreation, science, politics, or religion (i.e., CLASSIFICATIONS). However, the scripts can easily accomodate other supervised classification models (see NOTES below).

These scripts can also be used to build a model (i.e., AUTOENCODER) that transforms input text into fixed length vectors (i.e., ENCODINGS). These can in turn be incorporated as features for other prediction models or more generally for dimensionality reduction.

Models are trained using a sci-kit learn dataset containing ~18000 newsgroup posts divided by topic (https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) and pre-trained word embeddings (fitted on other data; https://nlp.stanford.edu/projects/glove/).




# INSTRUCTIONS FOR USE WITHOUT DOCKER:

1. Create a working directory for the project with the name of your choice and clone GitHub repository https://github.com/trdeca23/nlp/post_classifier to it. Next steps should be done from within this working directory.


2. Install required packages listed in requirements.txt.


3. Download the pre-trained word embeddings: glove.twitter.27B.100d.txt, from https://www.kaggle.com/icw123/glove-twitter or https://nlp.stanford.edu/projects/glove/. Make sure that the correct location is specified by the `embeddings_loc` variable at the top of `source/variables.py`.


4. (Optional) Make any desired changes. See Options below. In particular, make sure to change `model_type` at the top of `source/variables.py` from "classifier" to "autoencoder" if you want to build a model that generates ENCODINGS rather than CLASS PREDICTIONS.


5. Build the model by running `python source/main.py`.


6. To generate predictions on new data contained in a .txt file, where observations are separated by newlines:

	* If the model was built with `model_type` as "classifier", to make classifications, run: 
	`python3 source/predict.py input/example_text4.txt source/generated/my_model`. Or replace `input/example_text4.txt` with any file name. Results will be saved in `output/prediction.txt`.

	* If the model was built with `model_type` as "encoder", to make encoding, run: 
	`python3 source/encode.py input/example_text4.txt source/generated/my_model`. Or replace `input/example_text4.txt` with any file name. Results will be saved in `output/encoding.txt`.




# INSTRUCTIONS FOR USE WITH DOCKER:

1. To be continued..




# NOTES: 

- Running `python source/main.py` took about 20 minutes to run, with `epochs` set to 20, on my 2014 Macbook: Memory - 8 GB 1600 MHz DDR3, Processor - 2.6 GHz Intel Core i5. Bulk of the time spent in training; to reduce training time (e.g., for testing), reduce `epochs` parameter in train.py.

- Options: Changes can be made by editing `source/variables.py`. For example, you can change the number of epochs that the model is trained (`epochs`), the size of the text that will be processed (`max_features`; `max_text_len`). To creat multiple models, change `model_loc` to specify different model names (e.g., if you are building both a classifier and an autoencoder). On the other hand, if you would like to build models fitted using different data, with potentially different categories, you will also need to tweak `source/load_data.py`. Note that other scripts will not need to be edited as long as this script outputs a pandas dataframe containing exactly two columns, with the text column labeled 'text', and the category column labeled 'subject'.

- Future implementations will add the following functionality: 
	Part 3. using encoder from auto-encoder, plus additional layers (output later should be softmax) predict categories using supervised and unsupervised models


