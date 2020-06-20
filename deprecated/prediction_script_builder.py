#below is no longer needed since it doesn't take any arguments and can therefore doesn't need to be custom built
def prediction_script_builder():
	with open('predict.py', 'w') as f:
		print('#This .py script should be run from command line as follows: "python [point_to_this_script].py [point_to_observation_to_classify].txt [point_to_model]"', file=f)
		print('#The .txt should contain text from exactly one observation to be classified, with no new lines within that text.', file=f)
		print('import sys', file=f)
		print('import os', file=f)
		print('import pandas as pd', file=f)
		print('import pickle', file=f)
		print('from numpy import load', file=f)
		print('from numpy import argmax', file=f)
		print('from tensorflow.keras.models import load_model', file=f)
		print('observation = pd.read_csv(sys.argv[1], header=None, sep="\\n", names=["text"])', file=f)
		print('reconstructed_model = load_model(sys.argv[2]) #load model', file=f)
		print('handle = open("pipeline.pickle", "rb")', file=f)
		print('pipeline = pickle.load(handle)', file=f)
		print('prediction = pipeline.get_prediction(observation, reconstructed_model) #make prediction and decode it',file=f)
		print('out_file = open("prediction.txt", "w")', file=f)
		print('prediction = ["Prediction:\\n"] + prediction.tolist() + ["\\n\\n"] + ["Observation:\\n"] + list(observation.text)', file=f)
		print('out_file.writelines(prediction) #write to txt file', file=f)
		print('out_file.close()', file=f)
		print('print(*prediction, sep="")', file=f)
		print('print("Prediction written to `prediction.txt` in project directory.")', file=f)
		
