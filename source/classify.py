#This .py script should be run from command line as follows: "python [point_to_this_script].py [point_to_observation_to_classify].txt [point_to_model]"
#The .txt should contain text from one or more observations to be classified, new lines separating observations.
import sys
import os
import pandas as pd
import pickle
from numpy import load
from numpy import argmax
from tensorflow.keras.models import load_model


#local
from variables import pipeline_loc

observation = pd.read_csv(sys.argv[1], header=None, sep="\n", names=["text"])
reconstructed_model = load_model(sys.argv[2]) #load model

os.chdir('source')

handle = open(pipeline_loc, "rb")
pipeline = pickle.load(handle)
prediction = pipeline.get_prediction(observation, reconstructed_model) #make prediction and decode it
out_file = open("../output/prediction.txt", "w")
prediction_text = ["Prediction/s:\n"] + [p + "\n" for p in prediction] + ["\n\n"] + ["Observation/s:\n"] + [o + "\n\n" for o in observation.text]
out_file.writelines(prediction_text) #write to txt file
out_file.close()
print(*prediction_text, sep="")
print("Prediction/s written to `prediction.txt` in project directory.")
