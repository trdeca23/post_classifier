#This .py script should be run from command line as follows: "python [point_to_this_script].py [point_to_observation_to_classify].txt [point_to_model]"
#The .txt should contain text from exactly one observation to be classified, with no new lines within that text.
import sys
import os
import pandas as pd
import pickle
from numpy import load,savetxt
from tensorflow.keras.models import load_model, Model
from pathlib import Path


#local
from variables import pipeline_loc

observation = pd.read_csv(sys.argv[1], header=None, sep="\n", names=["text"])
reconstructed_model = load_model(sys.argv[2]) #load model

os.chdir('source')
Path("../output").mkdir(parents=True, exist_ok=True) #prepare directory for output if it doesn't already exist

handle = open(pipeline_loc, "rb")
pipeline = pickle.load(handle)
sequence = pipeline.preprocess(pipeline.clean(observation))
model = Model(inputs=reconstructed_model.inputs, outputs=reconstructed_model.layers[2].output)
prediction = model.predict(sequence) # get the feature vector for the input sequence

savetxt('../output/encoding.txt', prediction, fmt='%.5e')
print("Encoding/s written to `encoding.txt` in project directory.")



