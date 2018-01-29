from rasa_nlu.model import Metadata, Interpreter
from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer

import csv 
import sys

training_data = load_data('/root/rasa_nlu/classification/testData5.json')
trainer = Trainer(RasaNLUConfig("/root/rasa_nlu/classification/config_all.json"))
trainer.train(training_data)
model_directory = trainer.persist('./projects/default/')  # Returns the directory the model is stored in


# where `model_directory points to the folder the model is persisted in

 
f = open(testcase.csv, ‘rb’)
reader = csv.reader(f)
for row in reader:
interpreter = Interpreter.load(model_directory, RasaNLUConfig("/root/rasa_nlu/classification/config_all.json"))

interpreter.parse(row)
f.close()
