from rasa_nlu.model import Metadata, Interpreter
from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.components import ComponentBuilder
from rasa_nlu.model import Trainer

builder = ComponentBuilder(use_cache=True)      # will cache components between pipelines (where possible)

training_data = load_data('/root/rasa_nlu/classification/testData2.json')
trainer = Trainer(RasaNLUConfig("/root/rasa_nlu/classification/nlu_config.json"), builder)
trainer.train(training_data)
model_directory = trainer.persist('./projects/default/')  # Returns the directory the model is stored in

from rasa_nlu.model import Metadata, Interpreter

# where `model_directory points to the folder the model is persisted in
interpreter = Interpreter.load(model_directory, RasaNLUConfig("/root/rasa_nlu/classification/nlu_config.json"))

config = RasaNLUConfig("/root/rasa_nlu/classification/nlu_config.json")

# For simplicity we will load the same model twice, usually you would want to use the metadata of
# different models

interpreter = Interpreter.load(model_directory, config, builder)     # to use the builder, pass it as an arg when loading the model
# the clone will share resources with the first model, as long as the same builder is passed!
interpreter_clone = Interpreter.load(model_directory, config, builder)