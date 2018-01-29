from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.components import ComponentBuilder
from rasa_nlu.model import Trainer

builder = ComponentBuilder(use_cache=True)      # will cache components between pipelines (where possible)

training_data = load_data('/root/rasa_nlu/classification/testData2.json')
trainer = Trainer(RasaNLUConfig("/root/rasa_nlu/classification/nlu_config.json"), builder)
trainer.train(training_data)
model_directory = trainer.persist('./projects/default/')  # Returns the directory the model is stored in