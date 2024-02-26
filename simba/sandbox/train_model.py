# TRAIN MODELS

### This notebook assumes that you have annotated/labelled data within your SimBA project. The notebook also assumes
### that you have a apriori set the hyperparameters/sampling/evaluation settings either in the SimBA project_config.ini (when training a single model)
## or have a set of valid CSV config files inside the project_folder/configs directory of your SimBA project (when grid searching models).

from simba.model.grid_search_rf import GridSearchRandomForestClassifier
from simba.model.train_rf import TrainRandomForestClassifier

### Define the path to the SimBA project config ini.
CONFIG_PATH = "/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini"
#
# ###Create an instance of a model trainer based on the settings in the project config
# model_trainer = TrainRandomForestClassifier(config_path=CONFIG_PATH)
#
# ###Run the model trainer based on the settings in the project config
# model_trainer.run()
#
# ###Save the model
# model_trainer.save()

# TRAIN MULTIPLE MODELS: ONE FOR EACH SETTINGS FILE PRESENT IN THE PROJECT_FOLDER/CONFIGS DIRECTORY.
###Create an instance of a grid model trainer.
model_trainer = GridSearchRandomForestClassifier(config_path=CONFIG_PATH)

###Run the grid search model trainer. Note: Each model is saved without the need to call the save function (as when training a single model above).
model_trainer.run()
