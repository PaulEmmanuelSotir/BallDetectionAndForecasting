# 🎓 BallDetectionAndForecasting 🎓

INSA Lyon Deeplearning Course - exercice 1 - Ball detection and forecasting with deep learning on a synthetic dataset  
By Paul-Emmanuel Sotir

## Running instructions

``` shell
############## Installation ##############

git clone git@github.com:PaulEmmanuelSotir/BallDetectionAndForecasting.git
conda env create -f ./environement.yml
conda activate pytorch_5if
# Downloads datasets ('curl' and 'tar' packages needed on a Linux distro (or WSL - Linux subsystem on Windows)):
bash ./download_datasets.sh

############## Usage examples ##############

# Trains ball detection model
python -O ./src/train.py --model detect
# Performs an hyperparameter search for ball detection model (hyperopt)
python -O ./src/hp.py --model detect | tee ./docs/hp_search_logs/hp_detect4.log
# Trains ball position forecasting model
python -O ./src/train.py --model forecast
# Performs an hyperparameter search for ball position forecasting model (hyperopt)
python -O ./src/hp.py --model forecast | tee ./docs/hp_search_logs/hp_forecast2.log
```

Once hyperparameter searchs are done (or still running), you can use __```balldetect.torch_utils.extract_from_hp_search_log()```__ and __```balldetect.torch_utils.summarize_hp_search()```__ function to parse and summarize hyperparameter search results. See [./notebooks/hp_search_results.ipynb](./notebooks/hp_search_results.ipynb) (or [./docs/2_annexe_hp_search_results.html](./docs/2_annexe_hp_search_results.html) / [./docs/2_annexe_hp_search_results.pdf](./docs/2_annexe_hp_search_results.pdf)) for some hyperparameter search results of our own.

## Documentation

For (much) more details on this deeplearning course project see report located here: [./docs/rapport.md](./docs/rapport.md) (or [./docs/0_rapport.html](./docs/0_rapport.html) / [./docs/_rapport_export.pdf](./docs/_rapport_export.pdf)) _**(unfortunately, this report is in French.)**_

## Some cool emojis to use as is or animate someday, somehow; I don't really know why nor when... \^.^'

### 🎓🏅🏆🎯🧬🔬🧰📟💻⌨💽💾📡🔦💡📚📉📈⏲⏳⌛
### Loop it!: 👇👈👆👉👍
### Find some other cool loops among those emojis: 🙍‍♂️🙎‍♂️🙅‍♂️🙆‍♂️🧏‍♂️💁‍♂️🙋‍♂️🤦‍♂️🤷‍♂️💆‍♂️💇‍♂️🙇‍♂️
### Gradual emoji animation: 😶😗😙😚😕😐🙂😉😊😀😃😄😂😁
