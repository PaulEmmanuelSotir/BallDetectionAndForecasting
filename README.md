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
bash ./download_dataset.sh

############## Usage examples ##############

# Trains ball detection model
python -O ./src/train.py --model detect
# Performs an hyperparameter search for ball detection model (hyperopt)
python -O ./src/hp.py --model detect | tee ./hp_search_logs/hp_detect5.log
# Trains ball position forecasting model
python -O ./src/train.py --model forecast
# Performs an hyperparameter search for ball position forecasting model (hyperopt)
python -O ./src/hp.py --model forecast | tee ./hp_search_logs/hp_forecast3.log
```

Once hyperparameter searchs are done (or still running), you can use __```balldetect.torch_utils.extract_from_hp_search_log()```__ and __```balldetect.torch_utils.summarize_hp_search()```__ function to parse and summarize hyperparameter search results. See ```./notebooks/ball_detection_hp_search_results``` (or ```./docs/rapport.md```) for some hyperparameter search results of our own.

## Documentation

For (much) more details on this deeplearning course project see report located here: [./docs/rapport.md](./docs/rapport.md) unfortunately, this report is in French.

## Some cool emojis to use as is or animate someday, somehow; I don't really know why nor when... \^.^'

### 🎓🏅🏆🎯🧬🔬🧰📟💻⌨💽💾📡🔦💡📚📉📈⏲⏳⌛
### Loop it!: 👇👈👆👉👍
### Find some other cool loops among those emojis: 🙍‍♂️🙎‍♂️🙅‍♂️🙆‍♂️🧏‍♂️💁‍♂️🙋‍♂️🤦‍♂️🤷‍♂️💆‍♂️💇‍♂️🙇‍♂️
### Gradual emoji animation: 😶😗😙😚😕😐🙂😉😊😀😃😄😂😁
