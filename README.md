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
python -O ./src/hp.py --model detect
# Trains ball position forecasting model
python -O ./src/train.py --model forecast
# Performs an hyperparameter search for ball position forecasting model (hyperopt)
python -O ./src/hp.py --model forecast
```


# 🎓🏅🏆🎯🧬🔬🧰📟💻⌨💽💾📡🔦💡📚📉📈⏲⏳⌛
# 🙍‍♂️🙎‍♂️🙅‍♂️🙆‍♂️🧏‍♂️💁‍♂️🙋‍♂️🤦‍♂️🤷‍♂️💆‍♂️💇‍♂️🙇‍♂️
# 👇👈👆👉
# 👍
# 😶😗😕😐😙😚🙂😊😉😀😃😄😂😁