#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Evaluate ball detection and forecast - Deeplearning Exercice 1 - Part 1
.. moduleauthor:: Paul-Emmanuel Sotir  
.. See https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html and https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting  
"""
import os
import sys
import json
import os.path
import argparse

from mlflow import log_metric, log_param, log_artifact
from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics

try:
    import cPickle as pickle
except ImportError:
    import pickle

__all__ = ['evaluate']


def evaluate(model, dataset):
    # load model pickle
    with open(model_path, 'rb') as model_pkl:
        model = pickle.load(model_pkl)

    # apply model prediction on dataset
    preds = model.predict(dataset, metadata={})  # TODO: metadata...

    # evaluate model predictions with type-specific metrics
    if args.type == "detect":
        metrics = {}  # TODO: preds against groud truth
    else:
        metrics = {}  # TODO: preds against groud truth
    return {"dataset": dataset, "model": model.name, "metrics": metrics}


def _log_eval_results(results, type, JSON_log, JSON_log_template='./eval_log_template.json'):
    print("Storing evaluation results to " + JSON_log)

    if not os.path.isfile(JSON_log):
        # Copy empty JSON evaluation log template
        shutil.copy(JSON_log_template, JSON_log)

    # Append results to evaluation log
    with open(JSON_log, "w") as f:
        log = json.load(f)
        log[type].append(results)
        json.dump(log, f)


def _main():
    parser = argparse.ArgumentParser(description='Evaluates model(s) ball detections or position forecasts.')
    parser.add_argument('--type', metavar='t', type=str, nargs=1,
                        help='Prediction task performed by model to be evaluated', choices=["detect", "forecast"])
    parser.add_argument('--models', metavar='f', action='extend', type=str, nargs='+',
                        help='Path to ball detection or forecasting model pickle(s) to be evaluated')
    parser.add_argument('--dataset', type=str, nargs=1,
                        help='Path to ball detection or forecasting evaluation dataset')
    parser.add_argument('--output', metavar='o', type=str, nargs='?', default='./eval_log.json',
                        help='Path to JSON evaluation log (created if it doesn\'t exist yet).')
    args = parser.parse_args()

    print('#' * 10 + " RUNNING EVALUATION SCRIPT... " + "#" * 10)
    print("#" * 5 + " " + len(args.models) + " models to be evaluated on \'" + args.type + "\' task.")

    for model in args.models:
        print("#" * 5 + " Evaluation of \'" + model + "\' model running...")
        results = evaluate(model, args.dataset)
        print("#" * 5 + " EVALUATION DONE.")
        print("> EVAL RESULTS = " + str(results))
        _log_eval_results(results, args.type, args.output)

    print("#" * 10 + " EVALUATION SCRIPT DONE! " + "#" * 10)


if __name__ == "__main__":
    _main()
