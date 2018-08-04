#!/bin/sh
pipenv run python training/run_experiment.py --gpu 1 --save '{"dataset": "EmnistLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc", "train_args":{"batch_size": 64, "epochs": 25}}'
