#!/bin/sh
pipenv run python training/run_experiment.py --gpu 0 --save '{"dataset": "IamLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc", "train_args":{"batch_size": 4, "epochs": 70, "username": "praveen-palanisamy"}}'
