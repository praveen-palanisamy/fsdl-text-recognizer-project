#!/bin/sh
pipenv run python training/run_experiment.py --gpu 0 --save '{"dataset": "IamLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc", "train_args":{"batch_size": 2, "epochs": 15, "username": "praveen-palanisamy"}, "network_args":{"window_stride":1}}'
