#!/usr/bin/env bash
python semi-graphcut-crossentropy.py --lamda 0.01 --saved_name lambda_0.01_graphcut_CE_baseline && python semi-graphcut-crossentropy.py --lamda 0.1 --saved_name lambda_0.1_graphcut_CE_baseline
python semi-graphcut-crossentropy.py --lamda 1 --saved_name lambda_1_graphcut_CE_baseline && python semi-graphcut-crossentropy.py --lamda 5 --saved_name lambda_5_graphcut_CE_baseline