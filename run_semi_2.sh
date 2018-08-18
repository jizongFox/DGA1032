#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python semi-main.py --lamda 0.1  --saved_name lambda_0.1_graphcut_baseline
CUDA_VISIBLE_DEVICES=1 python semi-main.py --lamda 5 --saved_name lambda_5_graphcut_baseline

