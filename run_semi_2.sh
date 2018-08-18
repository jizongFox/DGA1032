#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python semi-main.py
CUDA_VISIBLE_DEVICES=1 python semi-main.py --lamda 0 --saved_name lambda_0

