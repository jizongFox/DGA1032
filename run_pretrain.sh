#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python semi-main.py --split_ratio 0.01
CUDA_VISIBLE_DEVICES=1 python semi-main.py --split_ratio 0.03
CUDA_VISIBLE_DEVICES=1 python semi-main.py --split_ratio 0.05
CUDA_VISIBLE_DEVICES=1 python semi-main.py --split_ratio 0.1
CUDA_VISIBLE_DEVICES=1 python semi-main.py --split_ratio 0.2
