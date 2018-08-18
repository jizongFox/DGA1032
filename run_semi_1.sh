#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0  python semi-graphcut.py --saved_name lambda_1_graphcut_baseline
CUDA_VISIBLE_DEVICES=0  python semi-graphcut.py --lamda 0 --saved_name lambda_0_graphcut_baseline

