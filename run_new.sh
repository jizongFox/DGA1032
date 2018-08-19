#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python weakly-supervised-new.py --lr 1e-3 --b_weight 1e-1
CUDA_VISIBLE_DEVICES=0 python weakly-supervised-new.py --lr 5e-4 --b_weight 5e-2
CUDA_VISIBLE_DEVICES=0 python weakly-supervised-new.py --lr 5e-4 --b_weight 1e-2
CUDA_VISIBLE_DEVICES=0 python weakly-supervised-new.py --lr 5e-4 --b_weight 5e-3
CUDA_VISIBLE_DEVICES=0 python weakly-supervised-new.py --lr 5e-4 --b_weight 1e-3