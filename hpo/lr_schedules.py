# Source: https://github.com/automl/nas_benchmarks/blob/master/data_collection/train_fcnet.py
import numpy as np

def fix(epoch, initial_lr):
    return initial_lr


def exponential(epoch, initial_lr, T_max, decay_rate=0.96):
    return initial_lr * decay_rate ** (epoch / T_max)


def cosine(epoch, initial_lr, T_max):
    final_lr = 0
    return final_lr + (initial_lr - final_lr) / 2 * (1 + np.cos(np.pi * epoch / T_max))