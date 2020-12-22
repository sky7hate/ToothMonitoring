from easydict import EasyDict
import os
import sys
import numpy as np

cfg = EasyDict()

"""
Path settings
"""
cfg.UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
cfg.LIB_DIR = os.path.dirname(cfg.UTILS_DIR)
cfg.ROOT_DIR = os.path.dirname(cfg.LIB_DIR)
cfg.DATA_DIR = os.path.join(cfg.ROOT_DIR, 'data')
cfg.OB_DIR = os.path.join(cfg.DATA_DIR, 'observation')