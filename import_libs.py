# Importing libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, zipfile
import os
import glob
import math
import random
import time
import datetime
import shutil
import imageio

from tqdm import tqdm, tqdm_notebook

from dataclasses import dataclass
from pathlib import Path

import warnings

from scipy import linalg

import xml.etree.ElementTree as ET

import cv2
from PIL import Image

import tensorflow as tf

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape,\
Conv2DTranspose, Conv2D, Flatten, Dropout, Embedding, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.contrib.eager.python import tfe

from tensorflow.keras import backend as K

from keras.engine import *
from keras.legacy import interfaces

from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints

from keras.utils.generic_utils import func_dump
from keras.utils.generic_utils import func_load
from keras.utils.generic_utils import deserialize_keras_object
from keras.utlis.generic_utils import has_arg

from keras.utils import conv_utils

print(os.listdir("../Input"))