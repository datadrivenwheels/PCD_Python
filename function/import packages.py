import pandas as pd
from mmdet.apis import DetInferencer
from rich.pretty import pprint
from collections import Counter
import os
import matplotlib.pyplot as plt
import numpy as np
from pygam import LinearGAM, s
from sklearn.metrics import precision_recall_curve, average_precision_score
import nltk

from sklearn.linear_model import LinearRegression
from scipy.interpolate import UnivariateSpline
from scipy.stats import chi2, norm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
