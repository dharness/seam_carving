from scipy import misc, signal, ndimage
from scipy.ndimage.filters import gaussian_gradient_magnitude
import matplotlib.pyplot as plt
import numpy as np


def apply_filter(img, img_filter):
  return signal.correlate(img, img_filter, mode='same')