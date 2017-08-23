from scipy import misc, signal, ndimage
from scipy.ndimage.filters import gaussian_gradient_magnitude
from scipy.ndimage import sobel, generic_gradient_magnitude
import matplotlib.pyplot as plt
import numpy as np


def normalize(img, max_value=255.0):
  """
  Normalizes all values in the provided image to lie between 0 and
  the provided max value
  """
  mins = np.min(img)
  normalized = np.array(img) + np.abs(mins)
  maxs = np.max(normalized)
  normalized *= (max_value/maxs)

  return normalized


def rgb_to_gray(img):
  r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray


def apply_filter(img, img_filter):
  """
  Applies crosss correlation filter to provided image
  """
  return signal.correlate(img, img_filter, mode='same')


def compute_eng_grad(img):
  """
  Computes the gradient magnitude matrix of the provided image
  """
  bw_img = rgb_to_gray(img)
  eng = generic_gradient_magnitude(bw_img, sobel)
  eng = gaussian_gradient_magnitude(bw_img, 1)
  return normalize(eng)

# TODO -- MAKE SURE THE NEW_IMG SHAPE IS ADJUSTED FOR THE
# SEAM YOU ARE REMOVING
def remove_seam(img4, seam):
  new_img = np.zeros((
    img4.shape[0],
    img4.shape[1],
    3,
  ))
  print(seam.shape)
  print(new_img.shape)
  for i, row in enumerate(seam):
    img_row = img4[i]
    for col in row:
      img_row = np.delete(img_row, col, axis=0)
    print(img_row.shape)
    new_img[i] = img_row
  