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


def remove_seam(img4, seam):
  """
  Removes 1 seam from the image either vertical or horizontal

  Returns
  =======
    4-D image with seam removed from all layers
  """
  width = img4.shape[0] if img4.shape[0] == seam.shape[0] else img4.shape[0] - 1
  height = img4.shape[1] if img4.shape[1] == seam.shape[1] else img4.shape[1] - 1
  new_img = np.zeros((
    width,
    height,
    img4.shape[2],
  ))
  for i, seam_row in enumerate(seam):
    img_row = img4[i]
    for col in seam_row:
      img_row = np.delete(img_row, col, axis=0)
    new_img[i] = img_row

  return new_img

def add_seam(img4, seam):
  """
  Adds the provided seam in either the horizontal or verical direction

  Returns
  =======
    4-D image with seam added to all layers
  """
  width = img4.shape[0] if img4.shape[0] == seam.shape[0] else img4.shape[0] + 1
  height = img4.shape[1] if img4.shape[1] == seam.shape[1] else img4.shape[1] + 1
  new_img = np.zeros((
    width,
    height,
    img4.shape[2],
  ))
  for i, seam_row in enumerate(seam):
    img_row = img4[i]
    for col in seam_row:
      new_pixel = img_row[col]
      img_row = np.insert(img_row, col + 1, new_pixel, axis=0)
    new_img[i] = img_row

  return new_img


def find_seams(eng):
  """
  Adds the provided seam in either the horizontal or verical direction

  Returns
  =======
    Tuple (M, P) 2-D matrices where M is the cummulative energy along a path
    and P is the parent along the path
  """
  rows = len(eng)
  cols = len(eng[0])
  M = np.zeros(shape=(rows, cols))
  P = np.zeros(shape=(rows, cols))
  M[0] = eng[0]
  P[0] = [-1] * cols
  inf = float('Inf')

  for r in range(1, rows):
    for c in range(0, cols):
      option_1 = M[r-1, c-1] if (c > 0) else inf
      option_2 = M[r-1, c] if (c < cols) else inf
      option_3 = M[r-1, c+1] if (c < cols - 1) else inf

      if (option_1 <= option_2 and option_1 <= option_3):
        M[r, c] = eng[r, c] + M[r-1, c-1]
        P[r, c] = c-1
      elif (option_2 <= option_1 and option_2 <= option_3):
        M[r, c] = eng[r, c] + M[r-1, c]
        P[r, c] = c
      else:
        M[r, c] = eng[r, c] + M[r-1, c+1]
        P[r, c] = c+1

  P = P.astype(int)
  return (M, P)