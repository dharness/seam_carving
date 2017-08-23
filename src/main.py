from scipy import misc, signal, ndimage
from scipy.ndimage.filters import gaussian_gradient_magnitude
import matplotlib.pyplot as plt
import numpy as np

def normalize(im):
  return (255*(im - np.max(im))/-np.ptp(im)).astype(int)

def rgb_to_gray(im):
  r, g, b = im[:,:,0], im[:,:,1], im[:,:,2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray

def compute_eng_color(im, W):
  eng = np.dstack((
    im[:,:,0]*W[0],
    im[:,:,1]*W[1],
    im[:,:,2]*W[2])
  )
  eng = np.sum(eng, axis=2)
  return normalize(eng)

def compute_eng_grad(im):
  bw_img = rgb_to_gray(im)
  eng = gaussian_gradient_magnitude(bw_img, 1)
  return normalize(eng)

def compute_eng(im, W):
  eng_color = compute_eng_color(im, W)
  eng_grad = compute_eng_grad(im)
  eng = np.add(
    eng_color,
    eng_grad
  )
  return eng

def find_seams(eng):
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

def remove_seam(im, seam):
  new_img = []
  for i, row in enumerate(im):
    new_img.append(np.delete(row, seam[i], axis=0))
  return np.array(new_img)

def get_best_seam(M, P):
  rows = len(P)
  seam = [None] * rows
  i = P[-1].argmin(axis=0)
  seam[rows-1] = i
  cost = 0
  for r in reversed(range(0, rows)):
    seam[r] = i
    i = P[r][i]
    cost = cost + M[i]
  return (seam, cost)

if (__name__ == '__main__'):
  im1 = misc.imread('cat.png')
  print(im1.shape)

  im = [[
    [101, 244, 231, 126, 249],
    [151, 249, 219,   9,  64],
    [88,  93,  21, 112, 155],
    [114,  55,  55, 120, 205],
    [84, 154,  24, 252,  63],
    ],[
    [115, 228, 195,  68, 102],
    [92,  74, 216,  64, 221],
    [218, 134, 123,  35, 213],
    [229,  23, 192, 111, 147],
    [164, 218,  78, 231, 146],
    ],[
    [91, 201, 137,  85, 182],
    [225, 102,  91, 122,  60],
    [85,  46, 139, 162, 241],
    [101, 252,  31, 100,  69],
    [158, 198, 196,  26, 239],
  ]]

  print(im)


  W = [-3, 1, -3]
  eng = compute_eng(im1, W)
  M, P = find_seams(np.array([
    [3, 0, 6, 4, 2],
    [1, 3, 6, 6, 4],
    [4, 3, 4, 6, 2],
    [4, 6, 0, 0, 4],
    [0, 6, 5, 6, 6]
  ]))

  # seam, _ = get_best_seam(M, P)
  # new_img = remove_seam(np.array(im1), seam)
  # print(new_img)


  # plt.imshow(eng, cmap='gray')
  # plt.show()
  # plt.imshow(eng, cmap='gray')
  # plt.show()
  

