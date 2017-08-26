import unittest
import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
from seam_carver import (
    apply_filter,
    normalize,
    compute_eng_grad,
    compute_eng_color,
    compute_eng,
    remove_seam,
    add_seam,
    find_seams,
    get_best_seam,
    reduce_width,
    reduce_height,
    increase_width,
    increase_height,
    intelligent_resize
)


class TestVisualOutput(unittest.TestCase):

  def test_compute_eng_grad(self):
    rgb_weights = [-3, 1, -3]
    mask_weight = 10
    cat_img = misc.imread('./demo/cat.png')
    eng = compute_eng_grad(cat_img)
    misc.imsave('./demo/cat_eng_grad.png', eng)

  def test_compute_eng_color(self):
    rgb_weights = [-3, 1, -3]
    cat_img = misc.imread('./demo/cat.png')
    eng = compute_eng_color(cat_img, rgb_weights)
    misc.imsave('./demo/cat_eng_color.png', eng)

  def test_compute_eng(self):
    rgb_weights = [-3, 1, -3]
    mask_weight = 10
    cat_img = misc.imread('./demo/cat.png')
    mask = np.zeros(cat_img.shape)
    img4 = np.dstack((cat_img, mask))
    eng = compute_eng(img4, rgb_weights, mask_weight)
    misc.imsave('./demo/cat_eng_total.png', eng)
  
  def test_intelligent_resize(self):
    rgb_weights = [-3, 1, -3]
    mask_weight = 10
    cat_img = misc.imread('./demo/cat.png')
    mask = np.zeros(cat_img.shape)

    resized_img = intelligent_resize(cat_img, 20, 0, rgb_weights, mask, mask_weight)
    misc.imsave('./demo/cat_shrunk.png', resized_img[:,:,0:3])

    resized_img = intelligent_resize(cat_img, -150, 0, rgb_weights, mask, mask_weight)
    misc.imsave('./demo/cat_grown.png', resized_img[:,:,0:3])

    castle_img = misc.imread('./demo/castle_small.jpg')
    castle_mask = np.zeros(castle_img.shape)
    resized_img = intelligent_resize(castle_img, 150, 0, [0,0,0], castle_mask, mask_weight)
    misc.imsave('./demo/castle_small_shrunk.png', resized_img[:,:,0:3])


if __name__ == '__main__':
    unittest.main()

# if __name__ == '__main__':
#   rgb_weights = [-3, 1, -3]
#   mask_weight = 10
#   cat_img = misc.imread('./demo/cat.png')
#   plt.imshow(cat_img)
#   plt.show()

  # resized_img = intelligent_resize(img, 1, 0, rgb_weights, mask, mask_weight)