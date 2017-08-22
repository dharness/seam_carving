import unittest
from seam_carver import apply_filter
import numpy as np


class TestSeamCarver(unittest.TestCase):

    def test_apply_filter(self):
        img = [
            [101, 244, 231, 126, 249],
            [151, 249, 219, 9,  64],
            [88, 93, 21, 112, 155],
            [114, 55, 55, 120, 205],
            [84, 154, 24, 252, 63]
        ]
        img_filter = [
            [-1, 0, 1],
            [-3, 0, 3],
            [-1, 0, 1],
        ]
        out_img = apply_filter(img, img_filter)
        expected_img = [
            [981, 458, -594, -101, -387],
            [1084, 267, -819, -313, -265],
            [583, -192, -118, 397, -465],
            [412, -304, 312, 623, -724],
            [517, -239, 359, 267, -876],
        ]

        self.assertEqual(out_img.tolist(), expected_img)
        self.assertEqual(np.sum(np.abs(out_img)), 11657)

if __name__ == '__main__':
    unittest.main()