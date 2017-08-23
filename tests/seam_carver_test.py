import unittest
from seam_carver import apply_filter, normalize, compute_eng_grad, remove_seam
import numpy as np
import matplotlib.pyplot as plt


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


    def test_normalize(self):
        img = np.array([
            [-100, 244, 231, 126, 249],
            [151, 249, 219, 9,  64],
            [88, 93, 21, 112, 155],
            [114, 55, 55, 120, 205],
            [84, 154, 24, 252, 500]
        ]).astype(float)
        
        normalized = normalize(img)
        self.assertLessEqual(np.max(normalized), 255)
        self.assertGreaterEqual(np.min(normalized), 0)


    def test_compute_eng_grad(self):
        img = np.zeros((5,5,3))
        
        img[:,:,0] = np.vstack((
            [101, 244, 231, 126, 249],
            [151, 249, 219, 9, 64],
            [88, 93, 21, 112, 155],
            [114, 55, 55, 120, 205],
            [84, 154, 24, 252, 63]
        ))

        img[:,:,1] = np.vstack((
            [115, 228, 195, 68, 102],
            [92, 74, 216, 64, 221],
            [218, 134, 123, 35, 213],
            [229, 23, 192, 111, 147],
            [164, 218,  78, 231, 146]
        ))

        img[:,:,2] = np.vstack((
            [91, 201, 137, 85, 182],
            [225, 102, 91, 122, 60],
            [85,  46, 139, 162, 241],
            [101, 252, 31, 100, 69],
            [158, 198, 196, 26, 239]
        ))
        eng = compute_eng_grad(img)
        self.assertLessEqual(np.max(eng), 255)
        self.assertGreaterEqual(np.min(eng), 0)


    def test_remove_seam(self):
        img4 = np.zeros((5,5,4))
        
        img4[:,:,0] = np.array([
            [101, 244, 231, 126, 249],
            [151, 249, 219, 9, 64],
            [88, 93, 21, 112, 155],
            [114, 55, 55, 120, 205],
            [84, 154, 24, 252, 63]
        ])

        img4[:,:,1] = np.array([
            [115, 228, 195, 68, 102],
            [92, 74, 216, 64, 221],
            [218, 134, 123, 35, 213],
            [229, 23, 192, 111, 147],
            [164, 218,  78, 231, 146]
        ])

        img4[:,:,2] = np.array([
            [91, 201, 137, 85, 182],
            [225, 102, 91, 122, 60],
            [85,  46, 139, 162, 241],
            [101, 252, 31, 100, 69],
            [158, 198, 196, 26, 239]
        ])
        
        img4[:,:,3] = np.array([
            [-1, 1, 0, -1, 0],
            [1, 1, 0, -1, 1],
            [0, 1, 0, -1, -1],
            [1, 1, 0, -1, 1],
            [-1, -1, 0, 1, 0]
        ])
        seam = np.array([[1],[2],[3],[4],[1]])
        remove_seam(img4, seam)


if __name__ == '__main__':
    unittest.main()