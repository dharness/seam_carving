import unittest
import matplotlib.pyplot as plt
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


class TestSeamCarver(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        img4 = np.zeros((5,5,4))
        img4[:,:,0] = np.vstack((
            [101, 244, 231, 126, 249],
            [151, 249, 219, 9, 64],
            [88, 93, 21, 112, 155],
            [114, 55, 55, 120, 205],
            [84, 154, 24, 252, 63]
        ))

        img4[:,:,1] = np.vstack((
            [115, 228, 195, 68, 102],
            [92, 74, 216, 64, 221],
            [218, 134, 123, 35, 213],
            [229, 23, 192, 111, 147],
            [164, 218,  78, 231, 146]
        ))

        img4[:,:,2] = np.vstack((
            [91, 201, 137, 85, 182],
            [225, 102, 91, 122, 60],
            [85,  46, 139, 162, 241],
            [101, 252, 31, 100, 69],
            [158, 198, 196, 26, 239]
        ))

        img4[:,:,3] = np.vstack((
            [-1, 1, 0, -1, 0],
            [1, 1, 0, -1, 1],
            [0, 1, 0, -1, -1],
            [1, 1, 0, -1, 1],
            [-1, -1, 0, 1, 0]
        ))
        self.img4 = img4

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
        img = self.img4[:,:,0:3]
        eng = compute_eng_grad(img)
        self.assertLessEqual(np.max(eng), 255)
        self.assertGreaterEqual(np.min(eng), 0)

    def test_compute_eng_color(self):
        img = self.img4[:,:,0:3]
        rgb_weights = [-3, 1, -3]
        expected_eng = [
            [-461, -1107,  -909,  -565, -1191],
            [-1036, -979, -714, -329, -151],
            [-301, -283, -357, -787, -975],
            [-416, -898,  -66, -549, -675],
            [-562, -838, -582, -603, -760]
        ]
        eng = compute_eng_color(img, rgb_weights)
        self.assertGreaterEqual(eng.tolist(), expected_eng)
    
    def test_compute_eng(self):
        img4 = self.img4
        rgb_weights = [-3, 1, -3]
        mask_weight = 10
        eng = compute_eng(img4, rgb_weights, mask_weight)

    def test_remove_seam(self):
        img4 = self.img4
        seam = np.array([[0],[1],[2],[3],[0]])
        img4_removed = remove_seam(img4, seam)

        self.assertEqual(img4_removed[:,:,0].tolist(),
            [[244, 231, 126, 249],
            [151, 219, 9, 64],
            [88, 93, 112, 155],
            [114, 55, 55, 205],
            [154, 24, 252, 63]]
        )
        
        self.assertEqual(img4_removed[:,:,1].tolist(),
            [[228, 195, 68, 102],
            [92, 216, 64, 221],
            [218, 134, 35, 213],
            [229, 23, 192, 147],
            [218, 78, 231, 146]]
        )
        
        self.assertEqual(img4_removed[:,:,2].tolist(),
            [[201, 137, 85, 182],
            [225, 91, 122, 60],
            [85, 46, 162, 241],
            [101, 252, 31, 69],
            [198, 196, 26, 239]]
        )
        
        self.assertEqual(img4_removed[:,:,3].tolist(),
            [[1, 0, -1, 0],
            [1, 0, -1, 1],
            [0, 1, -1, -1],
            [1, 1, 0, 1],
            [-1, 0, 1, 0]]
        )
    
    def test_add_seam(self):
        img4 = self.img4
        seam = np.array([[0],[1],[2],[3],[0]])
        rgb_weights = [-3, 1, -3]
        mask_weight = 10
        eng = compute_eng(img4, rgb_weights, mask_weight)
        img4_added, _ = add_seam(img4, seam, eng)
        self.assertEqual(img4_added[:,:,0].tolist(),
            [[101., 172.5, 244., 231., 126., 249.],
            [151., 249., 206.33333333333334, 219., 9., 64.],
            [88., 93., 21., 75.33333333333333, 112., 155.],
            [114., 55., 55., 120., 126.66666666666667, 205.],
            [84., 119., 154., 24., 252., 63.]]
        )

        self.assertEqual(img4_added[:,:,1].tolist(),
            [[ 115., 171.5, 228., 195., 68., 102.],
            [  92., 74., 127.33333333333333, 216., 64., 221.],
            [ 218., 134., 123., 97.33333333333333, 35., 213.],
            [ 229., 23., 192., 111., 150., 147.],
            [ 164., 191., 218., 78., 231., 146.]]
        )

        self.assertEqual(img4_added[:,:,2].tolist(),
            [[91., 146., 201., 137., 85., 182.],
            [225., 102., 139.33333333333334, 91., 122., 60.],
            [85., 46., 139., 115.66666666666667, 162., 241.],
            [101., 252., 31., 100., 66.66666666666667, 69.],
            [158., 178., 198., 196., 26., 239.]]
        )

        self.assertEqual(img4_added[:,:,3].tolist(),
            [[-1., 0., 1., 0., -1., 0.],
            [1., 1., 0.6666666666666666, 0., -1., 1.],
            [0., 1., 0., 0., -1., -1.],
            [1., 1., 0., -1., 0., 1.],
            [-1., -1., -1., 0., 1., 0.]]
        )


    def test_find_seams(self):
        eng = np.vstack((
            [577.2127, -474.0578, -211.6035, -183.2227, -471.2210],
            [-382.3976, -653.9937, -384.5538, 12.8625, 287.8903],
            [66.4436, -143.5701, -3.5477, -480.0582, -578.8598],
            [39.5649, -722.9933, 94.4719, -420.0252, -159.3623],
            [309.9671, -422.5076, -206.7120, -157.1606,  111.4617]
        ))
        expected_M = [
            [577.2127, -474.0578, -211.6035, -183.2227, -471.221],
            [-856.4554, -1128.0515, -858.6116, -458.3585, -183.33069999999998],
            [-1061.6079, -1271.6216, -1131.5992, -1338.6698, -1037.2183],
            [-1232.0566999999999, -1994.6149, -1244.1979, -1758.695, -1498.0321],
            [-1684.6478, -2417.1225, -2201.3269, -1915.8555999999999, -1647.2332999999999]
        ]
        expected_P = [
            [-1, -1, -1, -1, -1],
            [1, 1, 1, 4, 4],
            [1, 1, 1, 2, 3],
            [1, 1, 3, 3, 3],
            [1, 1, 1, 3, 3]
        ]
        M, P = find_seams(eng)
        self.assertEqual(M.tolist(), expected_M)
        self.assertEqual(P.tolist(), expected_P)

    def test_get_best_seam(self):
        M = np.vstack((
            [577.2127, -474.0578, -211.6035, -183.2227, -471.221],
            [-856.4554, -1128.0515, -858.6116, -458.3585, -183.33069999999998],
            [-1061.6079, -1271.6216, -1131.5992, -1338.6698, -1037.2183],
            [-1232.0566999999999, -1994.6149, -1244.1979, -1758.695, -1498.0321],
            [-1684.6478, -2417.1225, -2201.3269, -1915.8555999999999, -1647.2332999999999]
        ))
        P = np.vstack((
            [-1, -1, -1, -1, -1],
            [1, 1, 1, 4, 4],
            [1, 1, 1, 2, 3],
            [1, 1, 3, 3, 3],
            [1, 1, 1, 3, 3]
        ))
        seam, cost = get_best_seam(M, P)
        self.assertEqual(seam.tolist(), [
            [1],
            [1],
            [1],
            [1],
            [1]
        ])
        self.assertEqual(cost, -2417.1225)
    
    
    def test_reduce_width(self):
        img4 = self.img4
        rgb_weights = [-3, 1, -3]
        mask_weight = 10
        eng = compute_eng(img4, rgb_weights, mask_weight)
        seam, reduced_img4, cost = reduce_width(img4, eng)
        self.assertEqual(reduced_img4.shape, (5, 4, 4))


    def test_reduce_height(self):
        img4 = self.img4
        rgb_weights = [-3, 1, -3]
        mask_weight = 10
        eng = compute_eng(img4, rgb_weights, mask_weight)
        seam, reduced_img4, cost = reduce_height(img4, eng)
        self.assertEqual(reduced_img4.shape, (4, 5, 4))


    def test_increase_width(self):
        img4 = self.img4
        rgb_weights = [-3, 1, -3]
        mask_weight = 10
        eng = compute_eng(img4, rgb_weights, mask_weight)
        seam, increaseded_img4, cost, updated_eng = increase_width(img4, eng)
        self.assertEqual(increaseded_img4.shape, (5, 6, 4))


    def test_increase_height(self):
        img4 = self.img4
        rgb_weights = [-3, 1, -3]
        mask_weight = 10
        eng = compute_eng(img4, rgb_weights, mask_weight)
        seam, increaseded_img4, cost, updated_eng = increase_height(img4, eng)
        self.assertEqual(increaseded_img4.shape, (6, 5, 4))


    def test_intelligent_resize(self):
        img = self.img4[:,:,0:3]
        mask = self.img4[:,:,3]
        rgb_weights = [-3, 1, -3]
        mask_weight = 10

        resized_img = intelligent_resize(img, 0, -1, rgb_weights, mask, mask_weight)
        self.assertEqual(resized_img.shape, (5, 4, 4))

        resized_img = intelligent_resize(img, -1, 0, rgb_weights, mask, mask_weight)
        self.assertEqual(resized_img.shape, (4, 5, 4))

        resized_img = intelligent_resize(img, -1, -1, rgb_weights, mask, mask_weight)
        self.assertEqual(resized_img.shape, (4, 4, 4))

        resized_img = intelligent_resize(img, 0, 1, rgb_weights, mask, mask_weight)
        self.assertEqual(resized_img.shape, (5, 6, 4))

        resized_img = intelligent_resize(img, 1, 0, rgb_weights, mask, mask_weight)
        self.assertEqual(resized_img.shape, (6, 5, 4))

        resized_img = intelligent_resize(img, 1, 1, rgb_weights, mask, mask_weight)
        self.assertEqual(resized_img.shape, (6, 6, 4))

if __name__ == '__main__':
    unittest.main()