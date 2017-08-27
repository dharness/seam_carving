<p>
  <img src="https://circleci.com/gh/dharness/seam_carving.svg?&style=shield">
</p>

# seam_carver
Seam Carving as a Service

seam_carver is a small tool for retargetting images to any dimension greater or smaller. It uses the process of seam carving as originally described by Shai Avidan & Ariel Shamir in http://perso.crans.org/frenoy/matlab2012/seamcarving.pdf

A combination of the gradient energy (determined with the sobel filter) and a simple color energy is used to determine the least important seam. Additon of seams occurs by the same mechanism as described in the paper.

### Installation

```
pip install seam_carver
```

### Basic Usage
``` python
from scipy import misc
import numpy as np
from seam_carver import intelligent_resize

rgb_weights = [-3, 1, -3]
mask_weight = 10
cat_img = misc.imread('./demo/cat.png')
mask = np.zeros(cat_img.shape)

resized_img = intelligent_resize(cat_img, 0, -20, rgb_weights, mask, mask_weight)
misc.imsave('./demo/cat_shrunk.png', resized_img)
```

### Options

``` python
def intelligent_resize(img, d_rows, d_columns, rgb_weights, mask, mask_weight):
    """
    Changes the size of the provided image in either the vertical or horizontal direction,
    by increasing or decreasing or some combination of the two.

    Args:
        img (n,m,3 numpy matrix): RGB image to be resized.
        d_rows (int): The change (delta) in rows. Positive number indicated insertions, negative is removal.
        d_columns (int): The change (delta) in columns. Positive number indicated insertions, negative is removal.
        rgb_weights (1,3 numpy matrix): Additional weight paramater to be applied to pixels.
        mask (n,m,3 numpy matrix): Mask matrix indicating areas to make more or less likely for removal.
        mask_weight (int): Scalar multiple to be applied to mask.

    Returns:
        n,m,3 numpy matrix: Resized RGB image.
    """
```

### Examples

### Limitations

![Alt text](/demo/cat.png?raw=true "Optional Title")