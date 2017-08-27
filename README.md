<p>
  <img src="https://circleci.com/gh/dharness/seam_carving.svg?&style=shield">
</p>

# seam_carver
Seam Carving as a Service

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
misc.imsave('./demo/cat_shrunk.png', resized_img[:,:,0:3])
```

### Options

### Examples

### Limitations

![Alt text](/demo/cat.png?raw=true "Optional Title")