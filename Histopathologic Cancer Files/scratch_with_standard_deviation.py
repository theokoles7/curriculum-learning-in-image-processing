# A version of scratch.py with standard deviations

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pywt

img = np.array(Image.open("sample/0a0a85db9218e366569c913185cc0740f59f4d9e.tif"))
wd = pywt.wavedec2(img, "db2", level=1)
a, (h, v, d) = wd[0], (wd[1][0], wd[1][1], wd[1][2])
print(
    f"Horizontal:\t{np.sum(np.square(h))}\nVertical:\t{np.sum(np.square(v))}\nDiagonal:\t{np.sum(np.square(d))}"
)
