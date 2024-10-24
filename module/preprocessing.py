import numpy as np
from PIL import Image

def resize_image(img_path, size = (160, 160)):
    img = Image.open(img_path)
    img = img.resize(size, 1)
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    return img