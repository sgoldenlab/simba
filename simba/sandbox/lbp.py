import numpy as np
from skimage import feature
import cv2

def local_binary_patterns(img: np.ndarray, P=1, R=3):
    lbp = feature.local_binary_pattern(img, P=P, R=R, method="default")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P+3), range=(0, P + 2))
    print(hist)
    hist = hist.astype("float")
    hist /= (hist.sum() + 0.001)
    print(hist)

    pass




img = cv2.imread('/Users/simon/Desktop/gresyscale.png', cv2.IMREAD_GRAYSCALE)
local_binary_patterns(img=img, P=4)

