import pydicom
import numpy as np
from enum import Enum
import cv2

class LOAD_MODE(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


def im_reader(path) :
    if ".dcm" in path :
       im = pydicom.read_file(path).pixel_array 
    else :
        im = cv2.imread(path) 
    im = im/(np.max(im)+1e-5)
    return im