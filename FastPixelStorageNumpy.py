import os,sys
from PIL import Image
import functools
import numpy as np
import time
import cv2

def timeit(method):
    def timed(*args, **kw):#closure with args,and keyword args
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            # kw['log_time'][name] = int((te - ts) * 1000)
            kw['log_time'][name] = int((te - ts))
        else:
            #print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
            print('%r  %.10f seconds ' % (method.__name__, (te - ts) ))

        return result
    return timed
#! Personal learning about closures
# def make_counter():

#     count = 0
#     def inner():

#         nonlocal count
#         count += 1
#         return count

#     return inner
# counter=  make_counter()
# counter()
# counter()
# counter()
# #print(counter())
# def cons(a, b):
#     def pair(f):
#         return f(a, b)
#     return pair
# def car(outer_closure): return outer_closure.__closure__[0].cell_contents
# def cdr(outer_closure): return outer_closure.__closure__[1].cell_contents
# print(car(cons(3,4))) 
# print(cdr(cons(3,4))) 

isFewerColorsVersion = False#!True initially

FewerColorsTag = r"[LESSCOLORS]" if isFewerColorsVersion else ""
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = str(dir_path).replace("\\","/")
TEST_IMAGE = r"[LESSCOLORS]AngledPanoramaCut.jpg"#! Intial AngledPanoramaCut
ImagePath = dir_path + f"/{FewerColorsTag}{TEST_IMAGE}"


def get_image(image_path):
    """Get a numpy array of an image so that one can access values[x][y]."""
    image = Image.open(image_path, "r")
    width, height = image.size
    pixel_values = list(image.getdata())
    if image.mode == "RGB":
        channels = 3
    elif image.mode == "L":
        channels = 1
    else:
        print("Unknown mode: %s" % image.mode)
        return None
    pixel_values = np.array(pixel_values).reshape((width, height, channels))
    return pixel_values

img = cv2.imread(ImagePath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gammaCorrect4(im, gamma=0.5):
    # create a blank output Image
    outImg = np.zeros(im.shape, im.dtype)
    #rows, cols = im.shape

    # create a lookup table
    LUT = []

    for i in range(256):
        LUT.append(((i / 255.0) ** (1 / gamma)) * 255)

    LUT = np.array(LUT,dtype=np.uint8)
    outImg = LUT[im]

    return outImg



out = gammaCorrect4(img,1.5)
cv2.imwrite("testCV2_gamma150.png",out)
