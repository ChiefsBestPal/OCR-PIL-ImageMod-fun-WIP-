import os,sys
from PIL import Image
import functools
import time
import math
import threading
import numpy as np
from typing import NamedTuple,List,Container
def ConvRGB(x):
    """Convert #[AA]RRGGBB color in integer or string to (r,g,b) tuple
    
    Alpha (AA) component is simply ignored.
    
    rgb(0xff0000ff)
    >>> (0, 0, 255)
    rgb('#0xff0000')
    >>> (255, 0, 0)
    """
    
    if isinstance(x, str) and x[0] == '#':
        x = int(x[1:], 16)
    return ((x >> 16) & 0xff, (x >> 8) & 0xff, (x) & 0xff)
#*REMINDER; When alpha is changed, RGBA = (...,...,...,TRANSPARENCY_CHANGED)
#! Use previous successful personal project with this module as ref
#! : My Joachim Bot/.../minecraft_faces.py
import re
isFewerColorsVersion = False#!True initially

FewerColorsTag = r"[LESSCOLORS]" if isFewerColorsVersion else ""
#// wall iphone6 picture tests: 25173 -> 5605  makes good acceptable dif 350% less nuances


dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = str(dir_path).replace("\\","/")
TEST_IMAGE = r"[CIEDE2000][OP;7][TRANS;15]test_wallLIGHTGREY.jpg"#! Intial AngledPanoramaCut
ImagePath = dir_path + f"/{FewerColorsTag}{TEST_IMAGE}"


print(ImagePath,end=" <=== This will be the absolute path of the image manipulated\n\n")
picture = Image.open(ImagePath)

# picture = picture.convert("RGB")

# pixData = picture.load()
width, height = picture.size

def _sqrt(x):
    return x**0.5
def _square(x):
    return x * x
def rgb2lab(R_G_B):
    """Convert RGB colorspace to Lab
    
    Adapted from http://www.easyrgb.com/index.php?X=MATH.
    """
    
    R, G, B = R_G_B
    
    # Convert RGB to XYZ
    
    var_R = ( R / 255.0 )        # R from 0 to 255
    var_G = ( G / 255.0 )        # G from 0 to 255
    var_B = ( B / 255.0 )        # B from 0 to 255

    if ( var_R > 0.04045 ): var_R = ( ( var_R + 0.055 ) / 1.055 ) ** 2.4
    else:                   var_R = var_R / 12.92
    if ( var_G > 0.04045 ): var_G = ( ( var_G + 0.055 ) / 1.055 ) ** 2.4
    else:                   var_G = var_G / 12.92
    if ( var_B > 0.04045 ): var_B = ( ( var_B + 0.055 ) / 1.055 ) ** 2.4
    else:                   var_B = var_B / 12.92

    var_R = var_R * 100.0
    var_G = var_G * 100.0
    var_B = var_B * 100.0

    # Observer. = 2°, Illuminant = D65
    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505
    
    # Convert XYZ to L*a*b*
    
    var_X = X / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
    var_Y = Y / 100.000        # ref_Y = 100.000
    var_Z = Z / 108.883        # ref_Z = 108.883

    if ( var_X > 0.008856 ): var_X = var_X ** ( 1.0/3.0 )
    else:                    var_X = ( 7.787 * var_X ) + ( 16.0 / 116.0 )
    if ( var_Y > 0.008856 ): var_Y = var_Y ** ( 1.0/3.0 )
    else:                    var_Y = ( 7.787 * var_Y ) + ( 16.0 / 116.0 )
    if ( var_Z > 0.008856 ): var_Z = var_Z ** ( 1.0/3.0 )
    else:                    var_Z = ( 7.787 * var_Z ) + ( 16.0 / 116.0 )

    CIE_L = ( 116.0 * var_Y ) - 16.0
    CIE_a = 500.0 * ( var_X - var_Y )
    CIE_b = 200.0 * ( var_Y - var_Z )
    return (CIE_L, CIE_a, CIE_b)

def cie94(RGB1, RGB2):
    """Calculate color difference by using CIE94 formulae
    
    See http://en.wikipedia.org/wiki/Color_difference or
    http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CIE94.html.
    
    cie94(rgb2lab((255, 255, 255)), rgb2lab((0, 0, 0)))
    >>> 58.0
    cie94(rgb2lab(rgb(0xff0000)), rgb2lab(rgb('#0xff0000')))
    >>> 0.0
    """
    L1_a1_b1 = rgb2lab(RGB1)
    L2_a2_b2 = rgb2lab(RGB2)
    L1, a1, b1 = L1_a1_b1
    L2, a2, b2 = L2_a2_b2

    C1 = _sqrt(_square(a1) + _square(b1))
    C2 = _sqrt(_square(a2) + _square(b2))
    delta_L = L1 - L2
    delta_C = C1 - C2
    delta_a = a1 - a2
    delta_b = b1 - b2
    delta_H_square = _square(delta_a) + _square(delta_b) - _square(delta_C)
    return (_sqrt(_square(delta_L)
            + _square(delta_C) / _square(1.0 + 0.045 * C1)
            + delta_H_square / _square(1.0 + 0.015 * C1)))

def compare_color_simple(RGB1,RGB2):
    """>>> Can add more colors than the basic ones for higher precision 
    ref: https://en.wikipedia.org/wiki/Color_difference
    """
    r1, g1, b1 = RGB1  
    r2, g2, b2 = RGB2
    #! Should perhaps use redMEAN difference if more colors to database
    return ((r2 - r1) ** 2 + (g2 - g1) ** 2 + (b2 - b1) ** 2) ** 0.5

def CIEDE2000(RGB1, RGB2):
    '''Calculates CIEDE2000 color distance between two CIE L*a*b* colors'''
    C_25_7 = 6103515625 # 25**7
    
    L1_a1_b1 = rgb2lab(RGB1)
    L2_a2_b2 = rgb2lab(RGB2)
    L1, a1, b1 = L1_a1_b1
    L2, a2, b2 = L2_a2_b2

    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_ave = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(C_ave**7 / (C_ave**7 + C_25_7)))
    
    L1_, L2_ = L1, L2
    a1_, a2_ = (1 + G) * a1, (1 + G) * a2
    b1_, b2_ = b1, b2
    
    C1_ = math.sqrt(a1_**2 + b1_**2)
    C2_ = math.sqrt(a2_**2 + b2_**2)
    
    if b1_ == 0 and a1_ == 0: h1_ = 0
    elif a1_ >= 0: h1_ = math.atan2(b1_, a1_)
    else: h1_ = math.atan2(b1_, a1_) + 2 * math.pi
    
    if b2_ == 0 and a2_ == 0: h2_ = 0
    elif a2_ >= 0: h2_ = math.atan2(b2_, a2_)
    else: h2_ = math.atan2(b2_, a2_) + 2 * math.pi

    dL_ = L2_ - L1_
    dC_ = C2_ - C1_    
    dh_ = h2_ - h1_
    if C1_ * C2_ == 0: dh_ = 0
    elif dh_ > math.pi: dh_ -= 2 * math.pi
    elif dh_ < -math.pi: dh_ += 2 * math.pi        
    dH_ = 2 * math.sqrt(C1_ * C2_) * math.sin(dh_ / 2)
    
    L_ave = (L1_ + L2_) / 2
    C_ave = (C1_ + C2_) / 2
    
    _dh = abs(h1_ - h2_)
    _sh = h1_ + h2_
    C1C2 = C1_ * C2_
    
    if _dh <= math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2
    elif _dh  > math.pi and _sh < 2 * math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2 + math.pi
    elif _dh  > math.pi and _sh >= 2 * math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2 - math.pi 
    else: h_ave = h1_ + h2_
    
    T = 1 - 0.17 * math.cos(h_ave - math.pi / 6) + 0.24 * math.cos(2 * h_ave) + 0.32 * math.cos(3 * h_ave + math.pi / 30) - 0.2 * math.cos(4 * h_ave - 63 * math.pi / 180)
    
    h_ave_deg = h_ave * 180 / math.pi
    if h_ave_deg < 0: h_ave_deg += 360
    elif h_ave_deg > 360: h_ave_deg -= 360
    dTheta = 30 * math.exp(-(((h_ave_deg - 275) / 25)**2))
    
    R_C = 2 * math.sqrt(C_ave**7 / (C_ave**7 + C_25_7))  
    S_C = 1 + 0.045 * C_ave
    S_H = 1 + 0.015 * C_ave * T
    
    Lm50s = (L_ave - 50)**2
    S_L = 1 + 0.015 * Lm50s / math.sqrt(20 + Lm50s)
    R_T = -math.sin(dTheta * math.pi / 90) * R_C

    k_L, k_C, k_H = 1, 1, 1
    
    f_L = dL_ / k_L / S_L
    f_C = dC_ / k_C / S_C
    f_H = dH_ / k_H / S_H
    
    dE_00 = math.sqrt(f_L**2 + f_C**2 + f_H**2 + R_T * f_C * f_H)
    return dE_00

def ignore_unhashable(func): 
    uncached = func.__wrapped__
    attributes = functools.WRAPPER_ASSIGNMENTS + ('cache_info', 'cache_clear')
    @functools.wraps(func, assigned=attributes) 
    def wrapper(*args, **kwargs): 
        try: 
            return func(*args, **kwargs) 
        except TypeError as error: 
            if 'unhashable type' in str(error): 
                return uncached(*args, **kwargs) 
            raise 
    wrapper.__uncached__ = uncached
    return wrapper

def fixUnashable(item):
    """ Unhashable inputs will have their unique pointer address
    used as a valid hash instead !!! """
    
    try:
        hash(item)
    except TypeError as Unhashable:
        return id(item)
    else:
        return item
class ThreadForAnimation(object):
    """ Threading example class
    'The run() method will be started and it will run in the background
    until the application exits.'
    """

    def __init__(self,animation:callable=None,interval:int=0.5):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.animation = animation
        self.interval = interval

        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution

    def run(self):
        """ Method that runs forever """
        _animation = self.animation
        while True:
            _animation()
            time.sleep(self.interval)

def spinner(deciseconds:int=30):
    def spinning_cursor():
        while True:
            for cursor in iter('|/-\\'):
                yield cursor
    _spinner = spinning_cursor()
    for _ in range(deciseconds):
        print("\u001b[4m\u001b[44m\u001b[100D",end="")
        sys.stdout.write(f"{next(_spinner)} \u001b[0m")
        sys.stdout.flush()
        time.sleep(0.1)
        sys.stdout.write('\b')
        print("\u001b[0m",end="")

#!=------------------
sys.stdout.write("\u001b[1000D")
spinner1 = ThreadForAnimation(spinner,0.01)

COLOR_DISTANCE_ALGS = [cie94,compare_color_simple,CIEDE2000]
_input = str(list(map(lambda func: func.__name__,COLOR_DISTANCE_ALGS)))#!
alg_ix = int(input("\tChoose a color distance alg from:\t"+_input))
assert 0 <= alg_ix <= _input.count(",")
def C_DISTANCE_ALG(RGB1,RGB2): return COLOR_DISTANCE_ALGS[alg_ix](RGB1,RGB2)

OPACITY_TOLERANCE = int(input("\nEnter OPACITY Tolerance, between 0% (None) to 100%(twice the upper bound approximation)"))
assert 0 <= OPACITY_TOLERANCE <= 100
OPACITY_TOLERANCE /= 100.0
TRANSPARENCY_TOLERANCE = int(input("\nEnter TRANSPARENCY Tolerance, between 0% (None) to 100%(twice the lower bound approximation)"))
assert 0 <= TRANSPARENCY_TOLERANCE <= 100
TRANSPARENCY_TOLERANCE /= 100.0
#!=------------------
SIMPLE_REPLACEMENT_COLOR = (0,0,0)
#!-------------------
#//@ignore_unhashable
@functools.lru_cache(maxsize=None,typed=False) 
def getAverageRGBandDISTANCE(AreaOfPicture,ImageUsed=picture):#TODO CUSTOM COLOR DISTANCE ALG
    """ INPUT (default) (X1,X2,Y1,Y2)
        >>> OUTPUT Tuple(AverageRGB,AverageDistance) """
    COUNT = int()
    assert type(AreaOfPicture)
    X1,X2,Y1,Y2 = AreaOfPicture
    assert Y2 == max(Y1,Y2)
    assert X2 == max(X1,X2)
    #?https://yangcha.github.io/iview/iview.html for pixel index
    
    width, height = ImageUsed.size

    #selected_unique_pixels = set()
    selected_raw_pixels = list()

    distances_in_area = list()
    for x in range(width):
        for y in range(height):
            if (X1 <= x <= X2) and (Y1 <= y <= Y2):
                current_color = ImageUsed.getpixel( (x,y) )
                
                selected_raw_pixels.append(current_color)
                COUNT+= 1
            
        
    r,g,b = 0,0,0
    for tup in selected_raw_pixels:
        if not (tup == SIMPLE_REPLACEMENT_COLOR):
            _r,_g,_b =  tup
            r += _r
            g += _g
            b += _b
    simple_average_RGB = (map(lambda channel: channel//COUNT, [r,g,b])) #TODO USE GEOMETRIC MEAN SUCH THAT (x1*x2*...*xN)**(1/N) instead of (x1+x2+...+xN)/N
    r,g,b = simple_average_RGB
    simple_average_RGB = (r,g,b)

    for x in range(width):
        for y in range(height):
            if (X1 <= x <= X2) and (Y1 <= y <= Y2):
                current_color = ImageUsed.getpixel( (x,y) )#getdata() instead?
                distances_in_area.append(C_DISTANCE_ALG(current_color,simple_average_RGB))

    simple_average_distance = sum(distances_in_area)//len(distances_in_area)



    return (simple_average_RGB,simple_average_distance)
# print(*simple_average_RGB,sep=",")
# print(simple_average_distance,end=" % average color distance in selected area\n")
#!=------------------
class AreaStruct(NamedTuple):
    x1 : int
    x2 : int
    y1 : int
    y2 : int
    #median rgb
    #average rgb
    #no_distnace_abberations rgb
    #nuance approx rgba
    #SurfaceCovered
    #TODO use __slots__

AREAS_LIGHT = ((19,204,0,30),(195,204,0,89))
AREAS_DARK = ((0,95,0,29),) #! Replace Image my pixelMatrix
#!=------------------
#? UNIT TESTS, OPTIMISATION------------------------------------------------------
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


def UNIT_TESTS_FOR_MEMOIZATION_CACHE():
    @timeit
    def preProcessTest(custom_areas=list()):
        global AREAS_LIGHT
        # TEST_MEMOIZATION_TIME,COUNT = float(),int()
        for area in AREAS_LIGHT: # ?memoized, so processing area information once
            # start = time.time() 
            getAverageRGBandDISTANCE(area)
            # end = time.time()
            # TEST_MEMOIZATION_TIME += float(end-start)
            # COUNT += 1 
            #print(f"\rFIRST done in {end-start} seconds",flush=True,end="")
        # print(f"Average First processing time: {TEST_MEMOIZATION_TIME/COUNT}")
        #print("\b")

    preProcessTest()

    preProcessTest()

    preProcessTest()

@timeit
def printAreaInfo(*Areas):
    areaInfo = list()
    for AreaNumber,Area in enumerate(Areas):
        #// start = time.time() 
        #// RGBarea,DISTarea = getAverageRGBandDISTANCE(Area)
        #// end = time.time()
        #! BECAUSE *UNPACKING IS TOO SLOW 
        start = time.time() 
        areaInfo = getAverageRGBandDISTANCE(Area) 
        RGBarea,DISTarea = areaInfo[0],areaInfo[1]
        end = time.time()
        print(f"loop #{AreaNumber+1} processing time: {end-start}")
        print(f"AREA {AreaNumber+1}:",end="\r\n\t")
        print(*RGBarea,sep=",")
        print(DISTarea,end=" % average color distance in selected area\n\n")



printAreaInfo(*AREAS_LIGHT)

while input("Press Enter to start AntEnder color magic")!="":
    pass

sys.stdout.flush()
sys.stdout.write("\r\n")
def isPixelInArea(coords,*Areas)->bool:
    """INPUT coords:tuple[x int,y int],...Areas """

    for area in Areas:
        x1,x2,y1,y2 = area
        assert y2 == max(y1,y2)
        assert x2 == max(x1,x2)
        if (x1 <= coords[0] <= x2) and (y1 <= coords[1] <= y2):#any
            return True
    return False

def isPixelAreaMatch(rgb,*Areas)->bool:
    """ INPUT rgb:tuple[R byte,G byte,B byte],... Areas"""
    #? global TRANSPARENCY_TOLERANCE,OPACITY_TOLERANCE
    for area in Areas:
        averageRGBarea,averageDISTarea = getAverageRGBandDISTANCE(area)

        pixelColorDistance = C_DISTANCE_ALG(rgb,averageRGBarea)

        if (pixelColorDistance - pixelColorDistance*TRANSPARENCY_TOLERANCE) <= (averageDISTarea + averageDISTarea*OPACITY_TOLERANCE): #! NICE, TEST THIS SOON
            return True
    return False


def percentageDone(px_x,px_y,ImageUsed=picture,H=height,W=width)->float:
    """ Given pixel coordinates, find the percentage of image processing that is complete""" 
    return ((px_y * W + px_x)/(W*H - 1))*100.0

DELIM = u"\u001b[7m║║\u001b[0m"
def myProgressBar(percent)->None:
    sys.stdout.write(f"\u001b[200D\u001b[110C{DELIM}")
    percent += 1
    COLOR = str()
    if percent < 17: COLOR=u"\u001b[31m"
    elif percent < 33: COLOR=u"\u001b[31;1m"
    elif percent < 49: COLOR=u"\u001b[33m"
    elif percent < 67: COLOR=u"\u001b[33;1m"
    elif percent < 84: COLOR=u"\u001b[32m"
    else: COLOR=u"\u001b[32;1m"
    sys.stdout.write("\r{start} {percentage:02.1f}% {steps}➤\u001b[0m "
    .format(start=DELIM,percentage=percent,steps=f"{COLOR}❒"*int(percent)))
    sys.stdout.flush()


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
    return pixel_values,width, height

def simplePIL_get_image(image_path,*,returnImage,returnSize):#->Image.__loader__.PixelAccess:
    img = Image.open(image_path)
    pixel_array = img.load()
    yield pixel_array
    if returnImage: yield img
    if returnSize: yield img.size #__sizeof__    w: [0] h:[1]

def regexCleanPath(stringPath):
    removeImagePathExtension = lambda string: re.sub(r"\.[^.]{3,}$","",string)
    removeImagePathTags = lambda string: re.sub(r"\[.*\]","",string)
    removeImagePathDir = lambda string: re.sub(r"^.+\/(?=[^/]*$)","",string)
    return removeImagePathExtension(removeImagePathTags(removeImagePathDir(stringPath)))


def simple_superimpose_pixel_arrays(image_path1,image_path2,allowed_offset,**colorsRgb):#TODO IMPLEMENT SMART OFFSET TO REPLACE COLORS PROPERLY
    pixel_array1,dimensions= tuple(simplePIL_get_image(image_path1,returnImage=False,returnSize=True))#*get_image(image_path1)
    pixel_array2,outputImage = tuple(simplePIL_get_image(image_path2,returnImage=True,returnSize=False))#*get_image(image_path2)
    RGBvalues = iter(colorsRgb.values())
    #//dark_offset = lambda channel: 0 if (channel - allowed_offset) <= 0 else (channel - allowed_offset)  #!TEMPORARY
    #//light_offset = lambda channel: 255 if (channel + allowed_offset) >= 255 else (channel + allowed_offset) #!TEMPORARY
    try:
        while bool(colorsRgb):
            colorRGB = next(RGBvalues)
            for y in range(161):
                for x in range(266):

                    if tuple(pixel_array1[x,y]) == colorRGB:
                        pixel_array2[x,y] = colorRGB

    except StopIteration as noMoreRGBtoSuperimpose:
        #_im = Image.fromarray(pixel_array2) #* 0xFF).astype(np.uint8))#use uint because numpy

        image_path1,image_path2 = regexCleanPath(image_path1),regexCleanPath(image_path2)
        outputImage.save(f"{dir_path}/{image_path1}X{image_path2}Superimposed.jpg")
    else:
        raise Exception("Error occured, could not superimpose")


#TODO  
#TODO MAKE A DATABASE OF IMAGES/PIXEL ARRAYS THAT ARE KNOWN AND MEMOIZED (json) !!!
#TODO  
def main():
    for AREAS_TESTED in [AREAS_DARK]:#!missing AREAS_LIGHT
        for y in range(height):
            for x in range(width):
                current_color = current_color = picture.getpixel( (x,y) )
                # print(f"\r{(x,y)}",flush=True,end="")
                if isPixelAreaMatch(current_color,*AREAS_TESTED):
                    picture.putpixel((x,y), SIMPLE_REPLACEMENT_COLOR) #putdata() ?
                percent_completed = percentageDone(x,y)
                #? print("\r({:03d},{:03d}); \t {:02.1f}% done"
                #? .format(x,y,percent_completed,flush=True,end="\n"))
                myProgressBar(percent_completed)

    sys.stdout.write("\u001b[2K\u001b[4m\u001b[44m\u001b[100D ▂ ▄ ▅ ▆ ▇ █ 𝐈𝐌𝐀𝐆𝐄 𝐏𝐑𝐎𝐂𝐄𝐒𝐒𝐄𝐃 █ ▇ ▆ ▅ ▄ ▂ \u001b[0m\t\n\n")

#
if __name__ == "__main__":
    main()
    #picture.save(f"{dir_path}/[{COLOR_DISTANCE_ALGS[alg_ix].__name__}][OP;{round(OPACITY_TOLERANCE*100)}][TRANS;{round(TRANSPARENCY_TOLERANCE*100)}]test2_wall.jpg")
    
    image_path1 = dir_path + r"/[CIEDE2000][OP;1][TRANS;2]test_wallDARKGREY.jpg"
    image_path2 = dir_path + r"/[CIEDE2000][OP;7][TRANS;15]test_wallLIGHTGREY.jpg"
    simple_superimpose_pixel_arrays(image_path1,image_path2,allowed_offset=None,black=(0,0,0))





#!!! OP PYTHON OOP !!!!!!!!  Use adjacency matrices to find similar pixels ?
# class UndirectedGraphMetaClass(type):
#     def __init__(cls,name,bases,dict):
#         graph_value:list[list[int,int]] = []
#         setattr(cls,"undirected_graph",graph_value)
        
#         meta_class:cls = UndirectedGraphMetaClass
#         super(meta_class,cls).__init__(name,bases,dict)

# class UndirectedGraph(metaclass=UndirectedGraphMetaClass):
#     __metaclass__ = UndirectedGraphMetaClass
#     isWeighted = False


 
# MAX = 500
# N = 3
# M = 4

# class AdjNode:
#     def __init__(self, data):
#         self.vertex = data
#         self.next = None
#     def __repr__(self) -> str:
#         return f"({self.vertex}) -> {self.next}\n"
 

# class GraphAdjencyMatrix(object):
#     @staticmethod
#     def GRAPH_BUFFER_SIZE() -> Literal : return 100

#     def init_cell_counter():

#         count = 0
#         def inner():

#             nonlocal count
#             count += 1
#             return count

#         return inner
#     point_next_vertex = init_cell_counter()

#     # edge = TypeVarTuple('edge')
#     # vertex = TypeVar('vertex')
#     def __init__(self, rows,cols):
#         self.graph:list[list[int,int],list[int]] = [[None] for __vertex in range(rows*cols + 1 + GraphAdjencyMatrix.GRAPH_BUFFER_SIZE())]
#         self._current_vertex = 0
#         self.m,self.n = rows,cols
    
#     @property
#     def current_vertex(self):
#         GraphAdjencyMatrix.point_next_vertex()
#         self._current_vertex = GraphAdjencyMatrix.point_next_vertex.__closure__[0].cell_contents
#         return self._current_vertex
#     @classmethod
#     def useVertexValueInsteadOfIndex(cls):
#         pass#getattr(...)

#     def make_directed_graph(self):#using indexes: a cell will always point down, up,both,or none
#         vertex_ix = int()
#         m,n = self.m,self.n

#         for i in range(0,m):
#             for j in range(0,n):

#                 # If last row, then add edge on right side.
#                 if (i == m):
                    
#                     # If not bottom right cell.
#                     if (j != n):
#                         self.graph[vertex_ix].append(vertex_ix + 1)
#                         self.graph[vertex_ix + 1].append(vertex_ix)
    
#                 # If last column, then add edge toward down.
#                 elif (j == m):
#                     self.graph[vertex_ix].append(vertex_ix + n)
#                     self.graph[vertex_ix + n].append(vertex_ix)
                    
#                 # Else makes an edge DOWN and RIGHT directions.
#                 else:
#                     self.graph[vertex_ix].append(vertex_ix + 1)
#                     self.graph[vertex_ix + 1].append(vertex_ix)
#                     self.graph[vertex_ix].append(vertex_ix + n)
#                     self.graph[vertex_ix + n].append(vertex_ix)
            
#                 vertex_ix = self.current_vertex
#                 print(vertex_ix)