
"""
File   : file_operations.py
Brief  : Several useful functions related to file operations.
Date   : 2016-06-06
Version: 1.0
Email  : dli1988@126.com
"""
import Image

def getPathAndKey(path_with_key):
    """
    Parse the input argument into two values by the comma.
    The two values are image path and its key value.

    Args:
        path_with_key: Images path and its key value.

    Returns:
        Image path.
        Key value.
    """
    comma_pos = path_with_key.find(",")
    # The value before comma:
    val1 = path_with_key[0:comma_pos]
    # The value after comma:
    val2 = path_with_key[comma_pos+1:len(path_with_key)]
    # Results
    return val1, val2

def loadImages(path, w, h):
    """
    Load images.

    Args:
        path: Image path.
        w: Image width after resizing.
        h: Image height after resizing.

    Returns:
        Image.
    """
    new_size = (w, h)
    img_src = Image.open(path)
    img_new = img_src.resize(new_size)
    return img_new

def getPatchesSize(img_w, img_h, patches_w, patches_h, stride):
    """
    Calculate the number of patches extracted from one image

    Args:
      img_w: width of image.
      img_h: height of image.
      patches_w: width of patches.
      patches_h: height of patches.
      stride: space interval when extract the patches.

    Return:
      Patches number per image.
    """
    nx = (img_w - patches_w) / stride + 1
    ny = (img_h - patches_h) / stride + 1
    num = nx * ny
    return num

def getBatchSize(patches_size, max_size):
    """
    Calculate batch size.

    Args:
      patches_size: size of patches per image.
      max_size: patches wanted to process per time.

    Return:
      Batch size.
    """
    return max_size / patches_size

def getImgsNameFromFile(file_path):
    """
    Get all the images name from the text file.

    Args:
      file_path: path of the text file include all the images name.

    Return:
      Images name.
    """
    
    files = []
    read_file = open(file_path)
    file_line = read_file.readline()
    while ('' != file_line):
        string = file_line
        filename = string[0:len(string)-1]
        files.append(filename)
        file_line = read_file.readline()
    return files

