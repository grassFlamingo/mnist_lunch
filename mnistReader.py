import numpy as np

from ctypes import cdll

C_mnst_helper = cdll.LoadLibrary("./mnist_helper.so")

G_train_images = "MNIST/train-images.idx3-ubyte"
G_train_labels = "MNIST/train-labels.idx1-ubyte"
G_task_images = "MNIST/t10k-images.idx3-ubyte"
G_task_labels = "MNIST/t10k-labels.idx1-ubyte"

def read_train_data():
    """
    read the mnist train images and labels
    - return : a tupe (I, L)
        I: a numpy array of images, shape (numOfImage, width, height)
        L: a numpy array of lables, shape (numOfLabels,) all the lables are in[0...9]
    this function will take about 8s
    """
    return (read_train_images(), read_train_labels())

def read_task_data():
    """
    read the mnist task image and labels
    - return: a tupe (I,L)
        I: a numpy array of images, shape (numOfImages, width, height)
        L: a numpy array of labels, shape (numOfLables,) all the lables are in[0...9]
    this function will take 2s
    """
    return (read_task_images(), read_task_labels())

# ###
# [offset] [type]          [value]          [description] 
# 0000     32 bit integer  0x00000803(2051) magic number 
# 0004     32 bit integer  60000            number of images 
# 0008     32 bit integer  28               number of rows 
# 0012     32 bit integer  28               number of columns 
# 0016     unsigned byte   ??               pixel 
# 0017     unsigned byte   ??               pixel 
# ........ 
# xxxx     unsigned byte   ??               pixel
# Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
# ###
def read_train_images():
    """
    read train image only
    return numpy array which has the shape of (60000, 28,28)
    This function will need 7.524s.
    """
    with open(G_train_images, 'rb') as fImg:
        dataShape = _read_images_file_header(fImg, 0x00000803, 60000)
        allImgs = fImg.read()
        allImgs = np.array(list(allImgs))
        allImgs.shape = dataShape
        return allImgs

# ###
# TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
# [offset] [type]          [value]          [description] 
# 0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
# 0004     32 bit integer  60000            number of items 
# 0008     unsigned byte   ??               label 
# 0009     unsigned byte   ??               label 
# ........ 
# xxxx     unsigned byte   ??               label
# The labels values are 0 to 9.
# ###
def read_train_labels():
    """
    read the train lables
    return numpy array which has the shape of (60000, )
    This function will need 0.331s.
    """
    with open(G_train_labels, 'rb') as fLabel:
        _read_labels_file_header(fLabel, 0x00000801, 60000)
        labels = fLabel.read()
        nplabs = np.array(list(labels))
        return nplabs


# ###
# TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
# [offset] [type]          [value]          [description] 
# 0000     32 bit integer  0x00000803(2051) magic number 
# 0004     32 bit integer  10000            number of images 
# 0008     32 bit integer  28               number of rows 
# 0012     32 bit integer  28               number of columns 
# 0016     unsigned byte   ??               pixel 
# 0017     unsigned byte   ??               pixel 
# ........ 
# xxxx     unsigned byte   ??               pixel
# Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black). 
# ###
def read_task_images():
    """
    read the task images
    return numpy array shape of (10000, 28, 28)
    the function will take 1.538s
    """
    with open(G_task_images, 'rb') as fImg:
        dataShape = _read_images_file_header(fImg, 0x00000803, 10000)
        datas = fImg.read()
        npData = np.array(list(datas))
        npData.shape = dataShape
        return npData

# ###
# TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
# [offset] [type]          [value]          [description] 
# 0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
# 0004     32 bit integer  10000            number of items 
# 0008     unsigned byte   ??               label 
# 0009     unsigned byte   ??               label 
# ........ 
# xxxx     unsigned byte   ??               label
# The labels values are 0 to 9.
# ###
def read_task_labels():
    """
    read task lables
    the function will take 0.304s
    return numpy array with the shape of (10000,)
    """
    with open(G_task_labels, 'rb') as fLabel:
        _read_labels_file_header(fLabel, 0x00000801, 10000)
        datas = fLabel.read()
        npData = np.array(list(datas))
        return npData


def _read_images_file_header(fImg, mNumber, nImages):
    """
    - fImg: The opened file reader
    - mNumber: The magic number
    - nImages: The number of images
    check if the header information is at right side
    return (numOfImages, numOfCols, numOfRows)
    """
    header = fImg.read(16)
    magicNumber = C_mnst_helper.pybytes_to_int32(header,0)
    numOfImages = C_mnst_helper.pybytes_to_int32(header,4)
    numOfRows = C_mnst_helper.pybytes_to_int32(header,8)
    numOfCols = C_mnst_helper.pybytes_to_int32(header,12)
    assert magicNumber == mNumber, "magicNumber is not correct, now is %d"%(magicNumber)
    assert numOfImages == nImages, "numOfImages is not correct, now is %d"%(numOfImages)
    assert numOfRows == 28, "numOfRows is not correct, now is %d"%(numOfRows)
    assert numOfCols == 28, "numOfCols is not correct, now is %d"%(numOfCols)
    return (numOfImages, numOfCols, numOfRows)

def _read_labels_file_header(fLabel, mNumber, nLables):
    """
    - fLabel: The opened file reader
    - mNumber: The magic number
    - nLables: The number of lables
    check if the header information is right
    return (numOfLables,)
    """
    header = fLabel.read(8)
    magicNumber = C_mnst_helper.pybytes_to_int32(header,0)
    numOfLables = C_mnst_helper.pybytes_to_int32(header,4)
    assert magicNumber == mNumber, "magicNumber (%d) not correct"%(magicNumber)
    assert numOfLables == nLables, "numOfLables (%d) not correct"%(numOfLables)
    return (numOfLables,)


# tm = read_train_image()
# print(tm)
# print(tm.shape)
# print("\n")
# bytess = (0x00000803).to_bytes(4, byteorder= 'big')
# print(bytess)
# readd = read_bytes_to_int32(bytess, 0)
# print(readd, hex(readd))

# tl = read_task_labels()
# print(tl)
# print(tl.shape)