import numpy as np
from numpy import asarray
from PIL import Image
import cv2 as cv

im = cv.imread("./images/dog.bmp", cv.IMREAD_GRAYSCALE)
imColor = cv.imread("./images/dog.bmp")


img_in = Image.open('./images/dog.bmp')
print(im.shape)
array = np.array(img_in)

padded_array = np.empty([600, 700, 3], dtype=np.uint8)
padded_array[:, :] = np.array([0, 64, 128])
padded_array[50:361, 80:410] = array

img_out = Image.fromarray(padded_array)
img_out.save('padded-boat.jpg')


def box_filter_manual(image, k):
    """Apply a k x k box (mean) filter to a grayscale image, by hand.

    image: 2D NumPy array (uint8)
    k:     kernel size, odd (3 or 5)
    returns: 2D array, same shape as image
    """
    r = k // 2  # how many pixels the kernel reaches out from its center

    # STEP 1: Pad the image so edge pixels also have a full k x k neighborhood.
    #         Look up np.pad and choose a mode. (Discussion material!)

    
    
    # STEP 2: Create an empty output array the same shape as the ORIGINAL image.
    #         Think: which dtype? What happens if you store 84.9 in uint8?

    # STEP 3: Loop over every row y and column x of the ORIGINAL image.
    #         For each one:
    #           - slice the k x k window out of the PADDED image
    #           - compute its average (sum, then divide by... what?)
    #           - store it in the output at [y, x]

    # STEP 4: Return the output.

    raise NotImplementedError("Your turn!")
