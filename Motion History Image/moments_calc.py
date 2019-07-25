import numpy as np
import cv2
import math

def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return np.uint8(image_out)


def bg_subtraction(img1, img2, theta=10):
    """
    get the binary motion signal

    Args:
        img1 (numpy.array): image in t-1
        img2 (numpy.array): image in t
        theta (int): threshold value
    Returns:
        res (numpy.array): binary img of the person moving
    """
    my_img1 = np.copy(img1)
    my_img2 = np.copy(img2)
    my_img1 = cv2.GaussianBlur(my_img1,(7,7),0)
    my_img2 = cv2.GaussianBlur(my_img2,(7,7),0)
    my_img1 = cv2.cvtColor(my_img1, cv2.COLOR_BGR2GRAY)
    my_img2 = cv2.cvtColor(my_img2, cv2.COLOR_BGR2GRAY)
    res = np.zeros(my_img2.shape)
    res[cv2.absdiff(my_img1,my_img2) >= theta] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    #res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
    return res


def motion_img(img1, img2, tau):
    res = np.zeros(img2.shape)
    res[img2 == 1] = tau
    res[img2 != 1] = np.maximum(img1 - 1, 0)[img2 != 1]
    return res


def get_hu_moment(img):
    h, w = img.shape
    x_ind = np.repeat(np.arange(w), h).reshape(w, h).T
    y_ind = np.repeat(np.arange(h), w).reshape(h, w)
    he20 = get_scaled_moment(img, x_ind, y_ind, 2, 0)
    he02 = get_scaled_moment(img, x_ind, y_ind, 0, 2)
    he11 = get_scaled_moment(img, x_ind, y_ind, 1, 1)
    he30 = get_scaled_moment(img, x_ind, y_ind, 3, 0)
    he12 = get_scaled_moment(img, x_ind, y_ind, 1, 2)
    he21 = get_scaled_moment(img, x_ind, y_ind, 2, 1)
    he03 = get_scaled_moment(img, x_ind, y_ind, 0, 3)
    he22 = get_scaled_moment(img, x_ind, y_ind, 2, 2)

    h1 = he20 + he02
    h2 = (he20-he02)**2 + 4*he11**2
    h3 = (he30-3*he12)**2 + (3*he21-he03)**2
    h4 = (he30+he12)**2 + (he21+he03)**2
    h5 = (he30-3*he12)*(he30+he12)*((he30+he12)**2-3*(he21+he03)**2)\
         +(3*he21-he03)*(he21+he03)\
         *(3*(he30+he12)**2-(he21+he03)**2)
    h6 = (he20-he02)*((he30+he12)**2 - (he21+he03)**2)\
         +4*he11*(he30+he12)*(he21+he03)
    h7 = (3*he21-he03)*(he30+he12)*((he30+he12)**2-3*(he21+he03)**2)\
         -(he30-3*he12)*(he21+he03)*(3*(he30+he12)**2-(he21+he03)**2)

    hu = [he20, he02, he11, he30, he12, he21, he03, h1, h2, h3, h4, h5, h6, h7]#, he22]
    for i, hi in enumerate(hu):
        if hi != 0:
            hu[i] = -1 * math.copysign(1.0, hi) * math.log10(abs(hi))
    return hu #he20, he02, he11, he30, he12, he21, he03


def get_scaled_moment(img, x_ind, y_ind, p, q):
    hu = get_central_moment(img, x_ind, y_ind, p, q)
    hu00 = get_central_moment(img, x_ind, y_ind, 0, 0)
    if hu00 == 0:
        return 0
    else:
        return hu/(hu00**(1 + (p+q)/2))


def get_central_moment(img, x_ind, y_ind, p, q):
    M00 = get_moment(img, x_ind, y_ind, 0, 0)
    x_avg = 0
    y_avg = 0
    if M00 != 0:
        x_avg = get_moment(img, x_ind, y_ind, 1, 0)/M00
        y_avg = get_moment(img, x_ind, y_ind, 0, 1)/M00
    X = np.power(x_ind-x_avg, p)
    Y = np.power(y_ind-y_avg, q)
    return np.sum(X * Y * img)


def get_moment(img, x_ind, y_ind, i, j):
    X = np.power(x_ind, i)
    Y = np.power(y_ind, j)
    return np.sum(X * Y * img)