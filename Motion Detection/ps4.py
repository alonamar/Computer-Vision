"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2
import os


# Utility function
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

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """

    return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, scale=1./8)


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """

    return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, scale=1./8)


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """
    h, w = img_b.shape
    Ix = gradient_x(img_a)
    Iy = gradient_y(img_a)
    It = img_b - img_a
    grads = [Ix*Ix, Ix*Iy, Ix*It, Iy*Ix, Iy*Iy, Iy*It]

    kernel = np.ones((k_size, k_size)) / (k_size**2)  # default: uniform
    if k_type == 'gaussian':
        kernel = cv2.getGaussianKernel(k_size, sigma=sigma)
        kernel = kernel * kernel.T

    sum_grads = []
    for grad in grads:
        sum_grads.append(cv2.filter2D(grad, -1, kernel))
    sum_grads = np.array(sum_grads).reshape((2, 3, h, w)).transpose((2,3,0,1))

    A, b = sum_grads[:, :, :, 0:2], -1*sum_grads[:, :, :, 2]
    det = np.linalg.det(A)
    det[np.where(det == 0.0)] = np.inf

    u = (A[:, :, 1, 1] * b[:, :, 0] + -1*A[:, :, 0, 1] * b[:, :, 1]) / det
    v = (-1*A[:, :, 1, 0] * b[:, :, 0] + A[:, :, 0, 0] * b[:, :, 1]) / det
    return u, v


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    kernel = np.array((1, 4, 6, 4, 1)) / 16.
    filtered = cv2.sepFilter2D(image, cv2.CV_64F, kernel, kernel)
    return filtered[::2, ::2]


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """

    pyr = [image]
    for i in range(1,levels):
        pyr.append(reduce_image(pyr[i-1]))
    return pyr


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    out = []
    h = img_list[0].shape[0]
    for i in img_list:
        img = np.pad(i, [(0, h-i.shape[0]), (0, 0)], mode='constant')
        out.append(normalize_and_scale(img))
    return np.concatenate(out, axis=1)


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
    kernel = np.array((1, 4, 6, 4, 1)) / 8.
    expended = np.zeros((2*image.shape[0], 2*image.shape[1]))
    expended[::2, ::2] = image
    filtered = cv2.sepFilter2D(expended, cv2.CV_64F, kernel, kernel)
    return filtered


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """

    l_pyr = []
    for i in range(len(g_pyr)-1):
        exp_img = expand_image(g_pyr[i+1])
        if g_pyr[i].shape[0] % 2 != 0:
            exp_img = exp_img[:-1]
        if g_pyr[i].shape[1] % 2 != 0:
            exp_img = exp_img[:, :-1]
        l_pyr.append(g_pyr[i] - exp_img)
    l_pyr.append(g_pyr[-1])
    return l_pyr


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    M, N = image.shape
    X, Y = np.meshgrid(range(N), range(M))
    return cv2.remap(image, np.float32(X + U), np.float32(Y + V), interpolation, border_mode)


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """
    img_a_pyr = gaussian_pyramid(img_a, levels)
    img_b_pyr = gaussian_pyramid(img_b, levels)
    u, v = optic_flow_lk(img_a_pyr[levels - 1], img_b_pyr[levels - 1], k_size, k_type, sigma)
    for i in range(levels - 2, -1, -1):
        u, v = expend_u_v(u, v, img_a_pyr[i])
        warped_img = warp(img_b_pyr[i], u, v, interpolation, border_mode)
        u_cor, v_cor = optic_flow_lk(img_a_pyr[i], warped_img, k_size, k_type, sigma)
        u, v = u+u_cor, v+v_cor
    return u, v


def expend_u_v(u, v, pyr):
    u = 2 * expand_image(u)
    v = 2 * expand_image(v)
    if pyr.shape[0] % 2 != 0:
        u, v = u[:-1], v[:-1]
    if pyr.shape[1] % 2 != 0:
        u, v = u[:, :-1], v[:, :-1]
    return u, v
