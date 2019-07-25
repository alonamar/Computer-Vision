"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np


def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """
    return np.linalg.norm(np.array(p0).astype(float)-np.array(p1).astype(float))



def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """

    h, w = image.shape[:2]
    return [(0, 0), (0, h-1), (w-1, 0), (w-1, h-1)]


def getProjective(src, homography, dst):
    '''
    assisted by:
    https://stackoverflow.com/questions/46520123/failing-the-simplest-possible-cv2-remap-test-aka-how-do-i-use-remap-in-pyt
    '''
    h, w = dst.shape[:2]
    indy, indx = np.indices((h, w)).astype(np.float32)
    lin_homg_ind = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])

    H = np.linalg.inv(homography)  # inverse warping
    map_ind = H.dot(lin_homg_ind)
    map_x, map_y = map_ind[:-1] / map_ind[-1]  # ensure homogeneity
    map_x = map_x.reshape(h, w).astype(np.float32)
    map_y = map_y.reshape(h, w).astype(np.float32)

    res = cv2.remap(src, map_x, map_y, cv2.INTER_CUBIC, dst, cv2.BORDER_TRANSPARENT)  # cubic more clever per lecture
    return res


def check_dist(centers):
    for c1 in range(len(centers)):
        for c2 in range(c1 + 1, len(centers)):
            if euclidean_distance(centers[c1], centers[c2]) < 40:
                return False
    return True


def get_centers(template, img):
    h, w = template.shape[:2]
    for thresh in [0.9,0.88,0.8]:#[0.88, 0.73]:
        for theta in np.roll(range(-60, 105, 15), -4):
            for scale in [1, 1.5, 1.2]: #np.roll(np.linspace(0.5, 2, 7), -2):
                similarity = cv2.getRotationMatrix2D((16, 16), theta, scale)
                similarity = np.vstack((similarity, np.array([0, 0, 1])))
                dst = np.zeros((h, w)).astype(np.uint8)
                sim_template = getProjective(template, similarity, dst)
                res = cv2.matchTemplate(img, sim_template, cv2.TM_CCORR_NORMED)
                loc = np.dstack(np.where(res > thresh))[0].astype(np.float32)
                if loc.shape[0] >= 16:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    _, _, center = cv2.kmeans(loc, 4, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
                    if not check_dist(center):
                        continue
                    center = np.uint16(center)
                    center[:, [1, 0]] = center[:, [0, 1]]
                    center[:, 0] += int(np.round(w / 2))
                    center[:, 1] += int(np.round(h / 2))
                    res_loc = center[center[:, 0].argsort(axis=0)]
                    top_left, bottom_left = list(map(tuple, res_loc[:2][res_loc[:2, 1].argsort(axis=0)]))
                    top_right, bottom_right = list(map(tuple, res_loc[2:][res_loc[2:, 1].argsort(axis=0)]))
                    centers = [top_left, bottom_left, top_right, bottom_right]
                    return centers
    return None


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    img = np.copy(image)
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3,3),0)
    gray_template = cv2.GaussianBlur(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), (3,3),0)
    center = get_centers(gray_template, gray)
    if center is None:
        return [(-1,-1), (-1,-2), (-2,-1), (-2,-2)]
    return center

def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    img = np.copy(image)
    cv2.line(img, markers[0], markers[1], 255, thickness)
    cv2.line(img, markers[1], markers[3], 255, thickness)
    cv2.line(img, markers[3], markers[2], 255, thickness)
    cv2.line(img, markers[2], markers[0], 255, thickness)
    return img


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """

    return getProjective(imageA, homography, imageB)


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
    A = np.zeros((8, 9))
    iter = [0, 2, 4, 6]
    for (x, y), (x_tag, y_tag), i in zip(
            np.array(src_points).astype(np.float64), np.array(dst_points).astype(np.float64), iter):
        A[i, :] = np.array([x, y, 1, 0, 0, 0, -x*x_tag, -y*x_tag, -x_tag])
        A[i+1, :] = np.array([0, 0, 0, x, y, 1, -x*y_tag, -y*y_tag, -y_tag])

    u, s, v = np.linalg.svd(A)
    return v[-1].reshape(3, 3)/v[-1, -1]


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None


def showimg(img):
    cv2.imshow("img", img)
    cv2.waitKey()
