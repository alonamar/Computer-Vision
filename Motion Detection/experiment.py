"""Problem Set 4: Motion Detection"""

import cv2
import os
import numpy as np

import ps4

# I/O directories
input_dir = "input_images"
output_dir = "output"


def frame_interpolation(img1, img2, name, levels, k_size):
    k_type = ""  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u, v = ps4.hierarchical_lk(img1, img2, levels, k_size,
                               k_type, sigma, interpolation, border_mode)

    t_vec = np.linspace(0.2, 0.8, 4)
    h, w = img1.shape
    warped_list = [img1]
    for t in t_vec:
        warped_list.append(ps4.warp(img1, -t * u, -t * v, interpolation, border_mode))
    warped_list.append(img2)

    res = np.array(warped_list).reshape(2, 3, h, w)
    res = np.concatenate([np.concatenate(res[0, :], axis=1), np.concatenate(res[1, :], axis=1)])

    cv2.imwrite(os.path.join(output_dir, name),
                ps4.normalize_and_scale(res))


def get_optic_flow(img1, img2, k_size, out, smooth, sigmaX):
    my_img1 = cv2.GaussianBlur(img1, smooth, sigmaX)
    my_img2 = cv2.GaussianBlur(img2, smooth, sigmaX)
    k_type = ""  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(my_img1, my_img2, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=1, stride=10)
    cv2.imwrite(os.path.join(output_dir, out), u_v)


# Utility code
def quiver(u, v, scale, stride, color=(0, 255, 0)):

    img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

    for y in range(0, v.shape[0], stride):

        for x in range(0, u.shape[1], stride):
            try:
                cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                           y + int(v[y, x] * scale)), color, 1)
                cv2.circle(img_out, (x + int(u[y, x] * scale),
                                     y + int(v[y, x] * scale)), 1, color, 1)
            except:
                print(u[y, x], v[y, x])

    return img_out


# Functions you need to complete:

def scale_u_and_v(u, v, level, pyr):
    """Scales up U and V arrays to match the image dimensions assigned 
    to the first pyramid level: pyr[0].

    You will use this method in part 3. In this section you are asked 
    to select a level in the gaussian pyramid which contains images 
    that are smaller than the one located in pyr[0]. This function 
    should take the U and V arrays computed from this lower level and 
    expand them to match a the size of pyr[0].

    This function consists of a sequence of ps4.expand_image operations 
    based on the pyramid level used to obtain both U and V. Multiply 
    the result of expand_image by 2 to scale the vector values. After 
    each expand_image operation you should adjust the resulting arrays 
    to match the current level shape 
    i.e. U.shape == pyr[current_level].shape and 
    V.shape == pyr[current_level].shape. In case they don't, adjust
    the U and V arrays by removing the extra rows and columns.

    Hint: create a for loop from level-1 to 0 inclusive.

    Both resulting arrays' shapes should match pyr[0].shape.

    Args:
        u: U array obtained from ps4.optic_flow_lk
        v: V array obtained from ps4.optic_flow_lk
        level: level value used in the gaussian pyramid to obtain U 
               and V (see part_3)
        pyr: gaussian pyramid used to verify the shapes of U and V at 
             each iteration until the level 0 has been met.

    Returns:
        tuple: two-element tuple containing:
            u (numpy.array): scaled U array of shape equal to 
                             pyr[0].shape
            v (numpy.array): scaled V array of shape equal to 
                             pyr[0].shape
    """

    for i in range(level - 1, -1, -1):
        u, v = ps4.expend_u_v(u, v, pyr[i])
    return u, v


def part_1a():

    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                       'ShiftR2.png'), 0) / 255.
    shift_r5_u5 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                          'ShiftR5U5.png'), 0) / 255.

    # Optional: smooth the images if LK doesn't work well on raw images
    shift_0_smooth = cv2.GaussianBlur(shift_0, (9, 9), 0)
    shift_r2_smooth = cv2.GaussianBlur(shift_r2, (9, 9), 0)
    k_size = 21  # TODO: Select a kernel size
    k_type = ""  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0_smooth, shift_r2_smooth, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-1.png"), u_v)

    # Now let's try with ShiftR5U5. You may want to try smoothing the
    # input images first.
    shift_r5_u5_smooth = cv2.GaussianBlur(shift_r5_u5, (9, 9), 0)
    k_size = 77 # TODO: Select a kernel size
    k_type = ""  # TODO: Select a kernel type
    sigma = 0 # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0_smooth, shift_r5_u5_smooth, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-2.png"), u_v)


def part_1b():
    """Performs the same operations applied in part_1a using the images
    ShiftR10, ShiftR20 and ShiftR40.

    You will compare the base image Shift0.png with the remaining
    images located in the directory TestSeq:
    - ShiftR10.png
    - ShiftR20.png
    - ShiftR40.png

    Make sure you explore different parameters and/or pre-process the
    input images to improve your results.

    In this part you should save the following images:
    - ps4-1-b-1.png
    - ps4-1-b-2.png
    - ps4-1-b-3.png

    Returns:
        None
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.
    s_size, sigmaX = (29, 29), 17
    get_optic_flow(shift_0, shift_r10, 51, "ps4-1-b-1.png", s_size, sigmaX)
    get_optic_flow(shift_0, shift_r20, 55, "ps4-1-b-2.png", s_size, sigmaX)
    get_optic_flow(shift_0, shift_r40, 69, "ps4-1-b-3.png", s_size, sigmaX)


def part_2():

    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1',
                                         'yos_img_01.jpg'), 0) / 255.

    # 2a
    levels = 4
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_01_g_pyr_img = ps4.create_combined_img(yos_img_01_g_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-a-1.png"),
                yos_img_01_g_pyr_img)

    # 2b
    yos_img_01_l_pyr = ps4.laplacian_pyramid(yos_img_01_g_pyr)

    yos_img_01_l_pyr_img = ps4.create_combined_img(yos_img_01_l_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-b-1.png"),
                yos_img_01_l_pyr_img)


def part_3a_1():
    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.

    levels = 2  # Define the number of pyramid levels
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)

    level_id = 1  # TODO: Select the level number (or id) you wish to use
    k_size = 51 # TODO: Select a kernel size
    k_type = ""  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_01_g_pyr[level_id],
                             yos_img_02_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)

    diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-1.png"),
                ps4.normalize_and_scale(diff_yos_img_01_02))


def part_3a_2():
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
    yos_img_03 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.

    levels = 2  # Define the number of pyramid levels
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)
    yos_img_03_g_pyr = ps4.gaussian_pyramid(yos_img_03, levels)

    level_id = 1 # TODO: Select the level number (or id) you wish to use
    k_size = 51 # TODO: Select a kernel size
    k_type = ""  # TODO: Select a kernel type
    sigma = 0 # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_02_g_pyr[level_id],
                             yos_img_03_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_03_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_03_warped = ps4.warp(yos_img_03, u, v, interpolation, border_mode)

    diff_yos_img = yos_img_02 - yos_img_03_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-2.png"),
                ps4.normalize_and_scale(diff_yos_img))


def part_4a():
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.

    levels = 4  # TODO: Define the number of levels
    k_size = 29  # TODO: Select a kernel size
    k_type = ""  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u10, v10 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u10, v10, scale=1, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-1.png"), u_v)

    # You may want to try different parameters for the remaining function
    # calls.
    k_size = 45
    u20, v20 = ps4.hierarchical_lk(shift_0, shift_r20, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u20, v20, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-2.png"), u_v)

    k_size = 59
    u40, v40 = ps4.hierarchical_lk(shift_0, shift_r40, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    u_v = quiver(u40, v40, scale=6, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-3.png"), u_v)


def part_4b():
    urban_img_01 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban01.png'), 0) / 255.
    urban_img_02 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban02.png'), 0) / 255.

    levels = 5  # TODO: Define the number of levels
    k_size = 29  # TODO: Select a kernel size
    k_type = ""  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u, v = ps4.hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
                               k_type, sigma, interpolation, border_mode)

    u_v = quiver(u, v, scale=1, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-1.png"), u_v)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    urban_img_02_warped = ps4.warp(urban_img_02, u, v, interpolation,
                                   border_mode)

    diff_img = urban_img_01 - urban_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-2.png"),
                ps4.normalize_and_scale(diff_img))


def part_5a():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    frame_interpolation(shift_0, shift_r10, "ps4-5-a-1.png", 4, 9)


def part_5b():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    cooper1 = cv2.imread(os.path.join(input_dir, 'MiniCooper',
                                      'mc01.png'), 0) / 255.
    cooper2 = cv2.imread(os.path.join(input_dir, 'MiniCooper',
                                      'mc02.png'), 0) / 255.
    cooper3 = cv2.imread(os.path.join(input_dir, 'MiniCooper',
                                      'mc03.png'), 0) / 255.

    frame_interpolation(cooper1, cooper2, "ps4-5-b-1.png", 4, 39)

    frame_interpolation(cooper2, cooper3, "ps4-5-b-2.png", 4, 39)


def part_6():
    """Challenge Problem

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    my_video = "ps4-my-video.mp4"  # Place your video in the input_video directory
    fps = 20

    my_video = os.path.join("input_videos", my_video)
    my_image_gen = video_frame_generator(my_video)

    image = my_image_gen.__next__()
    h, w, d = image.shape

    out_path = "output/ps4-my-video.mp4"
    video_out = mp4_video_writer(out_path, (w, h), fps)

    frame_num = 1
    init_image = image
    while image is not None:
        print("Processing fame {}".format(frame_num))

        init_image_grey = cv2.cvtColor(init_image, cv2.COLOR_BGR2GRAY)
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        levels = 4  # TODO: Define the number of levels
        k_size = 51  # TODO: Select a kernel size
        k_type = ""  # TODO: Select a kernel type
        sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
        interpolation = cv2.INTER_CUBIC  # You may try different values
        border_mode = cv2.BORDER_REFLECT101  # You may try different values

        u, v = ps4.hierarchical_lk(init_image_grey, image_grey, levels, k_size, k_type,
                                       sigma, interpolation, border_mode)

        u_v = quiver(u, v, scale=0.1, stride=10, color=(0, 0, 255))

        out = cv2.addWeighted(np.copy(image), 0.8, u_v, 1, 0.8)

        if frame_num == 50:
            cv2.imwrite(os.path.join(output_dir, "ps4-6-a-1.png"), out)
        if frame_num == 150:
            cv2.imwrite(os.path.join(output_dir, "ps4-6-a-2.png"), out)

        video_out.write(out)
        init_image = image
        image = my_image_gen.__next__()
        frame_num += 1

    video_out.release()


def video_frame_generator(filename):
    video = cv2.VideoCapture(filename)
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break
    video.release()
    yield None


def mp4_video_writer(filename, frame_size, fps=20):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


if __name__ == '__main__':
    part_1a()
    part_1b()
    part_2()
    part_3a_1()
    part_3a_2()
    part_4a()
    part_4b()
    part_5a()
    part_5b()
    part_6()
