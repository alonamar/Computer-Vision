"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import cv2
import os
import numpy as np
import moments_calc

# I/O directories
INPUT_DIR = "input_images"
OUTPUT_DIR = "output"
CATEGORIES = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]

def parse_sequence():
    filepath = os.path.join(INPUT_DIR, "input_sequence.txt")
    d = {}
    with open(filepath) as f:
        for line in f:
            split_line = line.strip('\n').split("\t")
            (key, val) = split_line[0].strip(), split_line[-1]
            d[key] = val.split(",")
    return d
frame_dict = parse_sequence()


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
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    return cv2.VideoWriter(filename, -1, fps, frame_size)


def split_video(vid_dir, video, theta=10, tau=20, full_frame=False):
    my_image_gen = video_frame_generator(os.path.join(vid_dir,video))
    img = my_image_gen.__next__()
    h, w, d = img.shape
    video_name = video.replace("_uncomp.avi","")

    t_minus_img = img
    frame_num = 1
    history = np.zeros((h, w))

    s_frames=[]
    e_frames=[]
    for f in frame_dict[video_name]:
        s_frames.append(int(f.split('-')[0]))
        e_frames.append(int(f.split('-')[1]))

    i = 0
    moments = []
    while img is not None:
        if full_frame:
            tau = int(e_frames[i]) - int(s_frames[i])
        motion = moments_calc.bg_subtraction(t_minus_img, img, theta=theta)
        history = moments_calc.motion_img(history, motion, tau)
        energy = cv2.threshold(history, 0, 1, cv2.THRESH_BINARY)[1]
        if frame_num == s_frames[i] + tau and not full_frame:
            moments.append(np.concatenate([moments_calc.get_hu_moment(history), moments_calc.get_hu_moment(energy)]))
        if frame_num == e_frames[i]:
            i += 1
            if full_frame:
                moments.append(
                    np.concatenate([moments_calc.get_hu_moment(history), moments_calc.get_hu_moment(energy)]))
                history = np.zeros((h, w))
            if i >= len(s_frames):
                break
        t_minus_img = img
        img = my_image_gen.__next__()
        frame_num += 1
    return np.array(moments)


def get_moments(category, theta=10, tau=20, full_frame=False):
    vids_dir = os.path.join(INPUT_DIR, category)
    vids_list = [f for f in os.listdir(vids_dir)
                 if f[0] != '.' and f.endswith('.avi')]
    vids_list.sort()
    i = 0
    hu_moments = np.array([]).reshape(0,28)
    for vid in vids_list:
        print(vid)
        hu_moments = np.concatenate([hu_moments,
                                     split_video(vids_dir, vid, theta=theta, tau=tau, full_frame=full_frame)])
    return hu_moments


def run(full_frame=False):
    data = np.array([]).reshape(0, 29)
    th = 10
    tau = 20
    for i, cat in enumerate(CATEGORIES):
        features = get_moments(cat, th, tau=tau, full_frame=full_frame)
        answer = np.zeros((features.shape[0], 1))
        answer[:, :] = i
        data = np.concatenate([data, np.concatenate([features, answer], axis=1)])
    if full_frame:
        name = "train_data_" + str(th) + "th_full_frame"
    else:
        name = "train_data_" + str(th) + "th_" + str(tau) + "tau"

    np.save(name, data)


if __name__ == "__main__":
    run() # create data set with tau=20 for all actions
    run(True) # create data set with tau that is calculated based on the full action duration
