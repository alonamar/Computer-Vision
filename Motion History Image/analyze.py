import cv2
import os
import numpy as np
import moments_calc
import clf_finder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm

INPUT_DIR = "input_images"
OUTPUT_DIR = "output"
CATEGORIES = np.array(["walking", "jogging", "running", "boxing", "handwaving", "handclapping"])


def parse_sequence():
    filepath = os.path.join(INPUT_DIR, "input_sequence_multi.txt")
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


def analyze(vid_dir, video, theta, clf, scaler, action_num, tau=20, full_frame=False):
    my_image_gen = video_frame_generator(os.path.join(vid_dir,video))
    img = my_image_gen.__next__()
    h, w, d = img.shape
    fps = 20
    video_name = video.replace("_uncomp.avi","")

    t_minus_img = img
    frame_num = 1
    history = np.zeros((h, w))
    s_frames = []
    e_frames = []
    for f in frame_dict[video_name]:
        s_frames.append(int(f.split('-')[0]))
        e_frames.append(int(f.split('-')[1]))
    i = 0


    if full_frame:
        video_name = "ff_" + video_name
    else:
        video_name = "tau_" + video_name
    out_path = "output/" + video_name + "_.mp4"
    video_out = mp4_video_writer(out_path, (w, h), fps)

    res = np.zeros(6)
    full_motion = np.zeros(6)
    color = (0, 50, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while img is not None:
        myImg = np.copy(img)
        if full_frame:
            tau = int(e_frames[i]) - int(s_frames[i])
        motion = moments_calc.bg_subtraction(t_minus_img, img, theta=theta)
        history = moments_calc.motion_img(history, motion, tau)
        energy = cv2.threshold(history, 0, 1, cv2.THRESH_BINARY)[1]

        if full_frame:
            if frame_num == e_frames[i]:
                i += 1
                moments = np.concatenate([moments_calc.get_hu_moment(history), moments_calc.get_hu_moment(energy)])
                res_i = clf.predict(scaler.transform(moments.reshape(1, 28)))
                res[int(res_i)] += 1
                if i >= len(s_frames):
                    break
                cv2.putText(myImg, CATEGORIES[int(res_i)], (50, 50), font, 0.5, color, 1)
        else:
            if np.sum(motion) < 5:
                pass
            elif frame_num > tau:
                moments = np.concatenate([moments_calc.get_hu_moment(history), moments_calc.get_hu_moment(energy)])
                #res_i = clf.predict(scaler.transform(moments.reshape(1, 28)))
                #res[int(res_i)] += 1
                res[(clf.predict_proba(scaler.transform(moments.reshape(1, 28))) > 0.7).flatten()] += 1
                if (clf.predict_proba(scaler.transform(moments.reshape(1, 28))) > 0.7).any():
                    res_i = clf.predict(scaler.transform(moments.reshape(1, 28)))
                    cv2.putText(myImg, CATEGORIES[int(res_i)], (50, 50), font, 0.5, color, 1)
        video_out.write(myImg)
        t_minus_img = img
        img = my_image_gen.__next__()
        frame_num += 1
    video_out.release()
    print("The occurrences of each action:")
    print(res)


def run(full_frame=False):
    print("################")
    print("The categories are numbered in order:")
    print(CATEGORIES)
    print("################")
    vids_list = [f for f in os.listdir(INPUT_DIR)
                 if f[0] != '.' and f.endswith('.avi')]
    vids_list.sort()
    theta = 10
    if full_frame:
        clf = svm.SVC(C=1000000, gamma=1e-05, kernel='rbf', probability=True)
        datafile = "train_data_10th_full_frame.npy"
    else:
        clf = svm.SVC(C=10000, gamma=0.0001, kernel='rbf', probability=True)
        datafile = "train_data_10th_20tau.npy"

    X_train, X_test, y_train, y_test, scaler = clf_finder.split_data(datafile)
    clf.fit(X_train, y_train)
    y_res = clf.predict(X_test)
    print(accuracy_score(y_test, y_res))
    for vid in vids_list:
        print(vid)
        analyze(INPUT_DIR, vid, theta, clf, scaler, 3, full_frame=full_frame)


if __name__ == "__main__":
        run()
        run(True)
