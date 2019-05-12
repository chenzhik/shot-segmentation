import os
import cv2
import numpy as np
import scipy.signal as si
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("video_file")
parser.add_argument("-save_frames", type=bool)
parser.add_argument("-show_hist_diff", type=bool)
args = parser.parse_args()
print(args.video_file)


video_formats = [".MP4", ".mp4"]
videos_src_path = args.video_file
if os.path.isfile(videos_src_path):
    frames_save_path = os.path.dirname(videos_src_path)# + '/'
else:
    raise SystemExit('wrong file path')


def video2frame(video_src_path, formats, frame_save_path):
    """
    detect the shot boundary of the first and last frame
    :param video_src_path:  video file to detect
    :param formats:　video formats to detect
    :param frame_save_path: frames saved location
    :return:　frame and index
    """

    f = open("./result.txt", 'w')
    diff = 0.42

    if os.path.isfile(video_src_path):
        video = video_src_path.split('/')[-1]
    else:
        raise SystemExit('wrong path')

    if video[-4:] not in formats:
        raise SystemExit('wrong format')

    if not os.path.exists(frame_save_path):
        raise SystemExit('directory not exists')

    # open a video
    cap = cv2.VideoCapture(video_src_path)

    frame_num = cap.get(7)
    print(0, int(frame_num - 1))
    frame_diff = []
    frame_index = []

    if cap.isOpened():
        success = True
    else:
        success = False
        print("Fail in reading videos!")

    frame_count = 0

    # read a frame
    while success:
        read, frame = cap.read()
        print(frame_count, read)

        if read:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            n_pixel = frame.shape[0] * frame.shape[1]
            hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
            hist = hist * (1.0 / n_pixel)

            frame_diff.append(hist)
            frame_index.append(frame_count)

            frame_count += 1

        else:
            break

    # compute the distance between this frame with previous frame
    frame_count = len(frame_diff) - 1

    while frame_count >= 1:
        frame_diff[frame_count] = np.sum(np.abs(np.subtract(frame_diff[frame_count], frame_diff[frame_count - 1])))
        frame_count -= 1

    frame_diff[0] = 0

    vector = np.array(frame_diff)
    indexes, _ = si.find_peaks(vector, height=diff, distance=10)

    print('Numbers of shots', len(indexes))

    # write down the shot boundary frames' index
    f.write("0 ")
    for index in indexes:
        f.write(str(index-1)+"\n"+str(index)+" ")

        # save boundary frames
        if args.save_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)  # set the first frame index to read
            read_1, frame_1 = cap.read()
            print(index-1, read_1)
            cv2.imwrite(frames_save_path + "%d.png" % (index-1), frame_1)

            read_2, frame_2 = cap.read()
            print(index, read_2)
            cv2.imwrite(frames_save_path + "%d.png" % index, frame_2)

    f.write(str(int(frame_num-1)))

    # close the video
    cap.release()

    # show the frames' histogram differences
    if args.show_hist_diff:
        import matplotlib.pyplot as plt
        plt.plot(frame_index, frame_diff)
        plt.xlabel("frame")
        plt.ylabel("histogram difference with previous one")
        plt.show()

if __name__ == '__main__':
    video2frame(videos_src_path, video_formats, frames_save_path)