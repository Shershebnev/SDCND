from argparse import ArgumentParser
from functools import partial
import glob

import cv2
from IPython.display import HTML
from moviepy.editor import VideoFileClip
import numpy as np

from image_preparation import get_calibration_matrix
from line import Line
from pipeline import pipeline


def main(args):
    image_files = glob.glob("camera_cal/calibration*.jpg")
    mtx, dist = get_calibration_matrix(image_files)

    src = np.float32([[580, 460], [710, 460], [1150, 720], [220, 720]])
    dst = np.float32([[200, 0], [1080, 0], [1080, 720], [200, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    left_lane = Line()
    right_lane = Line()
    pipeline_partial = partial(pipeline, left_lane=left_lane, right_lane=right_lane, M=M, Minv=Minv, undist_mtx=mtx,
                               undist_dist=dist)
    test_output = args.output
    clip1 = VideoFileClip(args.input)
    white_clip = clip1.fl_image(pipeline_partial)
    white_clip.write_videofile(test_output, audio=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", dest="input", help="path to input video")
    parser.add_argument("--output", dest="output", help="path to output video")

    args = parser.parse_args()

    main(args)
