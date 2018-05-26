from argparse import ArgumentParser
from functools import partial
import pickle

from moviepy.editor import VideoFileClip
import numpy as np
from scipy.ndimage.measurements import label

from constants import *
from heatmap import Heatmap
from helpers import find_cars, add_heat, apply_threshold, draw_labeled_bboxes


def pipeline(image, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
             heatmap_history):
    scale1_bbox = find_cars(image, 380, 480, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                                               hist_bins, window_size=(64, 64), xy_overlap=(0.75, 0.75))
    scale2_bbox = find_cars(image, 400, 600, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                                               hist_bins, window_size=(96, 96), xy_overlap=(0.75, 0.75))
    scale3_bbox = find_cars(image, 500, 700, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                                               hist_bins, window_size=(128, 128), xy_overlap=(0.75, 0.75))
    bboxes = scale1_bbox + scale2_bbox + scale3_bbox
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heat = add_heat(heat, bboxes)
    heatmap_history.queue.append(heat)
    heat = apply_threshold(heatmap_history.sum_heatmap(), 8)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    return draw_img


def main(args):
    svc = pickle.load(open(args.svc, "rb"))
    scaler = pickle.load(open(args.scaler, "rb"))
    heatmap_history = Heatmap()
    pipeline_partial = partial(pipeline, svc=svc, X_scaler=scaler, orient=orient, pix_per_cell=pix_per_cell,
                               cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins,
                               heatmap_history=heatmap_history)
    clip1 = VideoFileClip(args.input)
    white_clip = clip1.fl_image(pipeline_partial)
    white_clip.write_videofile(args.output, audio=False)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-s", "--svc", dest="svc", help="Path to trained and pickled SVC model")
    parser.add_argument("-sc", "--scaler", dest="scaler", help="Path to fitted and pickled StandardScaler")
    parser.add_argument("-i", "--input", dest="input", help="Path to input video")
    parser.add_argument("-o", "--output", dest="output", help="Path to output video")

    args = parser.parse_args()

    main(args)