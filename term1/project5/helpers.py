import cv2
import numpy as np
from skimage.feature import hog
from tqdm import tqdm


def convert_color(img, color_space):
    """Convert image from RGB to color_space

    :param img: input image
    :param color_space: desired color space
    :return: converted image
    """
    if color_space == "RGB":
        return np.copy(img)
    if color_space == "HSV":
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if color_space == "LUV":
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if color_space == "HLS":
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if color_space == "YUV":
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if color_space == "YCrCb":
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """Generate hog features and optional visualization

    :param img: input image
    :param orient: number of orientations
    :param pix_per_cell: number of pixels per cell
    :param cell_per_block: number of cells per block
    :param vis: return visualization
    :param feature_vec: return feature vector
    :return: feature vector and visualization, if specified
    """
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm='L2-Hys',
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm='L2-Hys',
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    """Flattens the image

    :param img: input image
    :param size: size to resize to
    :return: flattened version of resized image
    """
    return cv2.resize(img, size).ravel()


def color_hist(img, nbins=32, bins_range=(0, 256)):
    """Computes the color histogram

    :param img: input image
    :param nbins: number of bins
    :param bins_range: range of bins
    :return: color histograms of each channel concatenated into one vector
    """
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8,
                     cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    """Extract specified features from the list of images

    :param imgs: list of paths to images
    :param color_space: color space to convert to
    :param spatial_size: resized image size for spatial features
    :param hist_bins: number of histogram bins
    :param orient: number of orientations
    :param pix_per_cell: number of pixels per cell
    :param cell_per_block: number of cells per block
    :param hog_channel: hog channel to extract, or ALL for all
    :param spatial_feat: return spatial feature vector
    :param hist_feat: return color histogram feature vector
    :param hog_feat: return hog feature vector
    :return: array of features for each image
    """
    features = []
    for file in tqdm(imgs):
        file_features = []
        image = cv2.imread(file)[..., ::-1]
        feature_image = convert_color(image, color_space)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    return np.array(features)


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            window_list.append(((startx, starty), (endx, endy)))
    return window_list


def find_cars(img, ystart, ystop, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
              window_size, xy_overlap):
    """Find candidate bounding boxes for cars

    :param img: input image
    :param ystart: top y threshold for searchable area of the image
    :param ystop: bottom y threshold for searchable area of the image
    :param scale: scale factor
    :param svc: trained SVC model
    :param X_scaler: fitted StandardScaler
    :param orient: number of orientations
    :param pix_per_cell: number of pixels per cell
    :param cell_per_block: number of cells per block
    :param spatial_size: size of image for spatial feature vector
    :param hist_bins: number of bins for histogram
    :return:
    """
    img_conv = convert_color(img, "YCrCb")

    windows = slide_window(img_conv, x_start_stop=[int(img_conv.shape[1] / 2), None], y_start_stop=[ystart, ystop],
                           xy_window=window_size, xy_overlap=xy_overlap)


    bboxes = []
    for window in windows:
        (startx, starty), (endx, endy) = window
        subimg = cv2.resize(img_conv[starty:endy, startx:endx], (64, 64))
        hog_features0 = get_hog_features(subimg[..., 0], orient, pix_per_cell, cell_per_block, feature_vec=True)
        hog_features1 = get_hog_features(subimg[..., 1], orient, pix_per_cell, cell_per_block, feature_vec=True)
        hog_features2 = get_hog_features(subimg[..., 2], orient, pix_per_cell, cell_per_block, feature_vec=True)
        hog_features = np.hstack((hog_features0, hog_features1, hog_features2))

        # Get color features
        spatial_features = bin_spatial(subimg, size=spatial_size)
        hist_features = color_hist(subimg, nbins=hist_bins)

        test_features = X_scaler.transform(
            np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
        test_prediction = svc.predict(test_features)

        if test_prediction == 1:
            bboxes.append(window)

    return bboxes


def add_heat(heatmap, bbox_list):
    """Count each pixel appearance in the bounding boxes

    :param heatmap: image-sized array of counts
    :param bbox_list: list of bounding boxes, each box takes the form ((x1, y1), (x2, y2))
    :return: updated heatmap
    """
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def apply_threshold(heatmap, threshold):
    """Threshold heatmap

    :param heatmap: heatmap
    :param threshold: threshold
    :return: thresholded heatmap
    """
    heatmap[heatmap <= threshold] = 0
    return heatmap


def draw_labeled_bboxes(img, labels):
    """Draw bounding boxes on the image

    :param img: image
    :param labels: labels of "heated" areas on the heatmap as returned by scipy.ndimage.measurements.label
    :return: image with drawn boxes
    """
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    return img
