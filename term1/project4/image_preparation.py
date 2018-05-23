import cv2
import numpy as np


NX = 9
NY = 6


def get_calibration_matrix(image_paths):
    """Create calibration matrix using chessboard images
    
    :param image_paths: list of images of chessboard pattern
    :return: camera matrix and distortion coefficients
    """
    objpoints = []
    imgpoints = []
    objp = np.zeros((NX * NY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2)

    for image_file in image_paths:
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (NX, NY), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist

def undistort_image(image, mtx, dist):
    """Undistortion of the image
    
    :param image: distorted image
    :param mtx: camera matrix
    :param dist: distortion coefficients
    :return: undistorted image
    """
    return cv2.undistort(image, mtx, dist, None, mtx)

def abs_sobel_threshold(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """Apply sobel operator in specified orientation and apply threshold to the
    calculated gradients
    
    :param image: input image
    :param orient: orientation in which to apply sobel
    :param sobel_kernel: kernel size
    :param thresh: tuple with lower and upper threshold for gradient value
    :return: mask of gradients that pass specified thresholds
    """
    # Calculate directional gradient
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if orient == "x":
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == "y":
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    grad_binary = np.zeros_like(gray)
    # Apply threshold
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_threshold(image, sobel_kernel=3, thresh=(0, 255)):
    """Calculate the magnitude of the gradients
    
    :param image: input image
    :param sobel_kernel: kernel size
    :param thresh: tuple with lower and upper threshold for magnitude of gradients
    :return: mask of magnitudes of gradients that pass specified thresholds
    """
    # Calculate gradient magnitude
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_mag = np.uint8(255 * mag / np.max(mag))
    mag_binary = np.zeros_like(gray)
    # Apply threshold
    mag_binary[(scaled_mag >= thresh[0]) & (scaled_mag <= thresh[1])] = 1
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    """Calculate the direction of the gradients
    
    :param image: input image
    :param sobel_kernel: kernel size
    :param thresh: tuple with lower and upper threshold for direction of gradients
    :return: mask of directions of gradients that pass specified thresholds
    """
    # Calculate gradient direction
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    dir_sobel = np.arctan2(sobely, sobelx)
    dir_binary = np.zeros_like(gray)
    # Apply threshold
    dir_binary[(dir_sobel >= thresh[0]) & (dir_sobel <= thresh[1])] = 1
    return dir_binary

def hls_threshold(image, thresh=(0, 255)):
    """Apply saturation channel thresholding
    
    :param image: input image
    :param thresh: tuple with lower and upper threshold for saturation   
    :return: mask of saturation channel that passes specified thresholds
    """
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    s_channel = hls[..., 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return s_binary
