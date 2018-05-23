from image_preparation import *
from lane_localisation import *


def pipeline(image, left_lane, right_lane, M, Minv, undist_mtx, undist_dist, ksize=9):
    """Apply processing pipeline to the image
    
    :param image: input image
    :param left_lane: instance of Line class for left lane
    :param right_lane: instance of Line class for right lane
    :param M: perspective transform matrix
    :param M: inverse perspective transform matrix
    :param undist_mtx: camera matrix for distortion correction
    :param undist_dist: distortion coefficients
    :param ksize: kernel size for Sobel
    :return: image with lane region colored
    """
    undist_image = undistort_image(image, undist_mtx, undist_dist)

    # Apply each of the thresholding functions
    gradx_binary = abs_sobel_threshold(image, orient='x', sobel_kernel=ksize, thresh=(50, 150))
    s_binary = hls_threshold(image, thresh=(180, 255))

    combined_binary = np.zeros_like(s_binary)
    combined_binary[(s_binary == 1) | (gradx_binary == 1)] = 1

    warped = cv2.warpPerspective(combined_binary, M, combined_binary.shape[::-1], flags=cv2.INTER_LINEAR)
    if not left_lane.detected or not right_lane.detected:
        left_fit, right_fit, y, left_fitx, right_fitx = find_lanes_initial(warped)
        if len(left_fit) != 0:
            left_lane.detected = True
        if len(right_fit) != 0:
            left_lane.detected = True
    else:
        left_fit, right_fit, y, left_fitx, right_fitx = find_lanes_w_previous(warped, left_lane.best_fit,
                                                                                  right_lane.best_fit)
        if len(left_fit) == 0:
            left_lane.detected = False
        if len(right_fit) == 0:
            left_lane.detected = False

    left_curvature = get_curvature(left_fitx, y)
    right_curvature = get_curvature(right_fitx, y)
    # setting left lane parameters
    left_lane.recent_xfitted.append(left_fitx)
    left_lane.current_fit = left_fit
    left_lane.last_fits.append(left_fit)
    left_lane.update_best()
    left_lane.radius_of_curvature = left_curvature
    # setting right lane parameters
    right_lane.recent_xfitted.append(right_fitx)
    right_lane.current_fit = right_fit
    right_lane.last_fits.append(right_fit)
    right_lane.update_best()
    right_lane.radius_of_curvature = right_curvature

    center_offset = get_center_offset(right_lane.bestx, left_lane.bestx, image.shape[1])
    
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_lane.bestx, y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane.bestx, y])))])
    pts = np.hstack((pts_left, pts_right))
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    average_curvature = (left_curvature + right_curvature) / 2
    cv2.putText(undist_image, "Curvature: {:.2f}".format(average_curvature), (10, 60), font, 2, (255, 255, 255), 5)
    cv2.putText(undist_image, "Vehicle is {:.2f} m from center".format(center_offset), (10, 120), font, 2,
                (255, 255, 255), 5)
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist_image, 1, newwarp, 0.3, 0)
    return result
