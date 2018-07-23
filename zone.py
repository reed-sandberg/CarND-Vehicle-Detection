"""A zone or region of interest within a video camera's scope."""

from collections import deque
import itertools
import pickle
import statistics

import numpy as np

import cv2

from scipy.ndimage.measurements import label

from lesson_functions import *


ONE_DEGREE = np.pi / 180


#######################################################################################################################
# Functions related to lane identification.
#######################################################################################################################
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def x_of_slope_intercept(y, b, m):
    """Given linear equation (slope-intercept) params, calculate x."""
    return int((y - b) / m)

def bounded_line_properties(line, y_bound):
    """Returns properties of the given line as a tuple of floats, members of which could be inf."""
    x1, y1, x2, y2 = line[0]
    if x2 - x1 == 0:
        slope = float('inf')
    else:
        slope = float(y2 - y1) / float(x2 - x1)
    y_intercept = y1 - slope * x1
    if slope == 0:
        x_intercept = float('inf')
        x_bound_intercept = float('inf')
    else:
        x_intercept = -y_intercept / slope
        x_bound_intercept = (y_bound - y_intercept) / slope
    return slope, y_intercept, x_intercept, x_bound_intercept

def smooth_line(lines):
    """Return a line (slope, x_bound) that is a smooth representation of the given list of lines."""
    x_bound_intercept = int(statistics.median([line[4] for line in lines]))
    slope = statistics.median([line[1] for line in lines])
    return slope, x_bound_intercept

def lines_intersection(right_line, left_line):
    """Lines intersect where x = (b2 - b1) / (m1 - m2)."""
    m1, b1, _ = right_line
    m2, b2, _ = left_line
    if m1 == m2:
        return float('inf'), float('inf')
    x_intersect = int((b2 - b1) / (m1 - m2))
    y_intersect = int(m1 * x_intersect + b1)
    return x_intersect, y_intersect

def window_mask(width, height, img_ref, center, level):
    """Draw window areas around the centroid where lane boundary pixels were identified."""
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height), \
           max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def driver_perspective(image, left_fit, right_fit, Minv):
    """Transform a frame back to the original driver's perspective."""
    img_y_series = range(0, image.shape[0])
    left_fitx = np.array(left_fit[0]*img_y_series*img_y_series + left_fit[1]*img_y_series + left_fit[2], dtype=np.int32)
    right_fitx = np.array(right_fit[0]*img_y_series*img_y_series + right_fit[1]*img_y_series + right_fit[2],
                          dtype=np.int32)

    # Create an image to draw the lines on.
    lanes_top_view = np.zeros_like(image)

    # Recast the x and y points into usable format for cv2.fillPoly().
    pts_left = np.array([np.transpose(np.vstack([left_fitx, img_y_series]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, img_y_series])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image.
    cv2.fillPoly(lanes_top_view, np.int_([pts]), (0, 255, 0))

    # Warp back to original image space using inverse perspective matrix (Minv).
    overlay_lanes_fit = cv2.warpPerspective(lanes_top_view, Minv, (image.shape[1], image.shape[0])) 
    return overlay_lanes_fit, left_fitx, right_fitx

#######################################################################################################################
# Functions related to vehicle tracking.
#######################################################################################################################
def areas_overlap(area_51, area_52):
    """Return true if the given rectangular coordinates overlap in space, false otherwise."""
    (x1_51, y1_51), (x2_51, y2_51) = area_51
    (x1_52, y1_52), (x2_52, y2_52) = area_52
    return x1_51 <= x2_52 and x2_51 >= x1_52 and y1_51 <= y2_52 and y2_51 >= y1_52

def area_contains(area_51, area_52):
    """Return true if area_52 is contained within area_51 (inclusive of boundaries), false otherwise."""
    (x1_51, y1_51), (x2_51, y2_51) = area_51
    (x1_52, y1_52), (x2_52, y2_52) = area_52
    return x1_51 <= x1_52 and x2_52 <= x2_51 and y1_51 <= y1_52 and y2_52 <= y2_51

def merge_areas(area_51, area_52):
    """Return a rectangular area that encompasses both area_51 area_52 in their entirety."""
    (x1_51, y1_51), (x2_51, y2_51) = area_51
    (x1_52, y1_52), (x2_52, y2_52) = area_52
    return (min(x1_51,x1_52), min(y1_51,y1_52)), (max(x2_51,x2_52), max(y2_51,y2_52))

def represent_heatmap(img, thermal_areas, temp_threshold):
    """Represent thermal areas as a heatmap image, apply a threshold (minimum representation to be considered a hot
    area) and return a tuple of the heatmap and coordinates of the extracted rectangular hot areas."""
    heat_canvas = np.zeros_like(img[:,:,0]).astype(np.float)

    # Represent a unit of "heat" for each thermal_area.
    heat_canvas = add_heat(heat_canvas, thermal_areas)

    # Apply a threshold to help remove noise.
    heat_canvas = apply_threshold(heat_canvas, temp_threshold)

    # Respect pixel value limits.
    heatmap = np.clip(heat_canvas, 0, 255)

    # Extract hot areas from the heatmap.
    labels = label(heatmap)
    _, hot_areas = extract_labeled_areas(np.copy(img), labels, draw=False)

    return heatmap, hot_areas

def match_vehicle_areas(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                        hist_bins, cells_per_step, cspace):
    """Extract features of img using hog sub-sampling and use them to detect other cars in img. Return a list of
    rectangular coordinates for all matching areas.
    
    Use a sliding window technique to cover the region of interest of img (from ystart to ystop) and generate a list
    of overlapping windows. Features of each window will be fed into the trained classifier to predict if a car is
    likely occupying that same area. For efficiency, a single HOG transform is performed on the entire ROI at once
    rather than several times for each overlapping window. To make sure the window dimensions align and produce
    a feature vector of the exact size the classifier expects, the input image will be scaled according to the desired
    size of each sliding window. To extract HOG features, a corresponding window will be extracted of a "map" of the
    HOG features represented as a 2-D array, one map for each color channel of the input image."""

    img_roi = img[ystart:ystop,:,:]
    img_roi = as_color_model(img_roi, cspace)

    # Scale the image so features extracted from each window align with the expected input feature vector size
    # of the classifier.
    if scale != 1:
        imshape = img_roi.shape
        img_roi = cv2.resize(img_roi, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = img_roi[:,:,0]
    ch2 = img_roi[:,:,1]
    ch3 = img_roi[:,:,2]

    # Determine HOG transform parameters to align with the desired scale.
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the sampling rate used for training the classifier, with 8 cells and 8 pix per cell.
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image.
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    vehicle_match_areas = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG features for this window.
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch for this window, which should match the dimensions of the training data: 64x64.
            subimg = img_roi[ytop:ytop+window, xleft:xleft+window]

            # Determine color features.
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale the features and make a prediction.
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                vehicle_match_areas.append(((xbox_left, ytop_draw+ystart),
                                            (xbox_left+win_draw,ytop_draw+win_draw+ystart)))

    return vehicle_match_areas

def add_heat(heatmap, bbox_list):
    """Increment pixel values on the heatmap image for each area of bbox_list. Each member of bbox_list takes the form
    ((x1, y1), (x2, y2))."""
    for bbox in bbox_list:
        # Add += 1 for all pixels inside each bbox
        heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1
    return heatmap
    
def apply_threshold(heatmap, threshold):
    """Zero out pixels below the threshold and keep only the hot areas."""
    heatmap[heatmap < threshold] = 0
    return heatmap

def extract_labeled_areas(img, labels, draw=False):
    """Return a 2-tuple of an image and a list of rectangular coordinates labeled via scipy.ndimage.measurements.label.
    The input image will have boxes drawn for each of the rectangular coordinates if draw is True."""

    bboxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value.
        nonzero = (labels[0] == car_number).nonzero()
        # Identify the x and y values of those pixels.
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y.
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        if draw:
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        bboxes.append(bbox)
    return img, bboxes

class LaneBoundaryZone(object):
    """Identify the current path of travel."""

    def __init__(self, height, width):
        # Zone dimensions.
        self.width = width
        self.height = height

        # Pre-calculate these handy quantities.
        self.width_minus1 = width - 1
        self.height_minus1 = height - 1
        self.halfwidth = self.width // 2
        self.halfheight = self.height // 2

        # Size of history buffer to store lane info for each frame.
        self.lane_frame_hist_size = 60
        # Number of previous frames to use to average/smooth lane boundary locations.
        self.lane_frame_smooth_size = 6
        # Number of previous frames to use to average/smooth horizon locations.
        self.horizon_frame_smooth_size = self.lane_frame_hist_size
        # Number of previous frames to use to average/smooth lane boundary curve info.
        self.lane_fit_smooth_size = 6

        # Add some margin around the located lane boundaries before cropping for a region-of-interest (ROI). The ROI
        # is used to limit the area of a frame for processing only that part of an image we're interested in and reduce
        # surrounding noise.
        self.roi_boundary_fuzz = 20

        # Size of buffer to store/average lane-finding section centroid movement in the x-direction. Used to determine
        # the direction of a curve within a frame.
        self.center_move_hist_size = 12

        # Window size used to bucket straight lines found to determine lane boundaries, size in px.
        self.bucket_size = 32
        # Number of lines in a bucket to be considered as a legitimate cluster to determine a lane boundary.
        self.min_lane_line_cluster_size = 5
        # Min/max slope of a line to be considered as a legitimate line for bucketing.
        self.max_lane_slope = 5.0
        self.min_lane_slope = 0.4

        self.lane_line_bucket_cnt = self.width // self.bucket_size + 1
        self.lane_line_bucket_cnt_half = self.lane_line_bucket_cnt // 2

        # Params for the Hough line detector step (based on point representation space).
        #
        # Resolution of the radius param in pixels during Hough line detection.
        self.hough_rho = 1
        # Resolution of the angle (in degrees Pi/180) during Hough line detection.
        self.hough_theta = 1
        # The minimum number of intersections to detect a line.
        self.hough_threshold = 51
        # The minimum number of points that can form a line.
        self.hough_min_line_len = 35
        # The maximum gap between two points to be considered in the same line.
        self.hough_max_line_gap = 16

        # Gradient direction thresholds (y).
        self.y_thresh_min = 18
        self.y_thresh_max = 255

        # Gradient direction thresholds (x).
        self.x_thresh_min = 13
        self.x_thresh_max = 255

        # V (from HSV) color channel thresholds.
        self.v_thresh_min = 220
        self.v_thresh_max = 255

        # S (from HSL) color channel thresholds.
        self.s_thresh_min = 64
        self.s_thresh_max = 255

        # If you need to crop any constant material from the bottom of the scene (hood of car, etc).
        self.bottom_roi_bound = self.height - 20

        # Use this much of the horizon for finding the trapezoid perspective, which assumes the road is straight
        # within this window.
        self.horizon_proportion_straight = 0.75

        # Sensible defaults for horizon and lane boundary positions prior to any processing.
        self.horizon_max = self.height * 0.67
        self.set_horizon(self.halfheight)
        self.right_roi = self.width_minus1
        self.right_roi_min = self.width * 0.67
        self.left_roi = 0
        self.left_roi_max = self.width * 0.33

        # Initialize history buffers for averaging/smoothing.
        self.right_lane_bound_hist = deque([None for i in range(self.lane_frame_hist_size)], self.lane_frame_hist_size)
        self.right_lane_upper_bound_hist = deque([None for i in range(self.lane_frame_hist_size)],
                                                 self.lane_frame_hist_size)
        self.left_lane_bound_hist = deque([None for i in range(self.lane_frame_hist_size)], self.lane_frame_hist_size)
        self.left_lane_upper_bound_hist = deque([None for i in range(self.lane_frame_hist_size)],
                                                self.lane_frame_hist_size)
        self.horizon_hist = deque([None for i in range(self.lane_frame_hist_size)], self.lane_frame_hist_size)
        self.right_lane_slope_hist = deque([None for i in range(self.lane_frame_hist_size)], self.lane_frame_hist_size)
        self.left_lane_slope_hist = deque([None for i in range(self.lane_frame_hist_size)], self.lane_frame_hist_size)
        self.window_centroids_hist = deque([None for i in range(self.lane_frame_hist_size)], self.lane_frame_hist_size)

        # Load pickled calibration params.
        cal_params = pickle.load(open('./calibration_params.p', 'rb'))
        self.cal_params_mtx = cal_params['mtx']
        self.cal_params_dist = cal_params['dist']

        # Lane boundary finding convolution window (size of each segment).
        self.lane_find_window_width = 50
        # For best results, this should divide evenly into image height.
        self.lane_find_window_height = 80
        # How much to slide the lane-finding convolution window left and right to find the next section/block of a lane
        # boundary.
        self.lane_find_margin = 50

        # Proportion of the image height (starting from the bottom) to be considered for finding lane boundary segments.
        self.proportion_image_height_levels = 1.0

        # Find at least this many windows in a lane boundary to be considered.
        self.min_windows_per_lane = 3

        # Define conversions in x and y from pixels space to meters.
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Min expected lane curvature radius (m).
        self.lane_curve_radius_min = 100

        # Threshold by which top view of lane boundaries may diverge to be considered valid.
        self.lane_diverge_max = 0.25

        # For finding a lane segment, there has to be clusters of pixels, or lack of noise.
        self.min_inverse_noise_factor = 100000

        # Maximum difference the the first order curve fit coefficient when right and left lane curve directions
        # differ.
        self.max_curve_direction_coeff_diff = 3

        # Average movement (vector) of windows for the previous frame.
        self.prev_ave_center_move = None

        # Min number of segments (levels) to consider before using an average for centroid movement/curve direction.
        self.min_segments_ave_move = 2

        # Min number of segments (levels) to consider before locking a particular lane curvature direction for a frame.
        self.min_levels_lock_direction = 8
        # Min average movement of lane boundary segments before locking a particular direction for a frame.
        self.min_center_move_keep = 3

        # Method used to combine pipeline of image processing to expose/highlight lane boundaries.
        self.combine_pipeline_method = 'and_related_or_groups'

        self.sample_pipeline = False
        self.frame_cnt = 0

    def set_horizon(self, horizon):
        if 0 <= horizon <= self.horizon_max:
            self.horizon = horizon
            horizon_height = self.height - self.horizon
            self.horizon_tangent_cropped = int(self.horizon + horizon_height * (1 - self.horizon_proportion_straight))

    def set_right_roi(self, roi):
        if self.right_roi_min <= roi <= self.width_minus1:
            self.right_roi = roi

    def set_left_roi(self, roi):
        if 0 <= roi <= self.left_roi_max:
            self.left_roi = roi

    def bucket_line(self, line, buckets):
        """Bucket here, as in verb - throw the line into an appropriate bucket based on where it projects onto the
        bottom of the purview.
        """
        ##at could further bucket lines by slope range as well, or remove outliers in a bucket by slope
        line_props = bounded_line_properties(line, self.height_minus1)
        slope, y_intercept, x_intercept, x_bound_intercept = line_props
        # Cheating here by considering only lines with finite slope, which otherwise should be considered, but they'll
        # start mucking up further calculations, mean, median, and throw exceptions on int(...), etc.
        abs_slope = abs(slope)
        if (abs_slope <= self.max_lane_slope and abs_slope >= self.min_lane_slope and
                0 <= x_bound_intercept < self.width and (
                    x_bound_intercept > self.halfwidth and slope > 0 or
                    x_bound_intercept < self.halfwidth and slope < 0)):
            bucket_idx = int(x_bound_intercept / self.bucket_size)
            buckets[bucket_idx].append((line, slope, int(y_intercept), int(x_intercept), int(x_bound_intercept)))
            return True
        return False

    def closest_cluster(self, buckets):
        """Return the contents of the first group of non-empty buckets (cluster) that satisfy a minimum cluster size."""
        bucket_group = []
        for bucket in buckets:
            if len(bucket) > 0:
                bucket_group.extend(bucket)
            else:
                if len(bucket_group) >= self.min_lane_line_cluster_size:
                    return bucket_group
                bucket_group = []
        return []

    def project_lanes(self, right_lane, left_lane):
        """Given properties of a right and left lane, project them onto the purview with specific endpoints bounded by
        the horizon.

        Update region of interest (roi) properties based on projections.
        """
        right_proj = None
        left_proj = None
        if right_lane is not None and left_lane is not None:
            x_intersect, y_intersect = lines_intersection(right_lane, left_lane)
            if 0 <= x_intersect < self.width and 0 <= y_intersect < self.height:
                self.set_horizon(y_intersect)
                ave_right_x, ave_left_x, ave_horizon = self.smooth_top_points(x_intersect, x_intersect,
                                                                              self.horizon)
                self.set_horizon(ave_horizon)
                right_proj = ((ave_right_x, self.horizon), (right_lane[2], self.height_minus1))
                self.set_right_roi(right_lane[2])
                if ave_right_x > right_lane[2]:
                    self.set_right_roi(ave_right_x)
                left_proj = ((ave_left_x, self.horizon), (left_lane[2], self.height_minus1))
                self.set_left_roi(left_lane[2])
                if ave_left_x < left_lane[2]:
                    self.set_left_roi(ave_left_x)
            else:
                print("intersection out of bounds")
        elif right_lane is not None:
            # Horizon won't change if either lane is missing.
            r_slope, r_y_intercept, r_x_bound = right_lane
            x1 = x_of_slope_intercept(self.horizon, r_y_intercept, r_slope)
            ave_right_x, ave_left_x, ave_horizon = self.smooth_top_points(x1, None, self.horizon)
            self.set_horizon(ave_horizon)
            right_proj = ((ave_right_x, self.horizon), (r_x_bound, self.height_minus1))
            self.set_right_roi(r_x_bound)
            if ave_right_x > r_x_bound:
                self.set_right_roi(ave_right_x)
        elif left_lane is not None:
            # Horizon won't change if either lane is missing.
            l_slope, l_y_intercept, l_x_bound = left_lane
            x1 = x_of_slope_intercept(self.horizon, l_y_intercept, l_slope)
            ave_right_x, ave_left_x, ave_horizon = self.smooth_top_points(None, x1, self.horizon)
            self.set_horizon(ave_horizon)
            left_proj = ((ave_left_x, self.horizon), (l_x_bound, self.height_minus1))
            self.set_left_roi(l_x_bound)
            if ave_left_x > l_x_bound:
                self.set_left_roi(ave_left_x)
        else:
            _, _, ave_horizon = self.smooth_top_points(None, None, self.horizon)
            self.set_horizon(ave_horizon)
        return right_proj, left_proj

    def smooth_lane_hist(self, slope, x_bound, slope_hist, bound_hist):
        """Smooth lanes from frame to frame in the video by averaging over a history of lanes in previous frames."""
        slope_hist.append(slope)
        bound_hist.append(x_bound)
        slope_usable_hist = [e for e in slope_hist if e is not None][-self.lane_frame_smooth_size:]
        bound_usable_hist = [e for e in bound_hist if e is not None][-self.lane_frame_smooth_size:]
        # Reset roi if we're not getting any new signals.
        if not slope_usable_hist:
            self.left_roi = 0
            self.right_roi = self.width_minus1
            self.set_horizon(self.halfheight)
            return None
        med_x_bound = int(statistics.median(bound_usable_hist))
        med_slope = statistics.median(slope_usable_hist)
        y_intercept = int(self.height_minus1 - med_slope * med_x_bound)
        return med_slope, y_intercept, med_x_bound

    def smooth_dim_hist(self, dim, dim_hist):
        """Smooth dimensions from frame to frame in the video by averaging over the history of previous frames."""
        dim_hist.append(dim)
        dim_usable_hist = [e for e in dim_hist if e is not None][-self.horizon_frame_smooth_size:]
        if not dim_usable_hist:
            return None
        med_dim = int(statistics.median(dim_usable_hist))
        return med_dim

    def smooth_top_points(self, x_right, x_left, horizon):
        """Smooth the horizon based on the history of horizon positions."""
        return (self.smooth_dim_hist(x_right, self.right_lane_upper_bound_hist),
                self.smooth_dim_hist(x_left, self.left_lane_upper_bound_hist),
                self.smooth_dim_hist(horizon, self.horizon_hist))

    def extract_trapezoid_src(self, right_proj, left_proj):
        """With the assumption that the lane line boundaries are straight for at least a few meters ahead, cut a
        trapezoid shape from the projected lane lines located previously and return the coordinates.
        """
        r_slope, r_y_intercept, r_x_intercept, r_x_bound_intercept = bounded_line_properties([
                [item for sublist in right_proj for item in sublist]
            ], self.bottom_roi_bound)
        x_right_top = x_of_slope_intercept(self.horizon_tangent_cropped, r_y_intercept, r_slope)
        right_x_dst = int((r_x_bound_intercept + x_right_top) / 2)

        l_slope, l_y_intercept, l_x_intercept, l_x_bound_intercept = bounded_line_properties([
                [item for sublist in left_proj for item in sublist]
            ], self.bottom_roi_bound)
        x_left_top = x_of_slope_intercept(self.horizon_tangent_cropped, l_y_intercept, l_slope)
        left_x_dst = int((l_x_bound_intercept + x_left_top) / 2)
        return np.float32([[x_left_top, self.horizon_tangent_cropped], [x_right_top, self.horizon_tangent_cropped],
                [r_x_bound_intercept, self.bottom_roi_bound], [l_x_bound_intercept, self.bottom_roi_bound]]), \
               np.float32([[left_x_dst, self.horizon_tangent_cropped], [right_x_dst, self.horizon_tangent_cropped],
                [right_x_dst, self.bottom_roi_bound], [left_x_dst, self.bottom_roi_bound]])

    def prep_image_pipeline(self, image):
        """Apply a series of transforms on the given image (frame) to emphasize the lane lines of the road and
        eliminate as much other "noise" as much as possible.
        """

        # Apply the Sobel algorithm to highlight the gradient in both x and y dimensions individually.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        scaled_sobelx = np.uint8(255*sobelx/np.max(sobelx))
        # Threshold the Sobel transform.
        x_thresh = np.zeros_like(scaled_sobelx)
        x_thresh[(scaled_sobelx >= self.x_thresh_min) & (scaled_sobelx <= self.x_thresh_max)] = 255
        # y gradient transforms.
        sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        scaled_sobely = np.uint8(255*sobely/np.max(sobely))
        y_thresh = np.zeros_like(scaled_sobely)
        y_thresh[(scaled_sobely >= self.y_thresh_min) & (scaled_sobely <= self.y_thresh_max)] = 255

        # Isolate the V channel (value) when converting to the HSV color model.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v = hsv[:,:,2]

        # Isolate the S channel (saturation) when converting to the HLS color model.
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        s = hls[:,:,2]

        # Threshold the isolated S and V color channels.
        v_thresh = np.zeros_like(s).astype(np.uint8)
        v_thresh[(v >= self.v_thresh_min) & (v <= self.v_thresh_max)] = 255
        s_thresh = np.zeros_like(s).astype(np.uint8)
        s_thresh[(s >= self.s_thresh_min) & (s <= self.s_thresh_max)] = 255

        # Combine the 4 thresholded transforms (binary image format) according to the configured combination method,
        # which is optimal for the given environment, lighting conditions, shadows, etc.
        combined_c_v = np.zeros_like(v_thresh).astype(np.uint8)
        if self.combine_pipeline_method == 'or_related_and_groups':
            combined_c_v[((v_thresh > 0) | (s_thresh > 0)) & ((x_thresh > 0) | (y_thresh > 0))] = 255
        elif self.combine_pipeline_method == 'and_related_or_groups':
            combined_c_v[((v_thresh > 0) & (s_thresh > 0)) | ((x_thresh > 0) & (y_thresh > 0))] = 255
        return combined_c_v

    def img_roi(self, img_prime):
        """Crop the image to focus on the region-of-interest, which is the lane of interest."""
        left_roi_bound = self.left_roi - self.roi_boundary_fuzz
        if left_roi_bound < 0:
            left_roi_bound = 0
        right_roi_bound = self.right_roi + self.roi_boundary_fuzz
        if right_roi_bound > self.width_minus1:
            right_roi_bound = self.width_minus1
        img = img_prime[self.horizon_tangent_cropped:self.bottom_roi_bound, left_roi_bound:right_roi_bound+1]
        return img, left_roi_bound

    def rough_lane_boundaries(self, img_prime, img, left_roi_bound):
        """Locate approximate lane line boundaries with a triangular boundary of right and left lanes converging to the
        horizon.
        """

        # Use the Hough algo to extract some quality lines from the image to find lane line boundaries.
        lines = cv2.HoughLinesP(img, self.hough_rho, ONE_DEGREE * self.hough_theta, self.hough_threshold, np.array([]),
                                minLineLength=self.hough_min_line_len, maxLineGap=self.hough_max_line_gap)
        if lines is None:
            lines = []

        # First, bucketize the lines based on where they project onto the bottom of the screen.
        line_buckets = [[] for i in range(self.lane_line_bucket_cnt)]
        for line in lines:
            # Translate coords to orig img.
            line[0][0] += left_roi_bound
            line[0][2] += left_roi_bound
            line[0][1] += self.horizon_tangent_cropped
            line[0][3] += self.horizon_tangent_cropped
            for x1, y1, x2, y2 in line:
                self.bucket_line(line, line_buckets)

        # With prominent lanes in the image, the lines should have formed clusters around the lanes. Take the first
        # qualifying cluster immediately to the left and to the right of the center line of the image.
        right_bounding_lines = self.closest_cluster(line_buckets[self.lane_line_bucket_cnt_half:])
        left_bounding_lines = self.closest_cluster(reversed(line_buckets[:self.lane_line_bucket_cnt_half]))
        r_lane_slope = None
        r_lane_lower_x = None
        if right_bounding_lines:
            r_lane_slope, r_lane_lower_x = smooth_line(right_bounding_lines)
        right_lane = self.smooth_lane_hist(r_lane_slope, r_lane_lower_x, self.right_lane_slope_hist,
                                           self.right_lane_bound_hist)

        # For each choice cluster, smooth them out into right and left lane lines.
        l_lane_slope = None
        l_lane_lower_x = None
        if left_bounding_lines:
            l_lane_slope, l_lane_lower_x = smooth_line(left_bounding_lines)
        left_lane = self.smooth_lane_hist(l_lane_slope, l_lane_lower_x, self.left_lane_slope_hist,
                                          self.left_lane_bound_hist)
        return self.project_lanes(right_lane, left_lane)

    def locate_lane_bounds(self, rgb_image):
        """Entry point to locating the immediately bounding lanes of the road."""

        self.frame_cnt += 1
        self.sample_this_pipeline = self.sample_pipeline and self.frame_cnt % 30 == 0

        # RGB -> BGR - normalized for cv2 processing.
        bgr_image = np.flip(rgb_image, 2)

        # Correct image distortion with camera calibration params.
        image = cv2.undistort(bgr_image, self.cal_params_mtx, self.cal_params_dist, None, self.cal_params_mtx)

        # Process the image through the pipeline to enhance lane lines on the road -> image'.
        img_prime = self.prep_image_pipeline(image)

        if self.sample_this_pipeline:
            cv2.imwrite('output_images/raw-{:03d}.jpg'.format(self.frame_cnt), bgr_image)
            cv2.imwrite('output_images/dist-correct-{:03d}.jpg'.format(self.frame_cnt), image)
            cv2.imwrite('output_images/img-prime-{:03d}.jpg'.format(self.frame_cnt), img_prime)

        # Crop image' to focus on the ROI.
        img, left_roi_bound = self.img_roi(img_prime)

        # Leverage code from the first lane-finding project to locate the bounding lane lines, which we assume will be
        # straight for at least a short distance ahead (5-10m).
        right_proj, left_proj = self.rough_lane_boundaries(img_prime, img, left_roi_bound)

        # Bail on this frame if we can't find any hint of lane boundaries.
        if right_proj is None or left_proj is None:
            return rgb_image

        # Cut a section of the lane lines found within an area where the lane is assumed to be straight.
        trapezoid_src, corrected_dst = self.extract_trapezoid_src(right_proj, left_proj)

        # Draw the trapezoid to outline the section of the road assumed to be straight used to transform into a map
        # perspective.
        lane_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        cv2.line(lane_img, tuple(trapezoid_src[0]), tuple(trapezoid_src[1]), [0, 0, 255], 3)
        cv2.line(lane_img, tuple(trapezoid_src[1]), tuple(trapezoid_src[2]), [0, 0, 255], 3)
        cv2.line(lane_img, tuple(trapezoid_src[3]), tuple(trapezoid_src[0]), [0, 0, 255], 3)
        lane_img_overlay = weighted_img(lane_img, image)

        # Use the trapezoid coordinates and extract params to transform to a rectangle for analysis from a bird's eye
        # view or road map perspective.
        M = cv2.getPerspectiveTransform(trapezoid_src, corrected_dst)
        # Also get the inverse params to transform back to the driver's perspective.
        Minv = cv2.getPerspectiveTransform(corrected_dst, trapezoid_src)

        # With the perspective params, warp image' into a map perspective, which should show parallel lane lines.
        parallel_lanes = cv2.warpPerspective(img_prime, M, (img_prime.shape[1], img_prime.shape[0]))
        map_perspective = cv2.warpPerspective(lane_img_overlay, M, (img_prime.shape[1], img_prime.shape[0]))

        if self.sample_this_pipeline:
            cv2.imwrite('output_images/trapezoid-{:03d}.jpg'.format(self.frame_cnt), lane_img_overlay)
            cv2.imwrite('output_images/trapezoid-warped{:03d}.jpg'.format(self.frame_cnt), map_perspective)

        # Follow the curves of the lane lines from the map perspective.
        window_centroids, ave_center_move = self.find_window_centroids(parallel_lanes, corrected_dst[0][0],
                                                                       corrected_dst[1][0])

        parallel_lanes_fit = self.draw_centroid_windows(parallel_lanes, window_centroids)

        # Derive the y coordinates of the lane positions (centroids) for each level.
        window_centers_y = np.arange(parallel_lanes.shape[0] - (self.lane_find_window_height/2), 0,
                                     -self.lane_find_window_height)
        # Fit centroids of each level onto a curved line.
        usable_l_centroids, usable_r_centroids, left_fit, right_fit, parallel_lanes_fit = \
            self.fit_centroid_lines(parallel_lanes, window_centroids, window_centers_y, ave_center_move,
                                    parallel_lanes_fit)

        # Validate and smooth the fit lane boundaries.
        fit_norm = self.normalize_fit_lanes(window_centers_y, window_centroids, usable_l_centroids, usable_r_centroids,
                                            left_fit, right_fit)
        overlay_lanes = image
        if fit_norm is not None:
            left_fit_norm, right_fit_norm, left_curve_rad = fit_norm
            overlay_lanes_fit, left_fitx, right_fitx = driver_perspective(image, left_fit_norm, right_fit_norm, Minv)
            overlay_lanes = weighted_img(overlay_lanes_fit, image)

            # Calculate the position of the car relative to the center of the lane.
            lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
            car_lane_offset = (lane_center - image.shape[1] / 2) * self.xm_per_pix
            offset_side = 'left'
            if car_lane_offset <= 0:
                offset_side = 'right'
            # Notate the radius and position info on the image.
            left_curve_rad_str = 'UNKNOWN'
            if left_curve_rad is not None:
                left_curve_rad_str = round(left_curve_rad, 3)
            cv2.putText(overlay_lanes, 'Curvature radius: {}m'.format(left_curve_rad_str), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(overlay_lanes, 'Vehicle is {}m {} of center'.format(abs(round(car_lane_offset, 3)), offset_side),
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            if self.sample_this_pipeline:
                cv2.imwrite('output_images/levels-{:03d}.jpg'.format(self.frame_cnt), parallel_lanes_fit)
                cv2.imwrite('output_images/drivers-lane-{:03d}.jpg'.format(self.frame_cnt), overlay_lanes)

        # Return to RGB.
        return np.flip(overlay_lanes, 2)

    def draw_centroid_windows(self, parallel_lanes, window_centroids):
        """Draw centroid windows for processing visualization."""
        window_width = self.lane_find_window_width
        window_height = self.lane_find_window_height
        # Points used to draw all the left and right windows.
        l_points = np.zeros_like(parallel_lanes)
        r_points = np.zeros_like(parallel_lanes)

        # Go through each level and draw the windows.
        for level in range(0, len(window_centroids)):
            if window_centroids[level][0] is not np.nan:
                l_mask = window_mask(window_width, window_height, parallel_lanes, window_centroids[level][0], level)
                # Add graphic points from window mask here to total pixels found.
                l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            if window_centroids[level][1] is not np.nan:
                r_mask = window_mask(window_width, window_height, parallel_lanes, window_centroids[level][1], level)
                r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        # Add both left and right window pixels together.
        centroid_windows = np.array(r_points+l_points, np.uint8)
        # Create a zero color channel.
        zero_channel = np.zeros_like(centroid_windows)
        # Color the window pixels.
        centroid_windows = np.array(cv2.merge((zero_channel, centroid_windows, zero_channel)), np.uint8)
        # Making the original road pixels 3 color channels.
        parallel_lanes_color = np.dstack((parallel_lanes, parallel_lanes, parallel_lanes))*255
        # Overlay the orignal road image with window results.
        return cv2.addWeighted(parallel_lanes_color, 1, centroid_windows, 0.5, 0.0)

    def fit_centroid_lines(self, parallel_lanes, window_centroids, window_centers_y, ave_center_move,
                           parallel_lanes_fit):
        """Fit centroid positions of each level onto a polynomial curve. One line for each right and left lane
        boundary.
        """
        # Assign an array of all y points on the map perspective image.
        yvals = range(0, parallel_lanes.shape[0])
        left_fit = None
        left_fitx = None
        right_fit = None
        right_fitx = None
        usable_l_centroids = np.array(
            [[tup[0][0], tup[1]] for tup in zip(window_centroids, window_centers_y) if tup[0][0] is not np.nan])
        if len(usable_l_centroids) >= self.min_windows_per_lane:
            left_fit = np.polyfit(usable_l_centroids[:,1], usable_l_centroids[:,0], 2)
            left_fitx = np.array(left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2], dtype=np.int32)
        usable_r_centroids = np.array(
            [[tup[0][1], tup[1]] for tup in zip(window_centroids, window_centers_y) if tup[0][1] is not np.nan])
        if len(usable_r_centroids) >= self.min_windows_per_lane:
            right_fit = np.polyfit(usable_r_centroids[:,1], usable_r_centroids[:,0], 2)
            right_fitx = np.array(right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2], dtype=np.int32)

        if left_fit is not None and right_fit is not None:
            self.prev_ave_center_move = ave_center_move
            left_fit_lane = np.array(list(zip(np.concatenate((left_fitx-3, left_fitx[::-1]+3), axis=0),
                                              np.concatenate((yvals, yvals[::-1]), axis=0))), dtype=np.int32)
            right_fit_lane = np.array(list(zip(np.concatenate((right_fitx-3, right_fitx[::-1]+3), axis=0),
                                              np.concatenate((yvals, yvals[::-1]), axis=0))), dtype=np.int32)
            road_top_view = np.zeros_like(parallel_lanes_fit)
            cv2.fillPoly(parallel_lanes_fit, [left_fit_lane], color=[255,0,0])
            cv2.fillPoly(parallel_lanes_fit, [right_fit_lane], color=[0,0,255])
            parallel_lanes_fit = weighted_img(parallel_lanes_fit, road_top_view)
        else:
            self.prev_ave_center_move = None

        return usable_l_centroids, usable_r_centroids, left_fit, right_fit, parallel_lanes_fit

    def level_lane_position(self, centroid_pair, image, lane_center_x, level, prev_found_center, prev_found_level,
                            ave_center_move, found_move_cnt, found_center_move, effective_ave_center_move, conv_signal,
                            keep_curve_direction):
        """Find the likely position (if any) of a lane line using conv_signal for this level and the previous position
        of this lane boundary.
        """

        window_width = self.lane_find_window_width
        margin = self.lane_find_margin

        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center
        # of window.
        offset = window_width//2

        # Account for the possibility that the lane has curved off to one side of the frame.
        if lane_center_x is np.nan:
            centroid_pair.append(lane_center_x)
        else:
            # Locate the area of interest (window + margin) where the lane line is likely to be.
            min_index = int(min(max(lane_center_x+offset-margin+effective_ave_center_move, 0),
                                    image.shape[1]-window_width))
            max_index = int(max(min(lane_center_x+offset+margin+effective_ave_center_move, image.shape[1]),
                                    window_width))
            inverse_noise_factor = np.max(conv_signal[min_index:max_index]) - np.min(conv_signal[min_index:max_index])

            # The new position captures the most pixel density in the area of interest.
            new_center = np.argmax(conv_signal[min_index:max_index]) + min_index - offset
            center_move = new_center - lane_center_x

            # Consider this new position only if there is sufficient density focus within the area (enough signal
            # to noise ratio).
            if inverse_noise_factor < self.min_inverse_noise_factor or (keep_curve_direction and
                    center_move * ave_center_move < 0):
                # Weak noise to signal ratio, assume the lane line position based on the previous level's position.
                if level == 0:
                    centroid_pair.append(lane_center_x)
                else:
                    centroid_pair.append(np.nan)
                    lane_center_x += effective_ave_center_move
                    if lane_center_x > image.shape[1] - 1:
                        lane_center_x = np.nan
                    elif lane_center_x < 0:
                        lane_center_x = np.nan
            else:
                # A likely position of the lane line was found, update aggregate lane state, curvature, etc.
                found_center_move.append(new_center - prev_found_center)
                found_move_cnt += level - prev_found_level
                prev_found_level = level
                ave_center_move = int(np.sum(found_center_move) / found_move_cnt)
                lane_center_x = prev_found_center = new_center
                centroid_pair.append(lane_center_x)

        return lane_center_x, prev_found_center, prev_found_level, ave_center_move, found_move_cnt

    def find_window_centroids(self, image, l_lane_center_x, r_lane_center_x):
        """Follow the curves of the lane lines from a map perspective starting from the bottom positions
        (l_lane_center_x, r_lane_center_x).

        The image will be cut into horizontal slices (levels) and for each lane boundary line (right and left) within a
        level, a convolution window will be passed over a sum of pixel density within an area where the lane line
        position is likely to be found with some margin to accommodate lane evolution from frame to frame.
        """

        if l_lane_center_x is None or r_lane_center_x is None:
            return [], []

        # Convolution window properties.
        window_width = self.lane_find_window_width
        window_height = self.lane_find_window_height
        margin = self.lane_find_margin

        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center
        # of window.
        offset = window_width//2

        # Store the (left, right) window centroid positions per level.
        window_centroids = []
        # Create our window template that we will use for convolutions.
        window = np.ones(window_width)

        # Add what we found for the first level.
        window_centroids.append([l_lane_center_x, r_lane_center_x])
        initial_lane_width = r_lane_center_x - l_lane_center_x

        # Keep some state between each level to find aggregate properties such as curve direction of the lane lines.
        #false_climb_steps = 0
        prev_l_found_center = l_lane_center_x
        prev_r_found_center = r_lane_center_x
        prev_l_found_level = 0
        prev_r_found_level = 0
        ave_center_move = 0
        found_move_cnt = 0
        found_center_move = deque([], self.center_move_hist_size)

        # Go through each level looking for max pixel clusters.
        for level in range(1, int(image.shape[0] * self.proportion_image_height_levels) // window_height):
            centroid_pair = []

            # Track average movement of lane line position movement along the x axis, which gives a sense of curvature
            # sharpness and direction.
            effective_ave_center_move = ave_center_move
            keep_curve_direction = False
            if level < self.min_segments_ave_move and self.prev_ave_center_move is not None:
                # For the first few levels, use the average movement from the previous frame.
                ##at should also consider valid levels where centroids were actually found (not nan)
                effective_ave_center_move = self.prev_ave_center_move
            elif level >= self.min_levels_lock_direction-1:
                # Lock onto a direction to stabilize lane-finding in noisy signals.
                if abs(ave_center_move) >= self.min_center_move_keep:
                    keep_curve_direction = True

            # Summarize pixel density of the image over this convolution window with a 1D result.
            image_level_sum = np.sum(image[int(image.shape[0]-(level+1)*window_height): \
                                 int(image.shape[0]-level*window_height),:], axis=0)

            # Convolve the template with the pixel density output to produce a signal used to find the lane line
            # boundaries.
            conv_signal = np.convolve(window, image_level_sum)

            # Starting from the previous level's lane line locations (l_lane_center_x and r_lane_center_x), determine
            # the likely locations for lane lines on this level using the convolved signal.

            l_lane_center_x, prev_l_found_center, prev_l_found_level, ave_center_move, found_move_cnt = \
                self.level_lane_position(centroid_pair, image, l_lane_center_x, level, prev_l_found_center,
                                         prev_l_found_level, ave_center_move, found_move_cnt, found_center_move,
                                         effective_ave_center_move, conv_signal, keep_curve_direction)

            r_lane_center_x, prev_r_found_center, prev_r_found_level, ave_center_move, found_move_cnt = \
                self.level_lane_position(centroid_pair, image, r_lane_center_x, level, prev_r_found_center,
                                         prev_r_found_level, ave_center_move, found_move_cnt, found_center_move,
                                         effective_ave_center_move, conv_signal, keep_curve_direction)

            # Finally, force some symmetry/parellelism in the lane lines if one of the boundary positions was not found
            # for this level.
            if centroid_pair[0] is np.nan and l_lane_center_x is not np.nan and centroid_pair[1] is not np.nan:
                centroid_pair[0] = l_lane_center_x
            elif centroid_pair[0] is not np.nan and centroid_pair[1] is np.nan and r_lane_center_x is not np.nan:
                centroid_pair[1] = r_lane_center_x

            window_centroids.append(centroid_pair)
        return window_centroids, ave_center_move

    def validate_fit_lanes(self, window_centers_y, window_centroids, usable_l_centroids, usable_r_centroids, left_fit,
                           right_fit):
        """Apply some sanity checks to the lane boundary curves."""

        # Verify curves are in the same direction.
        if left_fit[0] * right_fit[0] < 0 and abs(left_fit[0] - right_fit[0]) > self.max_curve_direction_coeff_diff:
            return None

        # Verify the lanes are consistently the same distance apart and don't diverge.
        #
        # The first set is always defined.
        start_l, start_r = window_centroids[0]
        expected_width = start_r - start_l
        l_center_keep = start_l
        r_center_keep = start_r
        for level in range(1, len(window_centroids)):
            l_center, r_center = window_centroids[level]
            if l_center is not np.nan:
                l_center_keep = l_center
            if r_center is not np.nan:
                r_center_keep = r_center
            lane_diverge = abs(expected_width - (r_center_keep - l_center_keep)) / expected_width
            if lane_diverge > self.lane_diverge_max:
                return None

        # Calculate the curvature radius using a conversion of pixels/actual distance, then verify it is reasonable.
        left_fit_cr = np.polyfit([e*self.ym_per_pix for e in usable_l_centroids[:,1]],
                                 [e*self.xm_per_pix for e in usable_l_centroids[:,0]], 2)
        lane_curve_start_y = window_centers_y[3]
        left_curve_rad = ((1 + (2*left_fit_cr[0]*lane_curve_start_y*self.ym_per_pix + left_fit_cr[1])**2)**1.5) / \
            np.absolute(2*left_fit_cr[0])
        if left_curve_rad < self.lane_curve_radius_min:
            return None

        return left_curve_rad

    def normalize_fit_lanes(self, window_centers_y, window_centroids, usable_l_centroids, usable_r_centroids, left_fit,
                            right_fit):
        """Validate and smooth the lines that fit the centroid positions for each right and left lane boundary."""
        if left_fit is None or right_fit is None:
            return None
        
        # Apply some sanity checks to the lane boundaries.
        left_curve_rad = self.validate_fit_lanes(window_centers_y, window_centroids, usable_l_centroids,
                                                 usable_r_centroids, left_fit, right_fit)
        if left_curve_rad is not None:
            self.window_centroids_hist.append(window_centroids)
        else:
            self.window_centroids_hist.append(None)

        # Find an average set of line properties based on a history of lane boundary lines.
        centroid_usable_hist = [e for e in self.window_centroids_hist if e is not None]
        if not centroid_usable_hist:
            return None
        ave_window_centroids = np.nanmedian(centroid_usable_hist[-self.lane_fit_smooth_size:], axis=0)
        usable_ave_l_centroids = np.array(
            [[tup[0][0], tup[1]] for tup in zip(ave_window_centroids, window_centers_y) if not np.isnan(tup[0][0])])
        if len(usable_ave_l_centroids) < self.min_windows_per_lane:
            return None
        left_fit_ave = np.polyfit(usable_ave_l_centroids[:,1], usable_ave_l_centroids[:,0], 2)
        usable_ave_r_centroids = np.array(
            [[tup[0][1], tup[1]] for tup in zip(ave_window_centroids, window_centers_y) if not np.isnan(tup[0][1])])
        if len(usable_ave_r_centroids) < self.min_windows_per_lane:
            return None
        right_fit_ave = np.polyfit(usable_ave_r_centroids[:,1], usable_ave_r_centroids[:,0], 2)
        return left_fit_ave, right_fit_ave, left_curve_rad

class VehicleCollisionZone(object):
    """Track positions of vehicles that share the road in the immediate vicinity."""

    def __init__(self):
        # HOG color model.
        self.hog_cspace = HOG_CSPACE

        # Number of HOG orientation buckets.
        self.hog_orientations = HOG_ORIENTATIONS

        # HOG pixels per cell.
        self.pix_per_cell = PIX_PER_CELL

        # HOG cells per block.
        self.cell_per_block = CELL_PER_BLOCK

        # Spatial binning dimensions.
        self.spatial_size = SPATIAL_SIZE

        # Number of histogram bins for color histogram features.
        self.hist_bins = HIST_BINS

        # Step distance between sliding windows used to predict vehicle presence.
        self.cells_per_step = 2

        # Thresholds of heatmaps generated from areas of high probability to contain a vehicle.
        # Generation 1 threshold.
        self.heatmap_threshold_gen1 = 2
        # Generation 2 threshold.
        self.heatmap_threshold_gen2 = 15

        # Size of history buffer to store vehicle detection tracking history.
        self.vehicle_matches_hist_size = 50

        # Initialize history buffers for averaging/smoothing.
        self.vehicle_matches_hist = deque([None for i in range(self.vehicle_matches_hist_size)],
                                          self.vehicle_matches_hist_size)

        # Overlap margin allowed for two adjacent areas to be considered the same object (in pixels).
        self.overlap_margin = 25
 
        # Load pickled classifier and training params.
        svc_training_params = pickle.load(open('svc_trained.p', 'rb'))
        self.svc = svc_training_params["svc"]
        self.X_scaler = svc_training_params["scaler"]

    def adjacent_areas(self, area_51, area_52):
        """Return true if area_51 and area_52 are close enough to be considered part of the same object, false
        otherwise."""
        (x1_51, y1_51), (x2_51, y2_51) = area_51
        return areas_overlap((np.subtract(area_51[0], self.overlap_margin), np.add(area_51[0], self.overlap_margin)),
                             area_52)

    def merge_bordering_areas(self, areas):
        """Return a set of merged input areas (rectangular coordinates). Two or more areas will be merged if they
        overlay or are close enough in proximity. If any two areas are merged, the process will repeat (recursively)
        since any other area may now be considered close enough to merge into the newly merged area."""
        merged_areas = areas[:]

        # Check every combination of the given list of areas.
        for i, match_area_i in enumerate(areas):
            merged_areas_stage = []
            for j, match_area_j in enumerate(merged_areas):
                if area_contains(match_area_i, match_area_j):
                    # If match_area_i contains match_area_j, disregard match_area_j.
                    continue
                if self.adjacent_areas(match_area_i, match_area_j):
                    # If the areas border each other, merge them.
                    merged_areas_stage.append(merge_areas(match_area_i, match_area_j))
                    merged_areas_stage.extend(merged_areas[j+1:])
                    # Recursively re-check every possible combination with the newly merged area, which may now
                    # overlap/border previously checked areas.
                    return self.merge_bordering_areas(merged_areas_stage)
                else:
                    # If the areas are unrelated, include both of them in the result set separately.
                    merged_areas_stage.append(match_area_j)
            merged_areas_stage.append(match_area_i)
            merged_areas = merged_areas_stage
        return merged_areas

    def search_frame(self, img):
        """Search the frame for areas matching car-like features using the trained classifier."""
        vehicle_match_areas = []

        # Check the bottom area of each frame with sliding windows using a larger scale since closer objects will
        # appear larger.
        ystart = 500
        ystop = 650
        scale = 2
        vehicle_match_areas.extend(match_vehicle_areas(img, ystart, ystop, scale, self.svc, self.X_scaler,
                                                       self.hog_orientations, self.pix_per_cell, self.cell_per_block,
                                                       self.spatial_size, self.hist_bins, self.cells_per_step,
                                                       self.hog_cspace))

        # Check the mid-to-horizon area of each frame with sliding windows of a smaller scale.
        ystart = 400
        ystop = 530
        scale = 1.5

        vehicle_match_areas.extend(match_vehicle_areas(img, ystart, ystop, scale, self.svc, self.X_scaler,
                                                       self.hog_orientations, self.pix_per_cell, self.cell_per_block,
                                                       self.spatial_size, self.hist_bins, self.cells_per_step,
                                                       self.hog_cspace))
        return vehicle_match_areas

    def locate_nearby_cars(self, img):
        """Entry point for detecting nearby cars."""
        
        # The list of areas of this frame that resemble a car.
        vehicle_match_areas = self.search_frame(img)

        # Extract the first generation (short-term) areas of the frame with a minimum amount of positive matching
        # activity from the classifier.
        heatmap_gen1, vehicle_feature_matches = represent_heatmap(img, vehicle_match_areas, self.heatmap_threshold_gen1)

        # Add the matching areas of this frame to a history buffer.
        self.vehicle_matches_hist.append(vehicle_feature_matches)

        # Use the matching areas over a history of frames to reveal those areas that match vehicle
        # features persistently over time.
        areas_usable_hist_wound = [e for e in self.vehicle_matches_hist if e is not None]
        areas_usable_hist = list(itertools.chain.from_iterable(areas_usable_hist_wound))

        # Extract the second generation (persistent) areas of the video where cars are more likely to be found.
        heatmap_gen2, persistent_matches = represent_heatmap(img, areas_usable_hist, self.heatmap_threshold_gen2)

        # Merge persistent matching areas that overlap or border each other into a single area.
        persistent_matches_merged = self.merge_bordering_areas(persistent_matches)
        return persistent_matches_merged
