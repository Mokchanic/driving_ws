#!/usr/bin/env python
# -*- coding: utf-8 -*-

#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import numpy as np
import cv2, math
import rospy, rospkg, time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from xycar_msgs.msg import xycar_motor
from math import *
import signal
import sys
import os
import random
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sensor_msgs.msg import Imu

#Camera_calib
mtx = np.array([[1.15396093e+03, 0.00000000e+00, 6.69705357e+02],
                [0.00000000e+00, 1.14802496e+03, 3.85656234e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist = np.array([[-2.41017956e-01, -5.30721173e-02, -1.15810355e-03,
                -1.28318856e-04,  2.67125290e-02]])

#=============================================
# 터미널에서 Ctrl-C 키입력으로 프로그램 실행을 끝낼 때
# 그 처리시간을 줄이기 위한 함수
#=============================================

def signal_handler(sig, frame):
    import time
    time.sleep(3)
    os.system('killall -9 python rosout')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
image = np.empty(shape=[0]) # 카메라 이미지를 담을 변수
bridge = CvBridge() 
motor = None # 모터 토픽을 담을 변수

#=============================================
# 프로그램에서 사용할 상수 선언부
#=============================================
CAM_FPS = 30    # 카메라 FPS - 초당 30장의 사진을 보냄
WIDTH, HEIGHT = 640, 480    # 카메라 이미지 가로x세로 크기

#=============================================
# 콜백함수 - 카메라 토픽을 처리하는 콜백함수
# 카메라 이미지 토픽이 도착하면 자동으로 호출되는 함수
# 토픽에서 이미지 정보를 꺼내 image 변수에 옮겨 담음.
#=============================================
def img_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")

#=============================================
# 모터 토픽을 발행하는 함수  
# 입력으로 받은 angle과 speed 값을 
# 모터 토픽에 옮겨 담은 후에 토픽을 발행함.
#=============================================
def drive(angle, speed):

    global motor

    motor_msg = xycar_motor()
    motor_msg.angle = angle
    motor_msg.speed = speed

    motor.publish(motor_msg)

#=============================================
# 실질적인 메인 함수 
# 카메라 토픽을 받아 각종 영상처리와 알고리즘을 통해
# 차선의 위치를 파악한 후에 조향각을 결정하고,
# 최종적으로 모터 토픽을 발행하는 일을 수행함. 
#=============================================

#====== 이미지 전처리 ==========
def preprocess_image(img):
    ysize = img.shape[0] # 480
    xsize = img.shape[1] # 640
    
    # undistortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    # perspective transformation
    src = np.float32([
        (410,300),
        (226,300),
        (19.48,400),
        (606.49,400)
        
        
        #(420,275),
        #(200,275),
        #(40,400),
        #(600,400)
        
        # (150,350),
        # (500,350),    
        # (610,400),
        # (30,400) 
    ])

    dst = np.float32([
        #(xsize - 175, 0),
        #(175, 0),
        #(175, ysize),
        #(xsize - 175, ysize)
        
        (xsize - 175, 0),
        (175, 0),
        (175, ysize),
        (xsize - 175, ysize)
        # (0,0),
        # (560 - 1, 0),
        # (560 - 1, 125 - 1),
        # (0, 125 -1)

    ])
    
    M = cv2.getPerspectiveTransform(src, dst)
    invM = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist, M, (xsize, ysize), flags=cv2.INTER_LINEAR)

    # roi crop
    vertices = np.array([ # region for crop
        [100, ysize],
        [100, 0],
        [540, 0],
        [540, ysize]
    ])
    
    vertices = np.array(vertices, ndmin=3, dtype=np.int32)

    if len(img.shape) == 3: # (row, column , channel)
        fill_color = (255,) * 3
    else:
        fill_color = 255
    
    mask = np.zeros_like(warped)
    mask = cv2.fillPoly(mask, vertices, fill_color) # mask에서 vertices 모양을 fill_color로 채움

    roi = cv2.bitwise_and(warped, mask) # img에서 mask와 비트연선으로 사진 자르기

    # cv2.imshow("CAM View", roi)
    # cv2.waitKey(1)
    
    return roi

def get_bird_eye(img):
    return preprocess_image(img)

def binary_threshold(img, low, high):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    else:
        pass
    
    if len(img.shape) == 2:
        output = np.zeros_like(img)
        mask = (img >= low) & (img <= high)
        
    elif len(img.shape) == 3:
        output = np.zeros_like(img[:,:,0])
        mask = (img[:,:,0] >= low[0]) & (img[:,:,0] <= high[0]) \
            & (img[:,:,1] >= low[1]) & (img[:,:,1] <= high[1]) \
            & (img[:,:,2] >= low[2]) & (img[:,:,2] <= high[2])
            
    output[mask] = 1
    return output

def get_binary_image(img):
    # 흰색, 노란색 정보만 바이너리 이미지로 추출
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    rgb = img[:,:,:]
    
    # color space
    #### LAB ####
    L = lab[:,:,0]
    L_max, L_mean = np.max(L), np.mean(L)
    B = lab[:,:,2]
    B_max, B_mean = np.max(B), np.mean(B)

    # YELLOW
    L_adapt_yellow = max(80, int(L_max * 0.45))
    B_adapt_yellow =  max(int(B_max * 0.70), int(B_mean * 1.2))
    lab_low_yellow = np.array((L_adapt_yellow, 120, B_adapt_yellow))
    lab_high_yellow = np.array((255, 145, 255))

    lab_yellow = binary_threshold(lab, lab_low_yellow, lab_high_yellow)
    lab_binary = lab_yellow

    # lab_binary = PIL.Image.fromarray(lab_binary)

    # cv2.imshow(lab_binary, cmap='gray')

    
    #### HSV ####
    H = hsv[:,:,0]
    H_max, H_mean = np.max(H), np.mean(H)
    S = hsv[:,:,1]
    S_max, S_mean = np.max(S), np.mean(S)
    V = hsv[:,:,2]
    V_max, V_mean = np.max(V), np.mean(V)
    # YELLOW
    S_adapt_yellow =  max(int(S_max * 0.25), int(S_mean * 1.75))
    V_adapt_yellow =  max(50, int(V_mean * 1.25))
    hsv_low_yellow = np.array((15, S_adapt_yellow, V_adapt_yellow))
    hsv_high_yellow = np.array((30, 255, 255))
    hsv_yellow = binary_threshold(hsv, hsv_low_yellow, hsv_high_yellow)    
    # WHITE
    V_adapt_white = max(150, int(V_max * 0.8),int(V_mean * 1.25))
    hsv_low_white = np.array((0, 0, V_adapt_white))
    hsv_high_white = np.array((255, 40, 220))

    hsv_white = binary_threshold(hsv, hsv_low_white, hsv_high_white)

    hsv_binary = hsv_yellow | hsv_white
    
    #### HLS ####
    L = hls[:,:,1]
    L_max, L_mean = np.max(L), np.mean(L)
    S = hls[:,:,2]
    S_max, S_mean = np.max(S), np.mean(S)
    # YELLOW
    L_adapt_yellow = max(80, int(L_mean * 1.25))
    S_adapt_yellow = max(int(S_max * 0.25), int(S_mean * 1.75))
    hls_low_yellow = np.array((15, L_adapt_yellow, S_adapt_yellow))
    hls_high_yellow = np.array((30, 255, 255))

    hls_yellow = binary_threshold(hls, hls_low_yellow, hls_high_yellow)
    # WHITE
    L_adapt_white =  max(160, int(L_max *0.8),int(L_mean * 1.25))
    hls_low_white = np.array((0, L_adapt_white,  0))
    hls_high_white = np.array((255, 255, 255))

    hls_white = binary_threshold(hls, hls_low_white, hls_high_white)
        
    hls_binary = hls_yellow | hls_white
    
    #### RGB ####
    R = rgb[:,:,0]
    R_max, R_mean = np.max(R), np.mean(R)
    # WHITE
    R_low_white = min(max(150, int(R_max * 0.55), int(R_mean * 1.95)),230)
    R_binary = binary_threshold(R, R_low_white, 255)
    
    
    ### Adaptive thresholding: Gaussian kernel 
    # YELLOW
    
    adapt_yellow_S = cv2.adaptiveThreshold(hls[:,:,2], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 161, -5)
    adapt_yellow_B = cv2.adaptiveThreshold(lab[:,:,2], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 161, -5)
    adapt_yellow = adapt_yellow_S & adapt_yellow_B
    
    # WHITE
    adapt_white_R = cv2.adaptiveThreshold(img[:,:,0], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 161, -27)
    adapt_white_L = cv2.adaptiveThreshold(hsv[:,:,2], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 161, -27)
    adapt_white = adapt_white_R & adapt_white_L
    adapt_binary =  adapt_yellow | adapt_white
    
    ### Ensemble Voting
    combined = np.asarray(R_binary + lab_binary + hls_binary + hsv_binary + adapt_binary, dtype=np.uint8)

    combined[combined < 3] = 0
    combined[combined >= 3] = 1

    return  combined

def get_poly_points(left_fit, right_fit):
    '''
    Get the points for the left lane/ right lane defined by the polynomial coeff's 'left_fit'
    and 'right_fit'
    :param left_fit (ndarray): Coefficients for the polynomial that defines the left lane line
    :param right_fit (ndarray): Coefficients for the polynomial that defines the right lane line
    : return (Tuple(ndarray, ndarray, ndarray, ndarray)): x-y coordinates for the left and right lane lines
    '''
    ysize, xsize = IMG_SHAPE
    
    # Get the points for the entire height of the image
    plot_y = np.linspace(0, ysize-1, ysize)

    plot_xleft = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
    plot_xright = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]

    # But keep only those points that lie within the image
    plot_xleft = plot_xleft[(plot_xleft >= 0) & (plot_xleft <= xsize - 1)]
    plot_xright = plot_xright[(plot_xright >= 0) & (plot_xright <= xsize - 1)]
    plot_yleft = np.linspace(ysize - len(plot_xleft), ysize - 1, len(plot_xleft))
    plot_yright = np.linspace(ysize - len(plot_xright), ysize - 1, len(plot_xright))
    
    return plot_xleft.astype(int), plot_yleft.astype(int), plot_xright.astype(int), plot_yright.astype(int)

def check_validity(left_fit, right_fit, diagnostics=False):
    '''
    Determine the validity of lane lines represented by a set of second order polynomial coefficients 
    :param left_fit (ndarray): Coefficients for the 2nd order polynomial that defines the left lane line
    :param right_fit (ndarray): Coefficients for the 2nd order polynomial that defines the right lane line
    :param diagnostics (boolean): Boolean flag for logging
    : return (boolean)
    '''

    if left_fit is None or right_fit is None:
        return False

    plot_xleft, plot_yleft, plot_xright, plot_yright = get_poly_points(left_fit, right_fit)

    # Check whether the two lines lie within a plausible distance from one another for three distinct y-values

    y1 = IMG_SHAPE[0] - 1 # Bottom
    y2 = IMG_SHAPE[0] - int(min(len(plot_yleft), len(plot_yright)) * 0.35) # For the 2nd and 3rd, take values between y1 and the top-most available value.
    y3 = IMG_SHAPE[0] - int(min(len(plot_yleft), len(plot_yright)) * 0.75)

    # Compute the respective x-values for both lines
    x1l = left_fit[0]  * (y1**2) + left_fit[1]  * y1 + left_fit[2]
    x2l = left_fit[0]  * (y2**2) + left_fit[1]  * y2 + left_fit[2]
    x3l = left_fit[0]  * (y3**2) + left_fit[1]  * y3 + left_fit[2]

    x1r = right_fit[0] * (y1**2) + right_fit[1] * y1 + right_fit[2]
    x2r = right_fit[0] * (y2**2) + right_fit[1] * y2 + right_fit[2]
    x3r = right_fit[0] * (y3**2) + right_fit[1] * y3 + right_fit[2]

    # Compute the L1 norms
    x1_diff = abs(x1l - x1r)
    x2_diff = abs(x2l - x2r)
    x3_diff = abs(x3l - x3r)

    # Define the threshold values for each of the three points
    min_dist_y1 = 480 # 510 # 530 
    max_dist_y1 = 730 # 750 # 660
    min_dist_y2 = 280
    max_dist_y2 = 730 # 660
    min_dist_y3 = 140
    max_dist_y3 = 730 # 660

    if (x1_diff < min_dist_y1) | (x1_diff > max_dist_y1) | \
        (x2_diff < min_dist_y2) | (x2_diff > max_dist_y2) | \
        (x3_diff < min_dist_y3) | (x3_diff > max_dist_y3):
        if diagnostics:
            print("Violated distance criterion: " + "x1_diff == {:.2f}, x2_diff == {:.2f}, x3_diff == {:.2f}".format(x1_diff, x2_diff, x3_diff))
        return False

    # Check whether the line slopes are similar for two distinct y-values
    # x = Ay**2 + By + C
    # dx/dy = 2Ay + B

    y1left_dx  = 2 * left_fit[0]  * y1 + left_fit[1]
    y3left_dx  = 2 * left_fit[0]  * y3 + left_fit[1]
    y1right_dx = 2 * right_fit[0] * y1 + right_fit[1]
    y3right_dx = 2 * right_fit[0] * y3 + right_fit[1]

    # Compute the L1-norm
    norm1 = abs(y1left_dx - y1right_dx)
    norm2 = abs(y3left_dx - y3right_dx)

def polyfit_sliding_window(binary, lane_width_px=578, diagnostics=False):
    global cache
    ret = True
    
    # Sanity check
    if binary.max() <= 0:
        return False, np.array([]), np.array([]), np.array([])
    
    #### histogram ####
    histogram = None
    cutoff = int(binary.shape[0] / 2)
    
    histogram = np.sum(binary[cutoff:, :], axis=0)
    
    if histogram.max() == 0:
        print('Unable to detect lane lines in this frame. Trying another frame!')
        return False, np.array([]), np.array([])
    
    mid_point = int(histogram.shape[0] / 2)
    left_x_base = np.argmax(histogram[:mid_point])
    right_x_base = np.argmax(histogram[mid_point:]) + mid_point
    
    
    #### sliding window ####
    out = np.dstack((binary, binary, binary)) * 255 # channel 3개로 차원을 늘려준것과 동일
    
    nb_windows = 30 # number of sliding windows
    margin = 10 # width of the windows +/- margin
    minpix = 10 # min number of pixels needed to recenter the window
    window_height = int(IMG_SHAPE[0] / nb_windows)
    min_lane_pts = 5  # min number of 'hot' pixels needed to fit a 2nd order polynomial as a 
                    # lane line
    
    # Identify the x-y positions of all nonzero pixels in the image
    # Note: the indices here are equivalent to the coordinate locations of the
    # pixel
    nonzero = binary.nonzero()
    nonzero_x = np.array(nonzero[1])
    nonzero_y = np.array(nonzero[0])
    
    # Current positions to be updated for each window
    left_x_current = left_x_base
    right_x_current = right_x_base

    # Empty lists to receive left and right lane pixel indices
    left_lane_indxs = []
    right_lane_indxs = []
    
    for window in range(nb_windows):
        # make window
        # 상자 높이 y
        win_y_low = IMG_SHAPE[0] - (1 + window) * window_height
        win_y_high = IMG_SHAPE[0] - window * window_height
        
        # 왼쪽 차선에 대한 상자 폭 x 
        win_left_x_low = left_x_current - margin
        win_left_x_high = left_x_current + margin
        
        # 오른쪽 차선에 대한 상자 폭 x
        win_right_x_low = right_x_current - margin
        win_right_x_high = right_x_current + margin
        
        # 상자 시각화
        cv2.rectangle(out, (win_left_x_low, win_y_low), (win_left_x_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out, (win_right_x_low, win_y_low), (win_right_x_high, win_y_high), (0, 255, 0), 2)
        
        # 상자 안에 0이 아닌 값들의 인덱스들을 추출한다.
        left_indxs = ((win_y_low <= nonzero_y) & (nonzero_y <= win_y_high) & (win_left_x_low <= nonzero_x) & (nonzero_x <= win_left_x_high)).nonzero()[0]
        right_indxs = ((win_y_low <= nonzero_y) & (nonzero_y <= win_y_high) & (win_right_x_low <= nonzero_x) & (nonzero_x <= win_right_x_high)).nonzero()[0]
        
        left_lane_indxs.append(left_indxs)
        right_lane_indxs.append(right_indxs)
        
        # If you found > minpix pixels, recenter next window on their mean position
        # 사각형 안에 점이 minpix 개수 초과이면 해당 점들의 평균을 낸다. 상자의 위치를 조절한다.
        # 슬라이딩 윈도우의 x를 조절
        if len(left_indxs) >  minpix:
            left_x_current = int(np.mean(nonzero_x[left_indxs]))

        if len(right_indxs) > minpix:
            right_x_current = int(np.mean(nonzero_x[right_indxs]))
    
    # 상자들을 이용해서 차선 인덱스들을 총 추출한다.
    left_lane_indxs = np.concatenate(left_lane_indxs)
    right_lane_indxs = np.concatenate(right_lane_indxs)
    
    # 인덱스들을 이용해서 x, y 값을 얻는다.
    left_x = nonzero_x[left_lane_indxs]
    left_y = nonzero_y[left_lane_indxs]
    right_x = nonzero_x[right_lane_indxs]
    right_y = nonzero_y[right_lane_indxs]
    
    # fit 2nd order polynomial
    # 차선 점들의 x,y 들을 이용하여 2차원 피팅하기
    # left_fit, right_fit = None, None
    if len(left_x) >= min_lane_pts and len(right_x) >= min_lane_pts:
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

#     valid = check_validity(left_fit, right_fit, diagnostics=diagnostics)
    valid = True
    
    if not valid:
        if len(cache) == 0:
            if diagnostics: 
                print('WARNING: Unable to detect lane lines in this frame.')
            return False, np.array([]), np.array([])
        
        avg_params = np.mean(cache, axis=0)
        left_fit, right_fit = avg_params[0], avg_params[1]
        ret = False

    plot_left_x, plot_left_y, plot_right_x, plot_right_y = get_poly_points(left_fit, right_fit)
    
    # Color the detected pixels for each lane line
    out[left_y, left_x] = [255, 0, 0]
    out[right_y, right_x] = [255, 10, 255]

    left_poly_pts = np.array([np.transpose(np.vstack([plot_left_x, plot_left_y]))])
    right_poly_pts = np.array([np.transpose(np.vstack([plot_right_x, plot_right_y]))])

    # Plot the fitted polynomial
    
    cv2.polylines(out, [left_poly_pts], isClosed=False, color=(255, 0, 0), thickness=10)
    cv2.polylines(out, [right_poly_pts], isClosed=False, color=(255,0, 0), thickness=10)
        
    return ret, out, np.array([left_fit, right_fit])

def polyfit_adapt_search(img, prev_poly_param, diagnostics=False):
    global cache
    global attempts
    
    assert(len(img.shape) == 3)
    
    # Setup
    nb_windows = 10 # Number of windows over which to perform the localised color thresholding  
    bin_margin = 20 # Width of the windows +/- margin for localised thresholding
    margin = 20 # Width around previous line positions +/- margin around which to search for the new lines
    window_height = int(img.shape[0] / nb_windows)
    smoothing_window = 5 # Number of frames over which to compute the Moving Average
    min_lane_pts = 10
    
    binary = np.zeros_like(img[:,:,0])
    img_plot = np.copy(img)
    
    prev_left_fit, prev_right_fit = prev_poly_param[0], prev_poly_param[1]
    plot_prev_left_x, plot_prev_left_y, plot_prev_right_x, plot_prev_right_y = get_poly_points(prev_left_fit, prev_right_fit)
    
    current_left_x = plot_prev_left_x[-1]
    current_right_x = plot_prev_right_x[-1]
    
    for window in range(nb_windows):
        # 상자 크기 세팅
        win_y_low = IMG_SHAPE[0] - (window + 1) * window_height
        win_y_high = IMG_SHAPE[0] - window * window_height
        win_left_x_low = min(max(0, current_left_x - bin_margin), IMG_SHAPE[1])
        win_left_x_high = min(max(0, current_left_x + bin_margin), IMG_SHAPE[1])
        win_right_x_low = min(max(0, current_right_x - bin_margin), IMG_SHAPE[1])
        win_right_x_high = min(max(0, current_right_x + bin_margin), IMG_SHAPE[1])
        
        # 해당 윈도우의 크기만큼 사진을 자르고 바이너리화 함
        img_win_left = img[win_y_low:win_y_high, win_left_x_low:win_left_x_high,:]
        binary[win_y_low:win_y_high, win_left_x_low:win_left_x_high] = get_binary_image(img_win_left)
        
        img_win_right = img[win_y_low:win_y_high, win_right_x_low:win_right_x_high,:]
        binary[win_y_low:win_y_high, win_right_x_low:win_right_x_high] = get_binary_image(img_win_right)
        
        # 이전 이미지에서 차선에 대한 2차 피팅 곡선과 현재 윈도우에 해당하는 점들의 인덱스 
        # 그리고 이를 이용한 해당 윈도우에서의 시작 위치를 설정
        idxs = np.where(plot_prev_left_y == win_y_low)[0]
        if len(idxs) != 0:
            current_left_x = int(plot_prev_left_x[idxs[0]])
        
        idxs = np.where(plot_prev_right_y == win_y_low)[0]
        if len(idxs) != 0:
            current_right_x = int(plot_prev_right_x[idxs[0]])
                        
    # Identify the x-y coordinates of all the non-zero pixels from the binary image
    # generated above
    nonzero = binary.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    left_lane_indxs = \
        ((nonzero_x > (prev_left_fit[0]*(nonzero_y**2) + prev_left_fit[1]*nonzero_y + prev_left_fit[2] - margin)) &
        (nonzero_x < (prev_left_fit[0]*(nonzero_y**2) + prev_left_fit[1]*nonzero_y + prev_left_fit[2] + margin))) 

    right_lane_indxs = \
        ((nonzero_x > (prev_right_fit[0]*(nonzero_y**2) + prev_right_fit[1]*nonzero_y + prev_right_fit[2] - margin)) &
        (nonzero_x < (prev_right_fit[0]*(nonzero_y**2) + prev_right_fit[1]*nonzero_y + prev_right_fit[2] + margin))) 
    
    left_x = nonzero_x[left_lane_indxs]
    left_y = nonzero_y[left_lane_indxs]
    right_x = nonzero_x[right_lane_indxs]
    right_y = nonzero_y[right_lane_indxs]
    
    # Sanity checks
    if len(left_x) > min_lane_pts:
        left_fit = np.polyfit(left_y, left_x, 2)
    else:
        if diagnostics: 
            print('WARNING: Less than {} pts detected for the left lane. {}'.format(min_lane_pts, len(left_x)))

    if len(right_x) > min_lane_pts:
        right_fit = np.polyfit(right_y, right_x, 2)
    else:
        if diagnostics: 
            print('WARNING: Less than {} pts detected for the right lane. {}'.format(min_lane_pts, len(right_x)))
        
#     valid = check_validity(left_fit, right_fit, diagnostics=diagnostics)
    valid = True
    
    
    # Perform smoothing via moving average
    if valid:
        if len(cache) < smoothing_window:
            cache = np.concatenate((cache, [np.array([left_fit, right_fit])]), axis=0)
        elif len(cache) >= smoothing_window:
            cache[:-1] = cache[1:]
            cache[-1] = np.array([left_fit, right_fit])

        avg_params = np.mean(cache, axis=0)
        left_fit, right_fit = avg_params[0], avg_params[1]
        plot_left_x, plot_left_y, plot_right_x, plot_right_y = get_poly_points(left_fit, right_fit)
        curr_poly_param = np.array([left_fit, right_fit])
    else:
        attempts += 1
        curr_poly_param = prev_poly_param
        
    
    # 선 그리기
    out = np.dstack([binary, binary, binary]) * 255
    win_img = np.zeros_like(out)
    
    out[left_y, left_x] = [255, 0, 0]
    out[right_y, right_x] = [255, 10, 255]
    
    left_window1 = np.array([np.vstack([plot_left_x - margin, plot_left_y]).T])
    left_window2 = np.array([np.flipud(np.vstack([plot_left_x + margin, plot_left_y]).T)])
    left_pts = np.hstack([left_window1, left_window2])
    
    right_window1 = np.array([np.vstack([plot_right_x - margin, plot_right_y]).T])
    right_window2 = np.array([np.flipud(np.vstack([plot_right_x + margin, plot_right_y]).T)])
    right_pts = np.hstack([right_window1, right_window2])
    
    cv2.fillPoly(win_img, [left_pts], (0, 255, 0))
    cv2.fillPoly(win_img, [right_pts], (0, 255, 0))
    
    out = cv2.addWeighted(out, 1, win_img, 0.25, 0)
    
    left_poly_pts = np.array([np.vstack([plot_left_x, plot_left_y]).T])
    right_poly_pts = np.array([np.vstack([plot_right_x, plot_right_y]).T])
    
    cv2.polylines(out, [left_poly_pts], isClosed=False, color=(255,255,255), thickness=4)
    cv2.polylines(out, [right_poly_pts], isClosed=False, color=(255,255,255), thickness=4)
    
    return out, curr_poly_param


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부),0)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshould, high_threshold):
    return cv2.Canny(img, low_threshould, high_threshold)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    # If there are no lines to draw, exit.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img,rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img=np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    draw_lines(line_img,lines)
    return line_img 


def weighted_img(img, initial_img, alpha=0.8,beta=1.,gamma=0.):
    return cv2.addWeighted(initial_img,alpha,img, beta,gamma)
#====== 이미지 전처리 ==========


def start():

    # 위에서 선언한 변수를 start() 안에서 사용하고자 함
    global motor, image, IMG_SHAPE, cache
    IMG_SHAPE = (480, 640)
    #=========================================
    # ROS 노드를 생성하고 초기화 함.
    # 카메라 토픽을 구독하고 모터 토픽을 발행할 것임을 선언
    #=========================================
    rospy.init_node('driving')
    motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
    imu   = rospy.Publisher('/imu', Imu, queue_size=1)
    image_sub = rospy.Subscriber("/usb_cam/image_raw/",Image,img_callback)

    print ("----- Xycar self driving -----")

    # 첫번째 카메라 토픽이 도착할 때까지 기다림.
    while not image.size == (WIDTH * HEIGHT * 3):
        continue
    #=========================================
    # 메인 루프 
    # 카메라 토픽이 도착하는 주기에 맞춰 한번씩 루프를 돌면서 
    # "이미지처리 +차선위치찾기 +조향각결정 +모터토픽발행" 
    # 작업을 반복적으로 수행함.
    #=========================================
    while not rospy.is_shutdown():

        # 이미지처리를 위해 카메라 원본이미지를 img에 복사 저장
        img = image.copy()  
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



        # 이미지처리 부분
        ## 파라미터
        kernel_size =5
        low_threshould = 50
        high_threshould = 200

        imshape = img.shape

        rho = 2
        theta = np.pi/180
        threshold=90
        min_line_len=80
        max_line_gap=100
        vertices = np.array([[(0,imshape[0]), (0,imshape[0]-150),(imshape[1],imshape[0]-150), (imshape[1],imshape[0])]], dtype=np.int32)

        gray = grayscale(imgRGB)
        edges= canny(gray, low_threshould, high_threshould)

        bird_eye_img = get_bird_eye(img)

        cv2.imshow("CAM View2", edges)       
        # cv2.waitKey(1)

        binary_img = get_binary_image(bird_eye_img)

        binary_img_plt = binary_img * 255

        # cv2.imshow("CAM View3", binary_img_plt)       
        # cv2.waitKey(1)        

        _, poly_img, poly_param = polyfit_sliding_window(binary_img)
        # poly_img = poly_img * 255
        cache = np.array([poly_param])

        poly_adapt_img, _ = polyfit_adapt_search(bird_eye_img, poly_param)

        cv2.imshow("CAM View4", poly_adapt_img)       
        cv2.waitKey(1)
                
        #=========================================
        # 핸들조향각 값인 angle값 정하기.
        # 차선의 위치 정보를 이용해서 angle값을 설정함.        
        #=========================================
		
        # 우선 테스트를 위해 직진(0값)으로 설정
        # angle = 0
		
        #=========================================
        # 차량의 속도 값인 speed값 정하기.
        # 직선 코스에서는 빠른 속도로 주행하고 
        # 회전구간에서는 느린 속도로 주행하도록 설정함.
        #=========================================

        # 우선 테스트를 위해 느린속도(10값)로 설정
        # speed = 10
		
        # drive() 호출. drive()함수 안에서 모터 토픽이 발행됨.
        # drive(angle, speed)


#=============================================
# 메인 함수
# 가장 먼저 호출되는 함수로 여기서 start() 함수를 호출함.
# start() 함수가 실질적인 메인 함수임. 
#=============================================
if __name__ == '__main__':
    start()

