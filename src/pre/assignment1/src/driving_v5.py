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
from matplotlib import animation, rc 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sensor_msgs.msg import Imu

#Global variables
mtx = np.array([[1.15396093e+03, 0.00000000e+00, 6.69705357e+02],
            [0.00000000e+00, 1.14802496e+03, 3.85656234e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist = np.array([[-2.41017956e-01, -5.30721173e-02, -1.15810355e-03, -1.28318856e-04,  2.67125290e-02]])

src = np.float32([
    (130,340),
    (0,410),
    (640,410),
    (495,340)

])

dst = np.float32([
    (175, 0),
    (175, 480),
    (640 - 175, 480),
    (640 - 175, 0)
])

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

# thresholded_binary
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
    if orient=='x': abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient=='y': abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel)) # rescale 8-bit
    sobel_binary = np.zeros_like(scaled_sobel) # create a copy and apply threshold
    sobel_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sobel_binary


# HLS_binary
def HLS_thresh(img, thresh=(0, 255)): # (170,255) (150,255) (180, 255)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    ch = hls[:,:,2] # saturation
    hls_binary = np.zeros_like(ch)
    hls_binary[(ch > thresh[0]) & (ch <= thresh[1])] = 1
    return hls_binary


# HSV_binary
def HSV_thresh(img, channel='v', thresh=(0,255)):
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    if channel=='v': ch = hsv[:,:,2] # value
    if channel=='s': ch = hsv[:,:,1] # saturation
    hsv_binary = np.zeros_like(ch)
    hsv_binary[(ch > thresh[0]) & (ch <= thresh[1])] = 1
    return hsv_binary



# LUV_binary
def LUV_thresh(img, thresh=(0,255)):
    luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    ch = luv[:,:,0] 
    luv_binary = np.zeros_like(ch)
    luv_binary[(ch >= thresh[0]) & (ch <= thresh[1])]
    return luv_binary
    

# LAB_binary
def LAB_thresh(img, thresh=(0,255)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab) # convert RGB to floating-point format and scaled to fit the 0 to 1 range.
    lab2 = lab[:,:,2]
    lab_binary = np.zeros_like(lab2)
    lab_binary[((lab2 >= thresh[0]) & (lab2 <= thresh[1]))] = 1
    return lab_binary


# Lightness_mask
def lightness_mask(img, thresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    saturation = hls[:,:,2]
    lightness = hls[:,:,1]
    lightness_binary = np.zeros_like(saturation)
    lightness_binary[(saturation >=thresh[0]) & (lightness>=thresh[1])] = 1
    return lightness_binary


# Gradient_thresholds: 위에서 적용한 binary들을 combined 해주는 함수
def combined_gradient_thresholds(img): # img: undistorted image
    sobelx_binary = abs_sobel_thresh(img, orient='x', sobel_kernel=5, thresh=(30,90))
    sobely_binary = abs_sobel_thresh(img, orient='y', sobel_kernel=5, thresh=(30,90))
    saturation_binary = HLS_thresh(img, thresh=(120,255))
    value_HSV_binary = HSV_thresh(img, channel='v', thresh=(75,255))
    saturation_HSV_binary = HSV_thresh(img, channel='s', thresh=(120,255))
    luv_binary = LUV_thresh(img, thresh=(225,255))
    lightness_binary = lightness_mask(img, thresh=(5,130))
    lab_binary = LAB_thresh(img, thresh=(155,200))
    combined_binary = np.zeros_like(sobelx_binary)
    combined_binary[(saturation_HSV_binary==1) & (lightness_binary==1)]=1 ## 
    combined_binary = np.asarray(value_HSV_binary - saturation_binary, dtype=np.uint8)
    

    # cv2.imshow("CAM sobelx_binary", sobelx_binary * 255)
    # cv2.imshow("CAM sobely_binary", sobely_binary * 255)
    # cv2.imshow("CAM saturation_binary", saturation_binary * 255)
    # cv2.imshow("CAM value_HSV_binary", value_HSV_binary * 255)
    # cv2.imshow("CAM saturation_HSV_binary", saturation_HSV_binary * 255)
    # cv2.imshow("CAM luv_binary", luv_binary * 255)
    # cv2.imshow("CAM lightness_binary", lightness_binary * 255)
    # cv2.imshow("lab_binary", lab_binary * 255)


    return combined_binary


# Perspective_transform: 시점 변환을 도와주는 함수.
def get_perspective_transform(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return M, Minv, warped


# bird_eye: 인식된 이미지를 탑뷰로 바꿔주는 함수.
def bird_eye_perspective(img, src, dst):
    undist = cv2.undistort(img, mtx, dist, None, mtx) # undistorted image
    M, Minv, warped = get_perspective_transform(undist, src, dst)
    return M, Minv, warped, undist


# apply_binary: 위에서 만든 함수들을 하나로 합쳐서 실행하여 이진화된 탑뷰를 얻는 함수
def apply_binary(img, src, dst):
    M, Minv, warped, undist = bird_eye_perspective(img, src, dst) # warped image is undistorte and warped.
    combined_binary = combined_gradient_thresholds(warped) # sobelx+saturation+value
    return combined_binary, warped, M, Minv, undist


# apply_binary에서 얻은 이미지를 적용하여 line을 얻는 함수.
def find_lane_lines(combined_binary):
    histogram = np.sum(combined_binary[combined_binary.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((combined_binary, combined_binary, combined_binary))*255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(combined_binary.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = combined_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
        
    global detected_line, left_fit, right_fit, left_fit_poly, right_fit_poly
    if detected_line==False:
        for window in range(nwindows): # Step through the windows one by one
            # Identify window boundaries in x and y (and right and left)
            win_y_low = combined_binary.shape[0] - (window+1)*window_height
            win_y_high = combined_binary.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2) 
            # 

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        #Pixel position for left and right
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        #Fit 2nd polynomial
        left_fit = np.polyfit(lefty,leftx, 2)
        right_fit = np.polyfit(righty,rightx,2)
        left_fit_poly = np.array([left_fit])
        right_fit_poly = np.array([right_fit])
        
    else:
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
        left_fit[1]*nonzeroy + left_fit[2] + margin))) 

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
        right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

    if left_fit is None or right_fit is None:
        detected_line = False
    else:
        detected_line = True
    
    # Average poly coefficient
    left_fit_poly = np.concatenate((left_fit_poly,[left_fit]),axis=0)[-5:]
    right_fit_poly = np.concatenate((right_fit_poly,[right_fit]),axis=0)[-5:]
    left_fit = np.average(left_fit_poly, axis=0)
    right_fit = np.average(right_fit_poly, axis=0)
    
    # x & y values for plotting
    out_img[lefty, leftx] = [255,255,0]
    out_img[righty,rightx] = [0,0,255]

    ploty = np.linspace(0, combined_binary.shape[0]-1, combined_binary.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return out_img, left_lane_inds, right_lane_inds, leftx, lefty, rightx, righty, left_fit, right_fit, nonzerox, nonzeroy


def sliding_window(combined_binary, nonzerox, nonzeroy, left_lane_inds, right_lane_inds):
    ploty = np.linspace(0, combined_binary.shape[0]-1, combined_binary.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    margin=50
    out_img = np.dstack((combined_binary, combined_binary, combined_binary))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return result


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


def polyfit_sliding_window(binary, lane_width_px=480, diagnostics=False):
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
    
    nb_windows = 12 # number of sliding windows
    margin = 50 # width of the windows +/- margin
    minpix = 50 # min number of pixels needed to recenter the window
    window_height = int(IMG_SHAPE[0] / nb_windows)
    min_lane_pts = 10  # min number of 'hot' pixels needed to fit a 2nd order polynomial as a 
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
    left_fit, right_fit = None, None
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

        
    return ret, out, np.array([left_fit, right_fit]), left_x, left_y, right_x, right_y, left_fit, right_fit

def sliding_window_perspective(img, result, dst, src):
    _, _, unwarped = get_perspective_transform(result, dst, src)   

    undist = cv2.undistort(img, mtx, dist, None, mtx) # undistorted image
    final = cv2.addWeighted(undist, 1, unwarped,0.5,0)
    return final


# def get_curvature(ploty, leftx, lefty, rightx, righty, left_fit, right_fit):
#     y_eval = np.max(ploty)
#     left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
#     right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
#     # Define conversions in x and y from pixels space to meters
#     ym_per_pix = 30/720 # meters per pixel in y dimension
#     xm_per_pix = 3.7/700 # meters per pixel in x dimension
#     # Fit new polynomials to x,y in world space
#     left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
#     right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
#     # Calculate the new radii of curvature
#     left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
#     right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
#     return (left_curverad, right_curverad)


def get_center(img, ploty, left_fit, right_fit):
    ymax = np.max(ploty)
    xm_per_pix = 9.1/640 # meters per px in x-dim (conversions in x,y from pixels space to meters)
    left = left_fit[0]*ymax**2 + left_fit[1]*ymax + left_fit[2]
    right = right_fit[0]*ymax**2 + right_fit[1]*ymax + right_fit[2]
    center = (left + right) / 2
    return (img.shape[1]/2 - center)*xm_per_pix


def draw_lines(img, leftx, lefty, rightx, righty):
    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(lefty, leftx, 2)
    left_fitx = left_fit[0]*lefty**2 + left_fit[1]*lefty + left_fit[2]
    right_fit = np.polyfit(righty, rightx, 2) 
    right_fitx = right_fit[0]*righty**2 + right_fit[1]*righty + right_fit[2]

    left_points = np.vstack(([left_fitx.T], [lefty.T])).T
    right_points = np.vstack(([right_fitx.T], [righty.T])).T
    all_points = np.concatenate((left_points, right_points[::-1]))

    cv2.fillConvexPoly(img, np.int32([all_points]), (0, 255, 0))
    cv2.polylines(img, np.int32([left_points]), False, (255, 0, 0), 20)
    cv2.polylines(img, np.int32([right_points]), False, (255, 0, 0), 20)        
    return img


def pipeline_process(img):
    combined_binary, warped, M, Minv, undist = apply_binary(img, src, dst)
    ploty = np.linspace(0, combined_binary.shape[0]-1, combined_binary.shape[0] )

    global detected_line, left_fit, right_fit, left_fit_poly, right_fit_poly
    
    ret, out, _, left_x, left_y, right_x, right_y, left_fit, right_fit = polyfit_sliding_window(combined_binary)
    
    drawwarped = draw_lines(np.zeros_like(warped), left_x, left_y, right_x, right_y)
    _, _, unwarped = get_perspective_transform(drawwarped, dst, src)
    output = cv2.addWeighted(undist, 1, unwarped, 0.3, 0.0)
    
    # left_curverad, right_curverad = get_curvature(lefty, leftx, lefty, rightx, righty, left_fit, right_fit)
    # curvature = min([left_curverad, right_curverad])
    # text = "Curvature of Radius: {:.2f} m".format(curvature)
    # cv2.putText(output, text, (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    # text1 = "Left curve: {:.2f} m".format(left_curverad)
    # cv2.putText(output, text1, (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    # text1 = "Right curve: {:.2f} m ".format(right_curverad)
    # cv2.putText(output, text1, (50,260), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    pos = get_center(img, ploty, left_fit, right_fit) - 0.22
    if pos < 0:
        text = "Vehicle is {:.2f} m on left".format(-pos)
    else:
        text = "Vehicle is {:.2f} m on right".format(pos)
    cv2.putText(output, text, (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    return output, pos

#====== 이미지 전처리 ==========


def start():
    # 위에서 선언한 변수를 start() 안에서 사용하고자 함
    global motor, image, IMG_SHAPE, detected_line, left_fit, right_fit, left_fit_poly, right_fit_poly
    detected_line=False
    left_fit, right_fit=None,None

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

        # apply binary를 이용하여 이미지의 바이너리 이미지들을 얻고 이를 combine하여 탑뷰로 만들어줌.
        bird_eye_combined_binary, warped, M, Minv, undist = apply_binary(img, src, dst)
        bird_eye_combined_binary_img = bird_eye_combined_binary * 255

        # find_lines, left_lane_inds, right_lane_inds, leftx, lefty, rightx, righty, left_fit, right_fit, nonzerox, nonzeroy\
        # = find_lane_lines(bird_eye_combined_binary)

        # sliding_window_img = sliding_window(bird_eye_combined_binary, nonzerox, nonzeroy, left_lane_inds, right_lane_inds)
        # sliding_perspective_img = sliding_window_perspective(img, sliding_window_img, dst, src)


        test, out, _, _, _, _, _, _, _ = polyfit_sliding_window(bird_eye_combined_binary)

        result_img, center_pos = pipeline_process(img)


        # 이미지 확인
        cv2.imshow("CAM img", img)
        # cv2.imshow("CAM bird_eye_combined_binary_img", bird_eye_combined_binary_img)

        cv2.imshow("CAM find_lines", out)
        # cv2.imshow("CAM sliding_sindow_img", sliding_window_img)
        # cv2.imshow("CAM sliding_perspective_img", sliding_perspective_img)
        cv2.imshow("CAM sliding_perspective_img", result_img)


        # cv2.imshow("CAM View2", sliding_window_img)
        cv2.waitKey(1)
                
        #=========================================
        # 핸들조향각 값인 angle값 정하기.
        # 차선의 위치 정보를 이용해서 angle값을 설정함.        
        #=========================================

        if -0.1 <= round(center_pos, 3) <= 0.1: # straight
            angle = 0
            speed = 20
        
        elif -0.4 <= round(center_pos, 3) < - 0.1: # Right
            angle = 10
            speed = 7

        elif 0.4 >= round(center_pos, 3) > 0.1:# Left
            angle = 10
            speed = 7

        elif round(center_pos, 3) < - 0.4:
            angle = 20
            speed = 5

        elif round(center_pos, 3) > 0.4:
            angle = -20
            speed = 5
	
        #=========================================
        # 차량의 속도 값인 speed값 정하기.
        # 직선 코스에서는 빠른 속도로 주행하고 
        # 회전구간에서는 느린 속도로 주행하도록 설정함.
        #=========================================

        # drive() 호출. drive()함수 안에서 모터 토픽이 발행됨.
        drive(angle, speed)


#=============================================
# 메인 함수
# 가장 먼저 호출되는 함수로 여기서 start() 함수를 호출함.
# start() 함수가 실질적인 메인 함수임. 
#=============================================
if __name__ == '__main__':
    start()

