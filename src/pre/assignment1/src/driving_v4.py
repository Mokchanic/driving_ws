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

def get_perspective_transform(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return M, Minv, warped


def preprocess_image(img):
    ysize = img.shape[0] # 480
    xsize = img.shape[1] # 640
    
    # undistortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    # perspective transformation
    src = np.float32([
        (210,300),
        (0,480),
        (630,440),
        (430,300)
    ])

    dst = np.float32([
        (175, 0),
        (175, ysize),
        (xsize - 175, ysize),
        (xsize - 175, 0)

    ])
    
    M = cv2.getPerspectiveTransform(src, dst)
    invM = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist, M, (xsize, ysize), flags=cv2.INTER_LINEAR)

    cv2.imshow("CAM View1", warped)

    # roi crop
    vertices = np.array([ # region for crop
        [100, 0],
        [100, ysize],
        [540, ysize],
        [540, 0]
    ])
    
    vertices = np.array(vertices, ndmin=3, dtype=np.int32)

    if len(img.shape) == 3: # (row, column , channel)
        fill_color = (255,) * 3
    else:
        fill_color = 255
    
    mask = np.zeros_like(warped)
    mask = cv2.fillPoly(mask, vertices, fill_color) # mask에서 vertices 모양을 fill_color로 채움

    roi = cv2.bitwise_and(warped, mask) # img에서 mask와 비트연선으로 사진 자르기

    cv2.imshow("CAM View", roi)
    # cv2.waitKey(1)
    
    return roi, undist, M, invM, warped


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
    
    cv2.imshow("check1", R_binary * 255)
    cv2.imshow("check2", lab_binary * 255)
    cv2.imshow("check3", hls_binary * 255)
    cv2.imshow("check4", adapt_binary * 255)
    cv2.imshow("check5", combined * 255)

    return  combined


# def histogram(combined_binary):
#     return np.sum(combined_binary[combined_binary.shape[0]//2:,:], axis = 0)


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
    nwindows = 15
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
        right_fit = np.polyfit(righty,rightx, 2)
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
        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        except:
            left_fit = np.polyfit(lefty, leftx, 1)
            right_fit = np.polyfit(righty, rightx, 1)


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
    out_img[lefty, leftx] = [255,0,0]  # blue
    out_img[righty,rightx] = [0,0,255] # red

    # ploty = np.linspace(0, combined_binary.shape[0]-1, combined_binary.shape[0] )
    # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return out_img, left_lane_inds, right_lane_inds, leftx, lefty, rightx, righty, left_fit, right_fit, nonzerox, nonzeroy

def sliding_window(combined_binary, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fitx, right_fitx, ploty):
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


# def get_curvature(ploty, leftx, lefty, rightx, righty, left_fit, right_fit):
#     y_eval = np.max(ploty)
#     left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
#     right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
#     # Define conversions in x and y from pixels space to meters
#     ym_per_pix = 30/480 # meters per pixel in y dimension
#     xm_per_pix = 3.7/640 # meters per pixel in x dimension
#     # Fit new polynomials to x,y in world space
#     left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
#     right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
#     # Calculate the new radii of curvature
#     left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
#     right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
#     return (left_curverad, right_curverad)


def get_center(img, ploty, left_fit, right_fit):
    ymax = np.max(ploty)
    xm_per_pix = 3.7/640 # meters per px in x-dim (conversions in x,y from pixels space to meters)
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


def pipeline_process(img, bird_eye, ploty):
    ysize = img.shape[0] # 480
    xsize = img.shape[1] # 640

    src = np.float32([
        (500,300),
        (140,300),
        (10,440),
        (630,440)
    ])

    dst = np.float32([
        (xsize - 175, 0),
        (175, 0),
        (175, ysize),
        (xsize - 175, ysize)
    ])

    # combined_binary, warped, M, Minv, undist = apply_binary(img, src, dst)
    _, undist, M, invM, warped = preprocess_image(img)
    combined_binary = get_binary_image(bird_eye)
    
    global detected_line, left_fit, right_fit, left_fit_poly, right_fit_poly
    
    out_img, left_lane_inds, right_lane_inds, leftx, lefty, rightx, righty, left_fit, right_fit, nonzerox, nonzeroy = find_lane_lines(combined_binary)

    drawwarped = draw_lines(np.zeros_like(warped), leftx, lefty, rightx, righty)
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
    
    pos = get_center(img, ploty, left_fit, right_fit)
    if pos < 0:
        text = "Vehicle is {:.2f} m on left".format(-pos)
    else:
        text = "Vehicle is {:.2f} m on right".format(pos)
    cv2.putText(output, text, (50,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    return output

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
        # imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # (480, 680, 3)

        # 이미지처리 부분

        bird_eye_img, _, _, _, _ = get_bird_eye(img)

        binary_img = get_binary_image(bird_eye_img)

        binary_img_plt = binary_img * 255 # 여기까지는 문제 없음.
        
        ## histogram 확인
        # histogram_plt = histogram(binary_img) # 왼쪽, 오른쪽 line의 histogram 계산
        # out_img = np.dstack((histogram_plt, histogram_plt, histogram_plt))*255
        # out_img = np.array(out_img, dtype=np.uint8)
        # plt.plot(histogram_img)
        # plt.show()

        ## histogram 확인 후 fine_line을 제작
        out_img, left_lane_inds, right_lane_inds, leftx, lefty, rightx, righty, left_fit, right_fit, nonzerox, nonzeroy = find_lane_lines(binary_img)

        ploty = np.linspace(0, binary_img.shape[0]-1, binary_img.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        sliding_window_img = sliding_window(binary_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fitx, right_fitx, ploty)

        final_img = pipeline_process(img, bird_eye_img, ploty)

        cv2.imshow("CAM View1", final_img)
        cv2.imshow("CAM View2", sliding_window_img)
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

