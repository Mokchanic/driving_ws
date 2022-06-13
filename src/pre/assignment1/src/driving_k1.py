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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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

#=============================================그래서 요즘 나오는 자율 주행으로 연결하기는 조금 어렵지만, 라인트레이싱을 영상으로 해보겠다~ 정도는 커버될 것 같네요^^
# 실질적인 메인 함수 
# 카메라 토픽을 받아 각종 영상처리와 알고리즘을 통해
# 차선의 위치를 파악한 후에 조향각을 결정하고,
# 최종적으로 모터 토픽을 발행하는 일을 수행함. 
#=============================================
"""
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
"""
#==========================================================================

def color_filter(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    lower = np.array([20, 150, 20])
    upper = np.array([255, 255, 255])

    yellow_lower = np.array([0, 85, 81])
    yellow_upper = np.array([190, 255, 255])

    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)

    return masked

def roi(image):
    x = float(image.shape[1])
    y = float(image.shape[0])
    # 한 붓 그리기
    _shape = np.array(
        [[float(0.01*x), float(y)],
        [float(0.01*x), float(0.01*y)],
        [float(0.4*x), float(0.01*y)], 
        [float(0.4*x), float(y)], 
        [float(0.7*x), float(y)], 
        [float(0.7*x), float(0.01*y)],
        [float(0.99*x), float(0.01*y)], 
        [float(0.99*x), float(y)], 
        [float(0.02*x), float(y)]])

    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

def wrapping(image):
    (h, w) = (image.shape[0], image.shape[1])

    #source = np.float32([[19.48, 400], [224.68, 300], [606.49, 400], [410.39, 300]])
    #destination = np.float32([[0+19.48, 0], [0+19.48, h-298], [w-32.47, 0], [w-32.47, h-298]])

    source = np.float32([[19.48, 400], [228, 300], [606.49, 400], [410, 300]])
    destination = np.float32([[0+32, h], [0+32, h/2], [w-32, h], [w-32, h/2]])

    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    minv = cv2.getPerspectiveTransform(destination, source)
    _image = cv2.warpPerspective(image, transform_matrix, (w, h))

    return _image, minv

def plothistogram(image):
    histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
    
    midpoint = np.int(histogram.shape[0]/2)
    leftbase = np.argmax(histogram[:midpoint])
    rightbase = np.argmax(histogram[midpoint:]) + midpoint

    return leftbase, rightbase, histogram

def slide_window_search(binary_warped, left_current, right_current):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    nwindows = 6
    window_height = np.int(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()  # 선이 있는 부분의 인덱스만 저장 
    nonzero_y = np.array(nonzero[0])  # 선이 있는 부분 y의 인덱스 값
    nonzero_x = np.array(nonzero[1])  # 선이 있는 부분 x의 인덱스 값 
    margin = 100
    minpix = 50
    left_lane = []
    right_lane = []
    color = [0, 255, 0]
    thickness = 2

    for w in range(nwindows):
        win_y_low = binary_warped.shape[0] - (w + 1) * window_height  # window 윗부분
        win_y_high = binary_warped.shape[0] - w * window_height  # window 아랫 부분
        win_xleft_low = left_current - margin  # 왼쪽 window 왼쪽 위
        win_xleft_high = left_current + margin  # 왼쪽 window 오른쪽 아래
        win_xright_low = right_current - margin  # 오른쪽 window 왼쪽 위 
        win_xright_high = right_current + margin  # 오른쪽 window 오른쪽 아래

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
        left_lane.append(good_left)
        right_lane.append(good_right)
        # cv2.imshow("oo", out_img)

        if len(good_left) > minpix:
            left_current = np.int(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            right_current = np.int(np.mean(nonzero_x[good_right]))

    left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침
    right_lane = np.concatenate(right_lane)

    leftx = nonzero_x[left_lane]
    lefty = nonzero_y[left_lane]
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    ltx = np.trunc(left_fitx)  # np.trunc() -> 소수점 부분을 버림
    rtx = np.trunc(right_fitx)

    out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
    out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]

    #plt.imshow(out_img)
    #plt.plot(left_fitx, ploty, color = 'yellow')
    #plt.plot(right_fitx, ploty, color = 'yellow')
    #plt.xlim(0, 640)
    #plt.ylim(480, 0)
    #plt.show()

    ret = {'left_fitx' : ltx, 'right_fitx': rtx, 'ploty': ploty}

    return ret

def draw_lane_lines(original_image, warped_image, Minv, draw_info):
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

    cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74))
    cv2.fillPoly(color_warp, np.int_([pts_mean]), (216, 168, 74))

    newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, newwarp, 0.4, 0)

    return pts_mean, result












def start():

    # 위에서 선언한 변수를 start() 안에서 사용하고자 함
    global motor, image

    #=========================================
    # ROS 노드를 생성하고 초기화 함.
    # 카메라 토픽을 구독하고 모터 토픽을 발행할 것임을 선언
    #=========================================
    rospy.init_node('driving')
    motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
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
#===========================================================
        # 이미지처리를 위해 카메라 원본이미지를 img에 복사 저장
        img = image.copy()  
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #bgrLower = np.array([150, 150, 150])    # 추출할 색의 하한(BGR)
        #bgrUpper = np.array([255, 255, 255])    # 추출할 색의 상한(BGR)
        #img_mask = cv2.inRange(image, bgrLower, bgrUpper) # BGR로 부터 마스크를 작성
        #result = cv2.bitwise_and(image, image, mask=img_mask)
#==========================================================================        
        


        # 디버깅을 위해 모니터에 이미지를 디스플레이
        
#        gray = grayscale(imgRGB)
#        
#        kernel_size =5
#        blur_gray = gaussian_blur(gray, kernel_size)
#        low_threshould = 50
#        high_threshould = 200
#        edges= canny(blur_gray, low_threshould, high_threshould)
#        # 엣지 de
#        
#        imshape = img.shape
#        vertices = np.array([[(0,imshape[0]), (0,imshape[0]-80),(280, 280),(350, 280),(imshape[1],imshape[0]-80), (imshape[1],imshape[0])]], dtype=np.int32)
#        mask = region_of_interest(edges, vertices)
#        rho = 2
#        theta = np.pi/180
#        threshold=90
#        min_line_len=80
#        max_line_gap=100

#        lines=hough_lines(mask,rho,theta,threshold,min_line_len,max_line_gap)
#        lines_edges = weighted_img(lines,img,alpha=0.8,beta=1.,gamma=0.)
#        
#        plt.figure(figsize=(10,8))
#        plt.imshow(gray)
#        plt.show()
#========================================================        
        img = image.copy()
        
        #plt.imshow(img)
        #plt.show()
        wrapped_img, minverse = wrapping(img)
        
        #cv2.imshow("CAM View", wrapped_img)
        
        
        bgrLower = np.array([150, 150, 150])    # 추출할 색의 하한(BGR)
        bgrUpper = np.array([255, 255, 255])    # 추출할 색의 상한(BGR)
        img_mask = cv2.inRange(wrapped_img, bgrLower, bgrUpper) # BGR로 부터 마스크를 작성
        result = cv2.bitwise_and(wrapped_img, wrapped_img, mask=img_mask)
        
        
        ## 조감도 필터링
        #w_f_img = color_filter(wrapped_img)
        
        w_f_img = result
        #cv2.imshow('w_f_img', w_f_img)
        
        ##조감도 필터링 자르기
        w_f_r_img = roi(w_f_img)
        #cv2.imshow('w_f_r_img', w_f_r_img)

        ## 조감도 선 따기 wrapped img threshold
        _gray = cv2.cvtColor(w_f_r_img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(_gray, 160, 255, cv2.THRESH_BINARY)
        #cv2.imshow('threshold', thresh)

        #  # 선 분포도 조사 histogram
        leftbase, rightbase, hist = plothistogram(thresh)
        #plt.plot(hist)
        #plt.show()

        ## histogram 기반 window roi 영역
        draw_info = slide_window_search(thresh, leftbase, rightbase)
        #plt.plot(left_fit)
        #plt.show()

        ## 원본 이미지에 라인 넣기
        meanPts, result = draw_lane_lines(img, thresh, minverse, draw_info)
        cv2.imshow("result", result)









        #cv2.imshow("CAM View", w_f_img)
        cv2.waitKey(1)       
        #=========================================
        # 핸들조향각 값인 angle값 정하기.
        # 차선의 위치 정보를 이용해서 angle값을 설정함.        
        #=========================================
		
        # 우선 테스트를 위해 직진(0값)으로 설정
        angle = 0
		
        #=========================================


#=============================================
        #=========================================

        # 우선 테스트를 위해 느린속도(10값)로 설정
        speed = 30
		
        # drive() 호출. drive()함수 안에서 모터 토픽이 발행됨.
        drive(angle, speed)


#=============================================
# 메인 함수
# 가장 먼저 호출되는 함수로 여기서 start() 함수를 호출함.
# start() 함수가 실질적인 메인 함수임. 
#=============================================
if __name__ == '__main__':
    start()






#ym_per_pix = 30 / 720
#xm_per_pix = 3.7 / 720





#    key = cv2.waitKey(25)
#    if key == 27:
#        break


#if cap.isOpened():
#    cap.release()

#cv2.destroyAllWindows()
