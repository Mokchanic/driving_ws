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

#=============================================그래서 요즘 나오는 자율 주행으로 연결하기는 조금 어렵지만, 라인트레이싱을 영상으로 해보겠다~ 정도는 커버될 것 같네요^^
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
        (420,275),
        (200,275),
        (40,400),
        (600,400) 
        # (150,350),
        # (500,350),    
        # (610,400),
        # (30,400) 
    ])

    dst = np.float32([
        (xsize - 175, 0),
        (175, 0),
        (175, ysize),
        (xsize - 175, ysize)
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

def birdView(img,M):
    '''
    Transform image to birdeye view
    img:binary image
    M:transformation matrix
    return a wraped image
    '''
    img_sz = (img.shape[1],img.shape[0])
    img_warped = cv2.warpPerspective(img,M,img_sz,flags = cv2.INTER_LINEAR)
    return img_warped
def perspective_transform(src_pts,dst_pts):
    '''
    perspective transform
    args:source and destiantion points
    return M and Minv
    '''
    M = cv2.getPerspectiveTransform(src_pts,dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts,src_pts)
    return {'M':M,'Minv':Minv}
#====== 이미지 전처리 ==========

def start():

    # 위에서 선언한 변수를 start() 안에서 사용하고자 함
    global motor, image

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
        
        # 전처리
        gray = grayscale(imgRGB)
        blur_gray = gaussian_blur(gray, kernel_size)
        edges= canny(blur_gray, low_threshould, high_threshould)
        mask = region_of_interest(edges, vertices)
        # bird_eye_img = get_bird_eye(mask)
        try: 
            lines=hough_lines(mask,rho,theta,threshold,min_line_len,max_line_gap)
        except:
            lines=cashe    
        lines_edges = weighted_img(lines,img,alpha=0.8,beta=1.,gamma=0.)
        cashe = lines
        # plt.figure(figsize=(10,8))
        # plt.imshow("Check_line"lines_edges)
        # plt.show()
        roi = preprocess_image(mask)
        # cv2.imshow("Img", img)
        # cv2.imshow("Gray", gray)
        # cv2.imshow("Blue_gray", blur_gray)
        # cv2.imshow("Edges", edges)
        # cv2.imshow("Mask", mask)
        # cv2.imshow("CAM View", lines)
        cv2.imshow("CAM View", roi)

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

