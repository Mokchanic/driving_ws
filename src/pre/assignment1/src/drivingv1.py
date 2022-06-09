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

        # 이미지처리를 위해 카메라 원본이미지를 img에 복사 저장
        img = image.copy()  
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 디버깅을 위해 모니터에 이미지를 디스플레이
        
        gray = grayscale(imgRGB)
        
        kernel_size =5
        blur_gray = gaussian_blur(gray, kernel_size)
        low_threshould = 50
        high_threshould = 200
        edges= canny(blur_gray, low_threshould, high_threshould)
        imshape = img.shape
        vertices = np.array([[(0,imshape[0]), (0,imshape[0]-80),(280, 280),(350, 280),(imshape[1],imshape[0]-80), (imshape[1],imshape[0])]], dtype=np.int32)
        mask = region_of_interest(edges, vertices)
        rho = 2
        theta = np.pi/180
        threshold=90
        min_line_len=80
        max_line_gap=100

        lines=hough_lines(mask,rho,theta,threshold,min_line_len,max_line_gap)
        lines_edges = weighted_img(lines,img,alpha=0.8,beta=1.,gamma=0.)
        

        plt.figure(figsize=(10,8))
        plt.imshow(lines_edges)
        plt.show()
        

        #cv2.imshow("CAM View", img)
        cv2.waitKey(1)       
        #=========================================
        # 핸들조향각 값인 angle값 정하기.
        # 차선의 위치 정보를 이용해서 angle값을 설정함.        
        #=========================================
		
        # 우선 테스트를 위해 직진(0값)으로 설정
        angle = 0
		
        #=========================================
        # 차량의 속도 값인 speed값 정하기.
        # 직선 코스에서는 빠른 속도로 주행하고 
        # 회전구간에서는 느린 속도로 주행하도록 설정함.
        #=========================================

        # 우선 테스트를 위해 느린속도(10값)로 설정
        speed = 10
		
        # drive() 호출. drive()함수 안에서 모터 토픽이 발행됨.
        drive(angle, speed)


#=============================================
# 메인 함수
# 가장 먼저 호출되는 함수로 여기서 start() 함수를 호출함.
# start() 함수가 실질적인 메인 함수임. 
#=============================================
if __name__ == '__main__':
    start()

