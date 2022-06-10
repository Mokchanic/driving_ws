import cv2


def clr_segment(hls, lower_lange, upper_range):
    mask_in_range = cv2.inRange(hls, lower_lange, upper_range)
    kernel = cv2.getStructureingElement(cv2.MORPH_ELLIPSE,(3,3))
    mask_dilated = cv2.morphologyEx(mask_in_range, cv2.MORPH_DILATE, kernel)
    return mask_dilated


def segment_lanes(frame, min_area):
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)