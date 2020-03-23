#!/usr/bin/env python3

import numpy as np
import cv2

from billdetect import BillBallDetect

def cvt_dets(dets):
    '''转换识别结果：
    [{'category_id': 1, 'score': 0.9252024, 'bbox': array([353.4217 , 219.71262, 373.4217 , 239.71262], dtype=float32)}, 
 {'category_id': 2, 'score': 0.8748096, 'bbox': array([667.62244, 483.04712, 687.62244, 503.04712], dtype=float32)}, 
 {'category_id': 12, 'score': 0.8492351, 'bbox': array([352.51196, 111.43207, 372.51196, 131.43207], dtype=float32)}]
    ==>
    3 0(363.4217, 229.71262);1(677.62244, 493.04712);11(362.51196,121.43207)
    '''
    ret = []
    for item in dets:
        one_new = {}
        one_new["id"] = item['category_id'] - 1
        one_new["score"] = item['score']
        one_new["pos_x"] = (item['bbox'][0] + item['bbox'][2]) * 0.5
        one_new["pos_y"] = (item['bbox'][1] + item['bbox'][3]) * 0.5
        ret.append(one_new)
    return ret

def show_result(img, result):
    for ball in result:
        cv2.circle(img, (int(ball["pos_x"]), int(ball["pos_y"])), 10, (0, 0, 255), 1, 0)
        point = ()
        if ball["id"] > 9:
            point = (int(ball["pos_x"]) - 11, int(ball["pos_y"]) + 5)
        else:
            point = (int(ball["pos_x"]) - 6, int(ball["pos_y"]) + 5)
        cv2.putText(img, str(ball["id"]), point, cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
        score = "{:.2f}".format(ball["score"])
        cv2.putText(img, score, (int(ball["pos_x"]) + 6, int(ball["pos_y"]) - 5), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0))
    return img


def main():
    detector = BillBallDetect()
    video_path = "/home/zealens/dyq/CenterNet/data/test_video.avi"
    cap = cv2.VideoCapture(video_path)
    videoWriter = cv2.VideoWriter("out_video_test.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 1, (960, 600))
    # ind = 0
    # while ind < 10:
    while True:
        ret,img = cap.read()
        if ret == False:
            break
        # ind = ind + 1
        dets = cvt_dets(detector.detect(img))
        # print (dets)
        
        frame = show_result(img, dets)
        videoWriter.write(frame) 

    cap.release()
    videoWriter.release()

if __name__ == '__main__':
    main()
