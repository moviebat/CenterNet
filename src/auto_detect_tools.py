#/usr/env/bin python3

import sys
import os
import numpy as np
import cv2
import time
from functools import wraps
from memory_profiler import profile
from billdetect import BillBallDetect

def takeFirst(elem):
    return elem[0]


def find_ball(balls, id):
    for i in range(len(balls)):
        if balls[i][0] == id:
            return i
    return -1


def get_frame_number(image_name):
    if -1 != image_name.find("frame_"):
        return int(image_name.split("_")[1].split(".")[0])
    else:
        return image_name

def get_balls_str_with_score(balls, image_name,withScore):
    '''返回球坐标对应的points文件中的每行字符串'''
    count = len(balls)
    balls_str = "{} ".format(count)
    
    for ball in balls:
      if not withScore:
        balls_str += "{id}({pos_x},{pos_y});".format(id=ball[0], pos_x=ball[2], pos_y=ball[3])
      else:
        score_string = "{:.2f}".format(ball[1])
        balls_str += "{id}({pos_x},{pos_y},{score});".format(id=ball[0], pos_x=ball[2], pos_y=ball[3],score=score_string)
    balls_str += " {}\n".format(get_frame_number(image_name))

    return balls_str


def cvt_dets(dets):
    '''转换识别结果：
    [(2, 0.87000257, 258, 308), (4, 0.8512376, 783, 297), (5, 0.67409694, 73, 274), (8, 0.6858853, 252, 152), (11, 0.7726598, 590, 421), (14, 0.85714597, 108, 219), (15, 0.7196537, 805, 482), (15, 0.56217825, 819, 498)]
    ==>
    3 2(258, 308);4(783, 297);5(73, 274);8(252, 152);11(590, 421);14(108, 219);15(805, 482);
    '''
    balls = []
    balls_deal = [] # 添加简单逻辑处理后的坐标
    for item in dets:
        id = item[0]
        score = item[1]
        pos_x = item[2]
        pos_y = item[3]
        one_new = (id, score, pos_x, pos_y)
        
        balls.append(one_new)

        # index = find_ball(balls, id)
        # if index == -1:
        #     balls.append(one_new)
        # else:
        #     if balls[index][1] < score: # 取score最大的那个
        #         balls[index] = one_new

        if score > 0.2:
            index = find_ball(balls_deal, id)
            if index == -1:
                balls_deal.append(one_new)
            else:
                if balls_deal[index][1] < score: # 取score最大的那个
                    balls_deal[index] = one_new
    balls.sort(key=takeFirst)
    balls_deal.sort(key=takeFirst)     
    return balls, balls_deal

def show_result(img, balls, image_name):
    for ball in balls:
        cv2.circle(img, (ball[2], ball[3]), 10, (0, 0, 255), 1, 0)
        point = ()
        if ball[0] > 9:
            point = (ball[2] - 11, ball[3] + 5)
        else:
            point = (ball[2] - 6, ball[3] + 5)
        cv2.putText(img, str(ball[0]), point, cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
        score = "{:.2f}".format(ball[1])
        cv2.putText(img, score, (ball[2] + 6, ball[3] - 5), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0))
    cv2.putText(img, image_name.split(".")[0], (700, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255))
    return img

def do_detect_process(image_dir, points_file_path, original_points_file_path, points_score_file_path,video_file_path, original_video_file_path):
    '''深度方案执行一个目录下图片的识别，图片是已经矫正后的图片'''
    print ("image_dir: %s\npoints_file: %s" %(image_dir, points_file_path))
    
    #videoWriter = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc('M','J','P','G'), 1, (960, 600))
    #videoWriter_ori = cv2.VideoWriter(original_video_file_path, cv2.VideoWriter_fourcc('M','J','P','G'), 1, (960, 600))

    fo = open(points_file_path, "w")
    #fo_ori = open(original_points_file_path, "w")
    #fo_score = open(points_score_file_path,"w")
    detector = BillBallDetect()
    filenames = os.listdir(image_dir)
    filenames.sort(key=lambda x:x[:-4]) # 文件夹文件排序

    for file in filenames:
        file_path = os.path.join(image_dir, file)
        img = cv2.imread(file_path)
        balls, balls_deal = cvt_dets(detector.detect(img))
        fo.write(get_balls_str_with_score(balls_deal, file,False))
        #fo_ori.write(get_balls_str_with_score(balls, file,False))
        #fo_score.write(get_balls_str_with_score(balls_deal, file,True))

        #frame = show_result(img, balls_deal, file)
        # cv2.imshow('img',frame)
        # cv2.waitKey(10)
        #videoWriter.write(frame)

        #frame_ori = show_result(img, balls, file)
        #videoWriter_ori.write(frame_ori)
    # cv2.destroyAllWindows()
    fo.close()
    #fo_ori.close()
    #fo_score.close()
    #videoWriter.release()
    #videoWriter_ori.release()

def auto_done(root_path):
    ''' 自动找到1目录作为图片目录去执行，points文件生成到同级别rec目录下
    '''
    for dirpath, dirnames, filenames in os.walk(root_path):
        if '1' in dirnames:
            image_dir = os.path.join(dirpath, '1')
            rec_dir = os.path.join(dirpath, 'rec')
            if not os.path.exists(rec_dir):
                os.mkdir(rec_dir)
            parent_dir_name = dirpath.split('/')[-1]
            points_file_path = os.path.join(rec_dir, "points__{parent}.txt".format(parent=parent_dir_name))
            original_points_file_path = os.path.join(rec_dir, "points__{parent}--original.txt".format(parent=parent_dir_name))
            video_file_path = os.path.join(rec_dir, "video__{parent}.avi".format(parent=parent_dir_name))
            original_video_file_path = os.path.join(rec_dir, "video__{parent}--original.avi".format(parent=parent_dir_name))
            '''生成带score文件
            '''
            points_score_file_path = os.path.join(rec_dir, "points__score__{parent}.txt".format(parent=parent_dir_name))

            do_detect_process(image_dir, points_file_path, original_points_file_path, points_score_file_path,video_file_path, original_video_file_path)

def main():
    #root_path = "/home/zealens/1210model"
    #root_path = "/media/zealens/4/dyq/20200203-higherrorrate"
    root_path = "/home/zealens/datas/20200203-mijiqiu-213pics-Test"
    auto_done(root_path)


if __name__ == "__main__":
    main()
