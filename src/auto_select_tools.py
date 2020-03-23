#/usr/env/bin python3

import sys
import os,shutil
import numpy as np
import cv2

from billdetect import BillBallDetect

# 获取列表的第一个元素
def takeFirst(elem):
    return elem[0]

# 获取列表的第二个元素
def takeSecond(elem):
    return elem[1]

def moveFile(srcfile,dstfile):
    result = False
    if os.path.isfile(srcfile):
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        result = True
    return result

def get_balls_str(balls, image_name):
    '''返回球坐标对应的points文件中的每行字符串'''
    count = len(balls)
    balls_str = "{} ".format(count)
    for ball in balls:
        balls_str += "{id}({pos_x},{pos_y});".format(id=ball[0], pos_x=ball[2], pos_y=ball[3])
    balls_str += " {}\n".format(get_frame_number(image_name))
    print ("{img}----{b}".format(img=get_frame_number(image_name), b=balls))
    return balls_str

def cvt_dets(dets):
    '''对识别结果筛选后按球号排序
    [(2, 0.87000257, 258, 308), (4, 0.8512376, 783, 297), (5, 0.67409694, 73, 274), (8, 0.6858853, 252, 152), (11, 0.7726598, 590, 421), (14, 0.85714597, 108, 219), (15, 0.7196537, 805, 482), (15, 0.56217825, 819, 498)]
    ==>
    [(2, 0.87000257, 258, 308);(2, 0.89000257, 783, 297);(5, 0.67409694, 73, 274);(8, 0.5858853, 252, 152);(8, 0.6058853, 590, 421);(11, 0.7726598, 590, 421);(14, 0.85714597, 590, 421);(15, 0.8196537, 805, 482);(15, 0.86217825, 819, 498);
    '''
    balls = []
    for item in dets:
        id = item[0]
        score = item[1]
        pos_x = item[2]
        pos_y = item[3]
        one_new = (id, score, pos_x, pos_y)
        
        balls.append(one_new)

    #print("1111111111")
    #print(balls)
    balls.sort(key=takeFirst)
    #print("2222222222")
    #print(balls)

    return balls

def isSelected(balls):
    '''判断是否是需要的图片，1是出现2个>0.2的黑8,2是出现2个>0.3的球
    [(2, 0.87000257, 258, 308);(2, 0.89000257, 783, 297);(5, 0.67409694, 73, 274);(8, 0.5858853, 252, 152);(8, 0.6058853, 590, 421);(11, 0.7726598, 590, 421);(14, 0.85714597, 590, 421);(15, 0.8196537, 805, 482);(15, 0.86217825, 819, 498)];
    '''
    result = False
    index = 0
    lastBall = 0
    lastScore = 0
    
    for item in balls:
        id = item[0]
        score = item[1]
        
        if index == 0:
           lastBall = id
           lastScore = score
        else:
           '''出现连续2个黑8,而且都大于0.2,说明异物干扰明显
           '''
           if lastBall == 8 and id ==8:
               result = True
               break
           '''2个同样的球，均大于0.3,说明串球
           '''
           if lastBall ==id and lastScore > 0.3 and score > 0.3:
               result = True
               break
           lastBall = id
           lastScore = score
        index = index + 1
        
    return result

def do_detect_process(image_dir, dst_dir):
    '''深度方案执行一个目录下图片的识别，图片是已经矫正后的图片'''
    
    detector = BillBallDetect()
    filenames = os.listdir(image_dir)
    filenames.sort(key=lambda x:x[:-4]) # 文件夹文件排序
    selectedPicsCount = 0

    for file in filenames:
        file_path = os.path.join(image_dir, file)
        img = cv2.imread(file_path)
        balls = []
        balls = cvt_dets(detector.detect(img))
        if isSelected(balls):
           selectedPicsCount = selectedPicsCount + 1
           destFilePath = os.path.join(dst_dir, file)
           tmpFilePath, tmpFileName = os.path.split(file_path)
           moveFile(file_path,destFilePath)
    print(selectedPicsCount)

def auto_done(root_path):
    ''' 自动找到1目录作为图片目录去执行，将图片文件移动到新的目标目录下的1文件夹下
    '''
    root_path.rstrip('/')
    root_dir_name = root_path.split('/')[-1]
    new_root_dir_name = root_dir_name + "_object"

    for dirpath, dirnames, filenames in os.walk(root_path):
        if '1' in dirnames:
            image_dir = os.path.join(dirpath, '1')
            dst_dir = image_dir.replace(root_dir_name, new_root_dir_name, 1)

            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            print(image_dir)
            print(dst_dir)

            do_detect_process(image_dir, dst_dir)

def main():
    root_path = "/home/zealens/datas/20200203-mijiqiu-213pics-Test"
    #root_path = "/media/zealens/4/20200117-select"
    auto_done(root_path)


if __name__ == "__main__":
    main()
