from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from functools import wraps
from progress.bar import Bar
import torch
from memory_profiler import profile
from self_help import vis_result, get_dets
from external.nms import soft_nms
from opts import opts
from utils.utils import AverageMeter
from detectors.ctdet import CtdetDetector


class BillBallDetect(object):
  def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %(function.__name__, str(t1-t0)))
        return result
    return function_timer

  @profile
  def __init__(self):
    opt = opts().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.mean = np.array([0.08948476, 0.35924495,0.05387309],
                dtype=np.float32).reshape(1,1,3)
    opt.std = np.array([0.07614725, 0.13400481,0.09814119],
                dtype=np.float32).reshape(1,1,3)
    #opt.mean = np.array([0.40789654, 0.44719302, 0.47026115],
    #               dtype=np.float32).reshape(1, 1, 3)
    #opt.std  = np.array([0.28863828, 0.27408164, 0.27809835],
    #               dtype=np.float32).reshape(1, 1, 3)
    opt.num_classes = 16
    opt.input_h = 320
    opt.input_w = 512
    opt.output_h = opt.input_h // opt.down_ratio
    opt.output_w = opt.input_w // opt.down_ratio
    opt.input_res = max(opt.input_h, opt.input_w)
    opt.output_res = max(opt.output_h, opt.output_w)
    opt.heads = {'hm': opt.num_classes, 'wh': 2, 'reg': 2}
    opt.load_model = "/home/zealens/dyq/CenterNet/models/model_best.pth"
    print('heads', opt.heads)
    print(opt)
    self.opt = opt
    self.detector = CtdetDetector(opt)
  @profile
  def detect(self, org_img):
    results = {}
    tictic =time.time() 
    ret = self.detector.run(org_img)
    toctoc =time.time()
    dets = get_dets(ret['results'])
    #frame = vis_result(img, ret['results'])
    fps = 1.0/(toctoc-tictic)
    #print("FPS: "+str(fps))
    return dets


@profile
def test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  opt.std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
  opt.num_classes = 16
  opt.input_h = 320
  opt.input_w = 512
  opt.output_h = opt.input_h // opt.down_ratio
  opt.output_w = opt.input_w // opt.down_ratio
  opt.input_res = max(opt.input_h, opt.input_w)
  opt.output_res = max(opt.output_h, opt.output_w)
  opt.heads = {'hm': opt.num_classes, 'wh':2, 'reg': 2}
  print('heads', opt.heads)
  print(opt)
  detector = CtdetDetector(opt)
  results = {}
  bar = Bar('{}'.format("test mode"), max=1000)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  video_path = "/home/zealens/dyq/CenterNet/data/test_video.avi"
  cap = cv2.VideoCapture(video_path)
  videoWriter = cv2.VideoWriter("out_video.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 1, (960, 600))
  ind = 0
  while True:
    ret,img=cap.read()
    tictic =time.time() 
    if ret == False:
        break
    ret = detector.run(img)
    toctoc =time.time()
    frame = vis_result(img, ret['results'])
    videoWriter.write(frame) 
    fps = 1.0/(toctoc-tictic)
    Bar.suffix = '[{0}]|Tot: {total:} |ETA: {eta:} |FPS: {fps:}'.format(
                   ind, total=bar.elapsed_td, eta=bar.eta_td,fps=fps)
    ind+=1
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()
  cap.release()
  videoWriter.release()
  bar.finish()

if __name__ == '__main__':
  opt = opts().parse()
  test(opt)
  

