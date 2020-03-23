
# coding: utf-8

# In[1]:


#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)


# In[2]:


annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print('Running demo for *%s* results.'% annType)


# In[3]:


#initialize COCO ground truth api
dataDir='/home/zealens/dyq/CenterNet/data/coco_bill/'
dataType='val2017'
annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
cocoGt=COCO(annFile)


# In[4]:


#initialize COCO detections api
resFile='/home/zealens/dyq/CenterNet/exp/ctdet/coco_bill_mbv2/results.json'
#resFile = resFile%(dataDir, prefix, dataType, annType)
cocoDt=cocoGt.loadRes(resFile)


# In[5]:


import json
dts = json.load(open(resFile,'r'))
imgIds = [imid['image_id'] for imid in dts]
imgIds = sorted(list(set(imgIds)))
del dts


# In[6]:


# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.params.catIds = [16]
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

