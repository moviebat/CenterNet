#coding:utf-8
import cv2
import os
import numpy as np

def remove_low(dets, thresh):
    outs = []
    for one in dets:
        if one["score"] >= thresh:
            outs.append(one)
    return outs
def remove_unusual_bbox(dets):
    outs = []
    for one in dets:
        width = one["bbox"][2]-one["bbox"][0]
        height = one["bbox"][3]-one["bbox"][1]
        per = width/height
        if per>1.2 or per<0.8:
            continue
        if width > 23 or height > 23:
            continue
        if width < 16.5 or height < 16.5:
            continue
        outs.append(one)
    return outs

def bill_nms(dets, thresh):
    dets_sorted = sorted(dets, key=lambda x:x["score"], reverse = True)
    keep = []
    while len(dets_sorted) > 0:
        keep.append(dets_sorted[0])
        cur_class = dets_sorted[0]["category_id"]
        if len(dets_sorted)==1:
            break
        areas =  [(dets_sorted[i]["bbox"][2]-dets_sorted[i]["bbox"][0])*(dets_sorted[i]["bbox"][3]-dets_sorted[i]["bbox"][1]) for i in range(1, len(dets_sorted))]
        xx1s = np.maximum(dets_sorted[0]["bbox"][0], [dets_sorted[i]["bbox"][0] for i in range(1, len(dets_sorted))])
        yy1s = np.maximum(dets_sorted[0]["bbox"][1], [dets_sorted[i]["bbox"][1] for i in range(1, len(dets_sorted))])
        xx2s = np.minimum(dets_sorted[0]["bbox"][2], [dets_sorted[i]["bbox"][2] for i in range(1, len(dets_sorted))])
        yy2s = np.minimum(dets_sorted[0]["bbox"][3], [dets_sorted[i]["bbox"][3] for i in range(1, len(dets_sorted))])
        w = np.maximum(0.0, xx2s - xx1s + 1)
        h = np.maximum(0.0, yy2s - yy1s + 1)
        inter = w * h
        ovr = inter / (areas - inter + (dets_sorted[0]["bbox"][2]-dets_sorted[0]["bbox"][0])*(dets_sorted[0]["bbox"][3]-dets_sorted[0]["bbox"][1]))
        inds = np.where(ovr <= thresh)[0]
        dets_nmsed = [dets_sorted[inds[i]+1] for i in range(len(inds))]
        dets_sorted = []
        for one in dets_nmsed:
            if one["category_id"] == cur_class:
                continue
            dets_sorted.append(one)
    return keep

def post_process(dets):
    dets_now = remove_low(dets, 0.2)
    dets_now = remove_unusual_bbox(dets_now)
    dets_now = bill_nms(dets_now, 0.5)
    return dets_now


def cvt_result(dets):
    dets_sorted = sorted(dets, key=lambda x:x["category_id"], reverse = False)
    balls = []
    for item in dets_sorted:
        id = item['category_id'] - 1
        score = item['score']

        # TEST: To test whether have
        if (item['bbox'][2] - item['bbox'][0] < 19) or (item['bbox'][3] - item['bbox'][1] < 19):
            #print (f"small box found ---- {item}")
            continue
        
        pos_x = int((item['bbox'][0] + item['bbox'][2]) * 0.5)
        pos_y = int((item['bbox'][1] + item['bbox'][3]) * 0.5)
        one_new = (id, score, pos_x, pos_y)
        balls.append(one_new)
    return balls

def vis_result(img , result):
    dets = []
    for cls_id in result:
        for one in result[cls_id]:
            one_new = {}
            one_new["category_id"] = cls_id
            one_new["score"] = one[4]
            one_new["bbox"] = one[0:4].tolist()
            dets.append(one_new)
    dets = post_process(dets)
    for bb in dets:
        cv2.rectangle(img,(int(bb["bbox"][0]), int(bb["bbox"][1])), (int(bb["bbox"][2]), int(bb["bbox"][3])), (0,0,255))
        cv2.putText(img, str(bb["category_id"]-1), (int(bb["bbox"][0]), int(bb["bbox"][1]+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
    return img
def get_dets(result):
    dets = []
    for cls_id in result:
        for one in result[cls_id]:
            one_new = {}
            one_new["category_id"] = cls_id
            one_new["score"] = one[4]
            one_new["bbox"] = one[0:4].tolist()
            dets.append(one_new)
    dets = post_process(dets)
    dets = cvt_result(dets)
    return dets
