# CenterNet
fork from https://github.com/xingyizhou/CenterNet

## 新功能
增加了Mobilenetv2模型
结合了台球的训练集合

## cocoapi
进入PythonAPI的路径（这里以ubuntu系统为例）：
cd coco/PythonAPI
直接编译：
make
没有报错说明编译成功。需要注意的是，如果你的机器上有多个虚拟环境A，B，C…那么比较稳妥的方式是先激活你需要的环境，再执行上述命令，这样可以避免冲突。
在.ipynb 文件所在的目录下打开一个终端，然后输入：
jupyter nbconvert --to script *.ipynb 
就能把当前文件夹下面的所有的.ipynb文件转化为.py文件

gedit pycocoEvalDemo.py
在第3个block处修改你的ground truth路径
#initialize COCO ground truth api
dataDir='/path/to/your/annotation/json/file/'
dataType='val2017'
annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
cocoGt=COCO(annFile)
在第4个block处修改你的结果文件路径
#initialize COCO detections api

resFile='/path/to/your/result/json/file/'
cocoDt=cocoGt.loadRes(resFile)
修改第5个block，使api可以遍历你所有的结果
原来的代码块里只遍历了前100个结果，所以这里首先将原代码注释掉，再加入下面代码
import json 
dts = json.load(open(resFile,'r'))
imgIds = [imid['image_id'] for imid in dts]
imgIds = sorted(list(set(imgIds)))
del dts
修改第6个block，使api可以仅检测某一类别的结果
原代码：
running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
这段代码默认是对所有类别的结果进行评估，为了使它可以对特定类别进行评估，我们需要往其中加入一行：

cocoEval.params.catIds = [1]

最后代码块如下：

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.params.catIds = [1] # 1代表’person’类，你可以根据需要增减类别
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
经过这样的处理，我们就可以对特定类别的目标进行评估。

## 训练
开始训练MobileNetV2  
python main.py ctdet --exp_id coco_bill_mbv2 --batch_size 12  --lr 1.25e-4  --num_epochs 90  --gpus 0 --arch mbv2

## 测试模型效果
验证新模型的效果  
python demo.py ctdet --demo ../images_bill/frame_00162000.bmp  --load_model ../exp/ctdet/coco_bill_mbv2/model_best.pth  --debug 2 --arch mbv2

发现之有一张才能跑，如果是目录则跑不通，修改demo.py里的image_ext，增加图片后缀BMP
image_ext = ['jpg', 'jpeg', 'png', 'webp', 'bmp']

然后用目录就可以实现多张的目标检测了

python demo.py ctdet --demo ../images_bill  --load_model ../exp/ctdet/coco_bill_mbv2/model_best.pth  --debug 2 --arch mbv2

## 测试模型性能
测试新模型的性能  
python test.py ctdet --exp_id coco_bill_mbv2 --dataset coco_bill  --resume --arch mbv2 --load_model ../exp/ctdet/coco_bill_mbv2/model_best.pth