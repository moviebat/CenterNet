4、开始训练MobileNetV2

python main.py ctdet --exp_id coco_bill_mbv2 --batch_size 12  --lr 1.25e-4  --num_epochs 90  --gpus 0 --arch mbv2


5、验证新模型的效果

python demo.py ctdet --demo ../images_bill/frame_00162000.bmp  --load_model ../exp/ctdet/coco_bill_mbv2/model_best.pth  --debug 2 --arch mbv2

发现之有一张才能跑，如果是目录则跑不通，修改demo.py里的image_ext，增加图片后缀BMP
image_ext = ['jpg', 'jpeg', 'png', 'webp', 'bmp']

然后用目录就可以实现多张的目标检测了

python demo.py ctdet --demo ../images_bill  --load_model ../exp/ctdet/coco_bill_mbv2/model_best.pth  --debug 2 --arch mbv2

6、测试新模型的性能

python test.py ctdet --exp_id coco_bill_mbv2 --dataset coco_bill  --resume --arch mbv2 --load_model ../exp/ctdet/coco_bill_mbv2/model_best.pth