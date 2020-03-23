cd src

# auto_detect_tools
python auto_select_tools.py --exp_id coco_dla_bill  --dataset coco_bill  --resume  --arch mbv2 --load_model /home/zealens/dyq/infer/billball/models/model_best.pth --pytorch_mode 

# test_ball_detect
#python test_ball_detect.py --exp_id coco_dla_bill  --dataset coco_bill  --resume  --arch mbv2 --load_model /home/zealens/dyq/infer/billball/models/model_best.pth --pytorch_mode 

cd ..
