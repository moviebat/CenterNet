cd src
# train
python test.py ctdet --exp_id coco_bill_mbv2 --dataset coco_bill  --resume --arch mbv2 --load_model ../exp/ctdet/coco_bill_mbv2/model_best.pth

cd ..
