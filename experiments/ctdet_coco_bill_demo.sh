cd src
# train
python demo.py ctdet --demo ../images_bill  --load_model ../exp/ctdet/coco_bill_mbv2/model_best.pth  --debug 2 --arch mbv2

cd ..
