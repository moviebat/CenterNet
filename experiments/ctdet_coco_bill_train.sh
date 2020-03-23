cd src
# train
python main.py ctdet --exp_id coco_bill_mbv2 --batch_size 4  --lr 1.25e-4  --num_epochs 1  --gpus 0 --arch mbv2

cd ..
