# anchor-free && mobile-net

## Introduction
Connect the mobilenet of Transformer model to the anchor free head of Yolov6 model. \
Exact approach is replacing the backbone 'Repvgg' of yolov6 with 'Mobilenetv1' 'Mobilenetv2' and 'Mobilevit' of transformer.

### Install

```shell
git clone https://github.com/meituan/YOLOv6
cd YOLOv6
pip install -r requirements.txt
```

### Training

Single GPU

```shell
python tools/train.py --batch 32 --conf configs/yolov6s.py --data data/coco.yaml --device 0
```

Multi GPUs (DDP mode recommended)

```shell
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --batch 256 --conf configs/yolov6s.py --data data/coco.yaml --device 0,1,2,3,4,5,6,7
```


<details>
<summary>Reproduce our results on COCO</summary>

For nano model
```shell
python -m torch.distributed.launch --nproc_per_node 4 tools/train.py \
									--batch 128 \
									--conf configs/yolov6n.py \
									--data data/coco.yaml \
									--epoch 400 \
									--device 0,1,2,3 \
									--name yolov6n_coco
```

For s/tiny model
```shell
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py \
									--batch 256 \
									--conf configs/yolov6s.py \ # configs/yolov6t.py
									--data data/coco.yaml \
									--epoch 400 \
									--device 0,1,2,3,4,5,6,7 \
									--name yolov6s_coco # yolov6t_coco
```

For mobilenetv1/v2/vit model
``` shell
python -m torch.distributed.launch --nproc_per_node 8 tools/trian.py \
                  --batch 256 \
                  --conf configs/yolov6s_mobile.py  \ # configs/yolov6s_mobilev2.py # configs/yolov6s_mobilevit_xxs.py
                  --data data/coco.yaml \
                  --epoch 400 \
                  --device 0,1,2,3,4,5,6,7 \
                  --name yolov6s_mobilev1_coco # yolov6s_mobilev2_coco # yolov6s_mobilevit_coco
```

For m/l model
```shell
# Step 1: Training a base model
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py \
									--batch 256 \
									--conf configs/yolov6m.py \ # configs/yolov6l.py
									--data data/coco.yaml \
									--epoch 300 \
									--device 0,1,2,3,4,5,6,7 \
									--name yolov6m_coco # yolov6l_coco
									
                                                                                      
# Step 2: Self-distillation training
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py \
									--batch 256 \ # 128 for distillation of yolov6l 
									--conf configs/yolov6m.py \ # configs/yolov6l.py
									--data data/coco.yaml \
									--epoch 300 \
									--device 0,1,2,3,4,5,6,7 \
									--distill \
									--pretrain_model_path runs/train/yolov6m_coco/weights/best_ckpt.pt \ # # yolov6l_coco
									--name yolov6m_coco # yolov6l_coco


```
</details>

- conf: select config file to specify network/optimizer/hyperparameters
- data: prepare [COCO](http://cocodataset.org) dataset, [YOLO format coco labels](https://github.com/meituan/YOLOv6/releases/download/0.1.0/coco2017labels.zip) and specify dataset paths in data.yaml
- make sure your dataset structure as follows:
```
├── coco
│   ├── annotations
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
│   ├── images
│   │   ├── train2017
│   │   └── val2017
│   ├── labels
│   │   ├── train2017
│   │   ├── val2017
│   ├── LICENSE
│   ├── README.txt
```
<details>
<summary>Resume training</summary>
If your training process is corrupted, you can resume training by
```shell
# multi GPU training.
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --resume
```
Your can also specify a checkpoint path to `--resume` parameter by
```shell
# remember to replace /path/to/your/checkpoint/path to the checkpoint path which you want to resume training.
--resume /path/to/your/checkpoint/path
```
</details>
	
### Evaluation

for yolov6 && mobilenet model
``` shell
python tools/eval.py --weights yolov6s.pt --data data/coco.yaml --device 0 --batch-size 32
```
for mobilevit model which need to use letterbox for resize all images to (640,640)
``` shell
python tools/eval.py --preprocess_img --weigths yolov6s_mobilevit_xxs.pt --data data/coco.yaml --device 0 --batch-size 32
```


### Inference

for yolov6 && mobilenet model
``` shell
python tools/infer.py --weights yolov6s.pt --source data/images
```
for mobilevit model which need to use letterbox for resize all images to (640,640)
```shell
python tools/infer.py --weights yolov6s_mobilevit_xxs.pt --source data/images --vit 
```


### Docker

Get the docker images from https://hub.docker.com/
``` shell
docker pull maoqijinwanzao3/pytorch:mobilevit-yolov6-py3.8-torch11.0-cu113
```

