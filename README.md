# Advancing Dataset Distillation: Efficiency, Scalability, and Diverse AI Task Applications

<p align="center">
    <img width="850" src="overview.png"/>
</p>

## Installation

Python 3.9 is required.

```bash
pip install -r requirements.txt
```


## Datasets

Download the Flickr30K 
[[Train](https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_train.json)]
[[Val](https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json)]
[[Test](https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json)]
[[Images](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)]
and MS-COCO
[[Train](https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json)]
[[Val](https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json)]
[[Test](https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json)]
[[Images](https://cocodataset.org/#download)]
datasets. 

Place the downloaded images and annotation JSON files as follows:

```
./data/datasets/
├── Flickr30k/
│   ├── flickr30k-images/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   ├── flickr30k_train.json
│   ├── flickr30k_val.json
│   └── flickr30k_test.json
└── COCO/
    ├── train2014/
    ├── val2014/
    ├── test2014/
    ├── coco_karpathy_train.json
    ├── coco_karpathy_val.json
    └── coco_karpathy_test.json
```

## Run

### Flickr30K
To distill the Flickr30K dataset into 100/300/500 pairs and evaluate the distilled dataset, use the following scripts:

```
bash Flickrtrain.sh
```

### MS-COCO
To distill the MS-COCO dataset into 100/300/500 pairs and evaluate the distilled dataset, use the following scripts:

```
bash cocotrain.sh
```

## Acknowledgement
The implementation and experiments are built upon the code of [LoRS](https://github.com/silicx/LoRS_Distill) and [PDS](https://github.com/junhyeok9712/PDS).